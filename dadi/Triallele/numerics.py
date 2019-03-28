import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapz
from scipy.special import gamma
from numpy import newaxis as nuax
from dadi import Numerics
from dadi.Spectrum_mod import Spectrum
import dadi
import math
import pickle

from dadi.Triallele.TriSpectrum_mod import TriSpectrum

def grid_dx(x):
    """
    We use uniform grids in x, using np.linspace(0,1,numpts)
    Grid spacing Delta, which is halved at the first and last grid points.
    """
    return (np.concatenate((np.diff(x),np.array([0]))) + np.concatenate((np.array([0]),np.diff(x))))/2

def grid_dx_2d(x,dx):
    """
    The two dimensional grid spacing over the domain.
    Grid points lie along the diagonal boundary, and Delta for those points is halved
    """
    DXX = dx[:,nuax]*dx[nuax,:]
    for ii in range(len(x)):
        DXX[ii,len(x)-ii-1] *= 1./2
    return DXX

def int2(DXX,U):
    """
    Integrate the density function over the domain
    DXX - two dimensional grid
    U - density function
    """
    return np.sum(DXX*U)

def domain(x):
    """
    Constructs a matrix with the same dimension as the density function discretization, a 1 indicates that the corresponding point is inside the triangular domain or on the boundary, while a 0 indicates that point falls outside the domain
    """
    tol = 1e-12
    U01 = np.ones((len(x),len(x)))
    XX = x[:,nuax] + x[nuax,:]
    U01[np.where(XX > 1+tol)] = 0
    return U01

### transition matrices

def transition1(x, dx, U01, sig1, sig2):
    """
    Implicit transition matrix for the ADI components of the discretization of the diffusion
    Time scaled by 2N, with variance and mean terms x(1-x) and \sigma*x(1-x), resp.
    Store the tridiagonal elements of the matrices, which need to be adjusted by I + dt*P, where I is the identity matrix
    x - grid
    dx - grid spacing
    U01 - domain markers
    sig1/2 - population scaled selection coefficients
    nu - relative population size
    """
    # XXX: Note that this function has been Cythonized. Below, an attempt is made to import the Cythonized version so that it will
    #      be automatically used instead of this version, if the user has compiled it.
    print("using numpy, not cython")
    PV = np.zeros((len(x),3,len(x)))
    PM = np.zeros((len(x),3,len(x)))
    for jj in range(len(x)):
        A = np.zeros((len(x),len(x)))
        if jj > 0:
            V = x*(1-x)
            V[np.max(np.where(U01[:,jj] == 1))] = 0
            for ii in np.where(U01[:,jj] == 1)[0][:-1]:
                if ii == 0:
                    A[ii,ii] =  - 1/(2*dx[ii]) * ( -V[ii]/(x[ii+1]-x[ii]) )
                    A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) )
                elif ii == np.where(U01[:,jj] == 1)[0][:-1][-1]:
                    A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) )
                    A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) )
                else:
                    A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) )
                    A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) -V[ii]/(x[ii+1]-x[ii]) )
                    A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) )
        if jj == 0:
            V = x*(1-x)
            for ii in range(len(x)):
                if ii == 0:
                    A[ii,ii] =  - 1/(2*dx[ii]) * ( -V[ii]/(x[ii+1]-x[ii]) )
                    A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) )
                elif ii == len(x)-1:
                    A[ii,ii-1] = - 1/(2*dx[ii])*2 * ( V[ii-1]/(x[ii]-x[ii-1]) )
                    A[ii,ii] = - 1/(2*dx[ii])*2 * ( -V[ii]/(x[ii]-x[ii-1]) )
                else:
                    A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) )
                    A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) -V[ii]/(x[ii+1]-x[ii]) )
                    A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) )
        
        PV[jj,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
        PV[jj,1,:] = np.diagonal(A)
        PV[jj,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))

        A = np.zeros((len(x),len(x)))
        x2 = x[jj]
        sig_new = sig1*(1-x-x2)/(1-x) + (sig1-sig2)*x2/(1-x)
        sig_new[-1] = 0
        M = sig_new*x*(1-x)
        M[np.where(U01[:,jj] == 1)[0][-1]] = 0
        for ii in np.where(U01[:,jj] == 1)[0]:
            if ii == 0:
                A[ii,ii] += 1/dx[ii] * ( M[ii] ) / 2
                A[ii,ii+1] += 1/dx[ii] * ( M[ii+1] ) / 2
            elif ii == np.where(U01[:,jj] == 1)[0][-1]:
                A[ii,ii-1] += 1/dx[ii] * 2 * ( - M[ii-1] ) / 2
                A[ii,ii] += 1/dx[ii] * 2 * ( - M[ii] ) / 2
            else:
                A[ii,ii-1] += 1/dx[ii] * ( - M[ii-1] ) / 2
                A[ii,ii] += 0 #1/dx[ii] * ( M[ii] - M[ii-1] ) / 2
                A[ii,ii+1] += 1/dx[ii] * ( M[ii+1] ) / 2

        PM[jj,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
        PM[jj,1,:] = np.diagonal(A)
        PM[jj,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    return PV,PM

def transition2(x, dx, U01, sig1, sig2):
    """
    Implicit transition matrix for the ADI components of the discretization of the diffusion
    Time scaled by 2N, with variance and mean terms x(1-x) and \sigma*x(1-x), resp.
    Store the tridiagonal elements of the matrices, which need to be adjusted by I + dt*P, where I is the identity matrix
    x - grid
    dx - grid spacing
    U01 - domain markers
    sig1/2 - population scaled selection coefficients
    nu - relative population size
    """
    # XXX: Note that this function has been Cythonized. Below, an attempt is made to import the Cythonized version so that it will
    #      be automatically used instead of this version, if the user has compiled it.
    PV = np.zeros((len(x),3,len(x)))
    PM = np.zeros((len(x),3,len(x)))
    for ii in range(len(x)):
        A = np.zeros((len(x),len(x)))
        if ii > 0:
            V = x*(1-x)
            V[np.max(np.where(U01[ii,:] == 1))] = 0
            for jj in np.where(U01[ii,:] == 1)[0][:-1]:
                if jj == 0:
                    A[jj,jj] =  - 1/(2*dx[jj]) * ( -V[jj]/(x[jj+1]-x[jj]) )
                    A[jj,jj+1] = - 1/(2*dx[jj]) * ( V[jj+1]/(x[jj+1]-x[jj]) )
                elif jj == np.where(U01[ii,:] == 1)[0][:-1][-1]:
                    A[jj,jj-1] = - 1/(2*dx[jj]) * ( V[jj-1]/(x[jj]-x[jj-1]) )
                    A[jj,jj] = - 1/(2*dx[jj]) * ( -V[jj]/(x[jj]-x[jj-1]) )
                else:
                    A[jj,jj-1] = - 1/(2*dx[jj]) * ( V[jj-1]/(x[jj]-x[jj-1]) )
                    A[jj,jj] = - 1/(2*dx[jj]) * ( -V[jj]/(x[jj]-x[jj-1]) -V[jj]/(x[jj+1]-x[jj]) )
                    A[jj,jj+1] = - 1/(2*dx[jj]) * ( V[jj+1]/(x[jj+1]-x[jj]) )
        if ii == 0:
            V = x*(1-x)
            for jj in range(len(x)):
                if jj == 0:
                    A[jj,jj] =  - 1/(2*dx[jj]) * ( -V[jj]/(x[jj+1]-x[jj]) )
                    A[jj,jj+1] = - 1/(2*dx[jj]) * ( V[jj+1]/(x[jj+1]-x[jj]) )
                elif jj == len(x)-1:
                    A[jj,jj-1] = - 1/(2*dx[jj])*2 * ( V[jj-1]/(x[jj]-x[jj-1]) )
                    A[jj,jj] = - 1/(2*dx[jj])*2 * ( -V[jj]/(x[jj]-x[jj-1]) )
                else:
                    A[jj,jj-1] = - 1/(2*dx[jj]) * ( V[jj-1]/(x[jj]-x[jj-1]) )
                    A[jj,jj] = - 1/(2*dx[jj]) * ( -V[jj]/(x[jj]-x[jj-1]) -V[jj]/(x[jj+1]-x[jj]) )
                    A[jj,jj+1] = - 1/(2*dx[jj]) * ( V[jj+1]/(x[jj+1]-x[jj]) )
        
        PV[ii,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
        PV[ii,1,:] = np.diagonal(A)
        PV[ii,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))

        A = np.zeros((len(x),len(x)))
        x1 = x[ii]
        sig_new = sig2*(1-x-x1)/(1-x) + (sig2-sig1)*x1/(1-x)
        sig_new[-1] = 0
        M = sig_new*x*(1-x)
        M[np.where(U01[ii,:] == 1)[0][-1]] = 0
        for jj in np.where(U01[ii,:] == 1)[0]:
            if jj == 0:
                A[jj,jj] += 1/dx[jj] * ( M[jj] ) / 2
                A[jj,jj+1] += 1/dx[jj] * ( M[jj+1] ) / 2
            elif jj == np.where(U01[ii,:] == 1)[0][-1]:
                A[jj,jj-1] += 1/dx[jj] * 2 * ( - M[jj-1] ) / 2
                A[jj,jj] += 1/dx[jj] * 2 * ( - M[jj] ) / 2
            else:
                A[jj,jj-1] += 1/dx[jj] * ( - M[jj-1] ) / 2
                A[jj,jj] += 0 # 1/dx[jj] * ( M[jj] - M[jj-1] ) / 2
                A[jj,jj+1] += 1/dx[jj] * ( M[jj+1] ) / 2
        
        PM[ii,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
        PM[ii,1,:] = np.diagonal(A)
        PM[ii,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    return PV,PM

def transition12(x, dx, U01):
    """
    Transition matrix for the covariance term of the diffusion operator, with term D_{xy} (-x*y*phi)
    As with the ADI components, final transition matrix is given by I + dt/nu*P
    x - grid
    dx - grid spacing
    U01 - domain markers
    """        
    # XXX: Note that this function has been Cythonized. Below, an attempt is made to import the Cythonized version so that it will
    #      be automatically used instead of this version, if the user has compiled it.
    C = lil_matrix((len(x)**2,len(x)**2))
    for ii in range(len(x)-1)[:-1]:
        for jj in range(len(x)-1)[:-1]:
            if U01[ii+2,jj+2] == 1 or U01[ii+1,jj+2] == 1 or U01[ii+2,jj+1] == 1:
                if ii+1 < len(x) and jj+1 < len(x) and U01[ii+1,jj+1] == 1:
                    C[ii*len(x)+jj,(ii+1)*len(x)+(jj+1)] += 1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii+1]*x[jj+1])
                    
                if ii+1 < len(x) and jj-1 >= 0 and U01[ii+1,jj-1] == 1:
                    C[ii*len(x)+jj,(ii+1)*len(x)+(jj-1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii+1]*x[jj-1])
                
                if ii-1 >= 0 and jj+1 < len(x) and U01[ii-1,jj+1] == 1:
                    C[ii*len(x)+jj,(ii-1)*len(x)+(jj+1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj+1])
                    
                if ii-1 >= 0 and jj-1 >= 0 and U01[ii-1,jj-1] == 1:
                    C[ii*len(x)+jj,(ii-1)*len(x)+(jj-1)] += 1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj-1])
            
            elif U01[ii+1,jj+1] == 1 or U01[ii+1,jj] == 1 or U01[ii,jj+1] == 1:
                if ii+1 < len(x) and jj-1 >= 0 and U01[ii+1,jj-1] == 1:
                    C[ii*len(x)+jj,(ii+1)*len(x)+(jj-1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii+1]*x[jj-1]) / 2
                
                if ii-1 >= 0 and jj+1 < len(x) and U01[ii-1,jj+1] == 1:
                    C[ii*len(x)+jj,(ii-1)*len(x)+(jj+1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj+1]) / 2
                    
                if ii-1 >= 0 and jj-1 >= 0 and U01[ii-1,jj-1] == 1:
                    C[ii*len(x)+jj,(ii-1)*len(x)+(jj-1)] += 1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj-1])
    ii = 0
    jj = len(x)-2
    C[ii*len(x)+jj,(ii+1)*len(x)+(jj-1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii+1]*x[jj-1]) / 2
    ii = len(x)-2
    jj = 0
    C[ii*len(x)+jj,(ii-1)*len(x)+(jj+1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj+1]) / 2
    return C

def transition1D(x, dx, sig):
    """
    transition matrix for one dimensional integration
    x - grid
    dx - grid spacing
    dt - timestep for integration
    sig - selection coefficient
    nu - relative population size
    """
    # XXX: Note that this function has been Cythonized. Below, an attempt is made to import the Cythonized version so that it will
    #      be automatically used instead of this version, if the user has compiled it.
    PV = np.zeros((len(x),len(x)))
    PM = np.zeros((len(x),len(x)))
    for ii in range(len(x)):
        if ii == 0:
            PV[ii,ii] = 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            PV[ii,ii+1] = -1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii])
            PM[ii,ii+1] = sig / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
        elif ii == len(x) - 1:
            PV[ii,ii-1] = - 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1])
            PM[ii,ii-1] = sig / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            PV[ii,ii] = 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1])
        else:
            PV[ii,ii-1] = - 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1])
            PM[ii,ii-1] = - sig / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            PV[ii,ii] = 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1]) + 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            PV[ii,ii+1] = - 1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii])
            PM[ii,ii+1] = sig / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
    return PV,PM

transition1D_cache = {}
def cached_transition1D(numpts,sig):
    key = (numpts,sig)
    try:
        return transition1D_cache[key]
    except KeyError:
        pass
    x = np.linspace(0,1,numpts+1)
    dx = grid_dx(x)
    V,M = transition1D(x,dx,sig)
    transition1D_cache[key] = [V,M]
    return transition1D_cache[key]

def remove_diag_density_weights_nonneutral(x,dt,nu,sig1,sig2):
    """
    Numerically determine the amount of density that should be lost to the diagonal boundary.
    Numerically integrate 1D array with initial point mass at z0, where z0 is the frequency x+y, integrated for time step dt.
    We then check the fraction of density that is lost to z=1.
    If sig1 or sig2 are nonzero, estimate the selection pressure on z as sig = sig1*x/(x+y) + sig2*y/(x+y)
    x - one dimensional grid of domain
    dt - time step of integration
    nu - relative population size
    sig1,sig2 - selection coefficients
    """
    dx = grid_dx(x)
    P = np.zeros((len(x),len(x)))
    for ii in range(len(x)):
        for jj in range(len(x)):
            if x[ii]+x[jj] < 1.0 and x[ii]+x[jj] > .75:
                sig = sig1*x[ii]/(x[ii]+x[jj]) + sig2*x[jj]/(x[ii]+x[jj])
                V,M = cached_transition1D(len(x)-1,sig)
                P1D = np.eye(len(x)) + dt*(V/nu+M)
                y = np.zeros(len(x))
                y[ii+jj] = 1./dx[ii+jj]
                y = advance1D(y,P1D)
                prob = y[-1]*dx[-1]
                P[ii,jj] = prob
                if ii+jj == len(x)-2:
                    if ii==1:
                        V,M = cached_transition1D(len(x)-1,sig1)
                        P1D = np.eye(len(x)) + dt*(V/nu+M)
                        y = np.zeros(len(x))
                        y[ii] = 1./dx[ii]
                        y = advance1D(y,P1D)
                        prob2 = y[0]*dx[0]
                        P[ii,jj] += prob2
                    elif jj==1:
                        V,M = cached_transition1D(len(x)-1,sig2)
                        P1D = np.eye(len(x)) + dt*(V/nu+M)
                        y = np.zeros(len(x))
                        y[jj] = 1./dx[jj]
                        y = advance1D(y,P1D)
                        prob2 = y[0]*dx[0]
                        P[ii,jj] += prob2
    
    P[:,0] = 0
    P[0,:] = 0
    return P

def move_density_to_bdry(x,phi,P):
    """
    P tells us how much should be removed, by multiplying phi*P
    Take that density and instead of deleting it, move straight to boundary
    P - stores how much denstity from each grid point should be moved to diagonal
    """
    for ii in range(len(x)):
        for jj in range(len(x)):
            if P[ii,jj] == 0:
                continue
            else:
                amnt = P[ii,jj]
                s = ii+jj
                if ii == 1 and jj == len(x)-3:
                    phi[ii+1,jj] += phi[ii,jj]*amnt/4. * 2
                    phi[ii,jj+1] += phi[ii,jj]*amnt/4. * 2
                    phi[ii-1,jj] += phi[ii,jj]*amnt/2 * 2
                    phi[ii,jj] *= (1-amnt)
                elif ii == len(x)-3 and jj == 1:
                    phi[ii+1,jj] += phi[ii,jj]*amnt/4. * 2
                    phi[ii,jj+1] += phi[ii,jj]*amnt/4. * 2
                    phi[ii,jj-1] += phi[ii,jj]*amnt/2 * 2
                    phi[ii,jj] *= (1-amnt)
                elif (len(x)-1-s) % 2 == 1:
                    # split between two points
                    dist = (len(x)-1-s) // 2
                    phi[ii+dist+1,jj+dist] += phi[ii,jj]*amnt/2. * 2
                    phi[ii+dist,jj+dist+1] += phi[ii,jj]*amnt/2. * 2
                    phi[ii,jj] *= (1-amnt)
                else:
                    # straight to boundary grid point
                    dist = (len(x)-1-s) // 2
                    phi[ii+dist,jj+dist] += phi[ii,jj]*amnt * 2
                    phi[ii,jj] *= (1-amnt)

    return phi

### forward integration methods

def advance_adi(U,U01,P1,P2,x,ii):
    """
    Integrate the ADI components forward in time, alternating which direction occurs first
    U - density function
    U01 - stores which points are in the domain
    P1,P2 - transition matrices
    ii - count of integration step
    """
    if np.mod(ii,2) == 0:
        for jj in range(len(x)):
            if np.sum(U01[:,jj]) > 1:
                U[:,jj] = dadi.Integration.tridiag.tridiag(P1[jj,0,:],P1[jj,1,:],P1[jj,2,:],U[:,jj])
        for ii in range(len(x)):
            if np.sum(U01[ii,:]) > 1:
                U[ii,:] = dadi.Integration.tridiag.tridiag(P2[ii,0,:],P2[ii,1,:],P2[ii,2,:],U[ii,:])
    else:
        for ii in range(len(x)):
            if np.sum(U01[ii,:]) > 1:
                U[ii,:] = dadi.Integration.tridiag.tridiag(P2[ii,0,:],P2[ii,1,:],P2[ii,2,:],U[ii,:])
        for jj in range(len(x)):
            if np.sum(U01[:,jj]) > 1:
                U[:,jj] = dadi.Integration.tridiag.tridiag(P1[jj,0,:],P1[jj,1,:],P1[jj,2,:],U[:,jj])
    return U

def advance_cov(U,C,x,dx):
    """
    Explicit integration of the covariance term, using scipy's sparse matrix for C
    U - density function
    C - transition matrix
    """
    U = ( C * U.reshape(len(x)**2)).reshape(len(x),len(x))
    return U

def advance1D(u,P):
    """
    tridiag breakdown for integration along the diagonal boundary
    u - density along that diagonal
    P - transition matrix
    """
    a = np.concatenate((np.array([0]),np.diag(P,-1)))
    b = np.diag(P)
    c = np.concatenate((np.diag(P,1),np.array([0])))
    u = dadi.Integration.tridiag.tridiag(a,b,c,u)
    return u

def advance_line(x,phi,P):
    """
    Integrate along the diagonal boundary. Density gets fixed along the boundary, and then diffuses along that boundary until being fixed in one of the two corners
    P - one dimensional transition matrix for the diagonal boundary
    """
    u = np.diag(np.fliplr(phi))
    u = advance1D(u,P)
    for ii in range(len(x)):
        phi[ii,len(x) - ii - 1] = u[ii]
    return phi

### sampling methods

sample_cache = {}
def sample(phi, ns, x):
    """
    Obtain the expected sample frequency spectrum from the density function
    """
    dx = grid_dx(x)
    DXX = grid_dx_2d(x,dx)

    # Assume ns is typically a list of length 1, if not, make it into one.
    try:
        ns = tuple(ns)
    except TypeError:
        ns = (ns,)

    # We cache calculations of several big matrices that will be re-used 
    # within and between integrations.
    key = (ns, tuple(x))
    if key not in sample_cache:
        this_cache = {}
        for ii in range(1,ns[0]-1):
            # Create our cache
            this_cache[ii] = (1-x[:,nuax]-x[nuax,:])**ii
            # Somewhat ugly hack to use negative values to store second array
            # to cache.
            this_cache[-ii] = x[nuax,:]**ii
        sample_cache[key] = this_cache
    else:
        this_cache = sample_cache[key]

    #dx = grid_dx(x)
    F = np.zeros((ns[0]+1,ns[0]+1))
    prod_phi = DXX*phi
    for ii in range(len(F)):
        prod_x = prod_phi * x[:,nuax]**ii
        for jj in range(len(F)):
            if ii+jj < ns[0] and ii != 0 and jj != 0:
                #F[ii,jj] = math.factorial(ns)/(math.factorial(ii)*math.factorial(jj)*math.factorial(ns-ii-jj)) * int2(x, dx, phi*x[:,nuax]**ii*x[nuax,:]**jj*(1-x[:,nuax]-x[nuax,:])**(ns-ii-jj) )
                F[ii,jj] = trinomial(ns[0],ii,jj) * np.sum(prod_x * this_cache[-jj] * this_cache[ns[0]-ii-jj])
    F = TriSpectrum(F)
    F.folded_major = False
    F.folded_ancestral = False
    F.extrap_x = x[1]
    return F

def trinomial(ns,ii,jj):
    """
    Return ns!/(ii! * jj! * (ns-ii-jj)!) for large values
    """
    return np.exp(math.lgamma(ns+1) - math.lgamma(ii+1) - math.lgamma(jj+1) - math.lgamma(ns-ii-jj+1))


### various methods for spectrum manipulation or other methods needed for data fitting

def misidentification(F, p):
    """
    Given folded spectrum, and probability p that one of the derived alleles is the actual ancestral allele
    Then refold to return folded spectrum
    """
    F = TriF(np.zeros((len(F),len(F))))
    for ii in range(len(F))[1:-1]:
        for jj in range(len(F))[1:ii+1]:
            if ii+jj < len(F):
                F[ii,jj] += (1 - p) * F[ii,jj]
                F[len(F)-1-ii-jj,jj] += p/2. * F[ii,jj]
                F[ii,len(F)-1-ii-jj] += p/2. * F[ii,jj]
    
    return F.fold()
    
def fold(spectrum):
    """
    Note: this is now handled in the TriSpectrum class
    Given a frequency spectrum over the full domain, fold into a spectrum with major and minor derived alleles
    """
    spectrum = TriSpectrum(spectrum)
    if spectrum.folded_major == True:
        print("error: trying to fold a spectrum that is already folded")
        return spectrum
    else:
        spectrum = (spectrum + np.transpose(spectrum))
        for ii in range(len(spectrum)):
            spectrum[ii,ii] = spectrum[ii,ii]/2
        spectrum.mask[0,:] = True
        spectrum.mask[:,0] = True
        for ii in range(len(spectrum)):
            spectrum.mask[ii,ii+1:] = True
            spectrum.mask[ii,len(spectrum)-1-ii:] = True
        return spectrum

def univariate_lognormal_pdf(x,sigma,mu):
    """
    Can compare to scipy.stats.lognorm.pdf(x,sigma,0,np.exp(mu))
    """
    return 1./(x*sigma*np.sqrt(2*math.pi)) * np.exp ( -(np.log(x) - mu)**2 / (2*sigma**2) )

def bivariate_lognormal_pdf(xx, params):
    """
    mu_i = mu_yi and sigma_i = sigma_yi are the associated means and variances of the bivariate normal distr which gets exponentiated
    We assume for our application that mu1=mu2 and sigma1=sigma2, though this isn't necessary for the general bivariate lognormal distribution
    """
    mu1, mu2, sigma1, sigma2, rho = params
    norm = 1./(2 * np.pi * (sigma1*sigma2) * np.sqrt(1-rho**2) * np.outer(xx,xx) )
    q = 1/(1-rho**2) * ( ((np.log(xx[nuax,:])-mu1)/sigma1)**2 - 2*rho*((np.log(xx[nuax,:])-mu1)/sigma1)*((np.log(xx[:,nuax])-mu2)/sigma2) + ((np.log(xx[:,nuax])-mu2)/sigma2)**2 )
    prob = norm * np.exp( -q/2. )
    
    return prob

def ms_demo_params_to_dadi(nu_ms,tau_ms):
    """
    convert from ms parameters (which use current pop size) for nu and tau to dadi parameters (which use ancestral pop size)
    """
    return 1./nu_ms,2*tau_ms/nu_ms

def optimal_sfs_scaling(model,data):
    data = numerics.fold(data)
    model = numerics.fold(data)
    model, data = Numerics.intersect_masks(model, data)
    return data.sum()/model.sum()

#def tri_spectrum(array):
#    """
#    takes array and outputs correctly masked triallele spectrum
#    """
#    F = dadi.Spectrum(array)
#    F.mask[:,0] = True
#    F.mask[0,:] = True
#    for ii in range(len(F))[1:]:
#        F.mask[ii,len(F)-ii-1:] = True
#    return F
    
def fold_ancestral(F):
    """
    Note: this is now handled by the TriSpectrum class
    Don't know ancestral state, so track minor frequencies
    Store spectrum of two minor allele frequencies
    """
    F_new = 0*F
    F_new = TriSpectrum(F_new)
    ns = len(F)-1
    for ii in range(ns):
        for jj in range(ns):
            kk = ns-ii-jj
            if F.mask[ii,jj] == True:
                continue
            elif ii <= kk and jj <= kk:
                if ii >= jj:
                    F_new[ii,jj] += F[ii,jj]
                else:
                    F_new[jj,ii] += F[ii,jj]
            elif ii > kk and jj <= kk:
                F_new[kk,jj] += F[ii,jj]
            elif ii <= kk and jj > kk:
                F_new[kk,ii] += F[ii,jj]
            else: # ii > kk and jj > kk
                if ii >= jj:
                    F_new[jj,kk] += F[ii,jj]
                else:
                    F_new[ii,kk] += F[ii,jj]
    # mask if not a valid entry for ancestrally folded spectrum
    for ii in range(ns):
        for jj in range(ns):
            kk = ns-ii-jj
            if not (kk>=ii>=jj):
                F_new.mask[ii,jj] = True
    return F_new

def ln_binomial(n,k):
    return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)

projection_cache = {}
def cached_projection(proj_to,proj_from,hits):
    """
    Coefficients for projection from a larger size to smaller
    proj_to: Number of samples to project down to
    proj_from: Number of samples to project from
    hits: Number of derived alleles projecting from - tuple of (n1,n3)
    """
    key = (proj_to, proj_from, hits)
    try:
        return projection_cache[key]
    except KeyError:
        pass
    
    X1, X2 = hits
    X3 = proj_from - X1 - X2
    proj_weights = np.zeros((proj_to+1,proj_to+1))
    for ii in range(X1+1):
        for jj in range(X2+1):
            kk = proj_to - ii - jj
            if kk > X3 or kk <0:
                continue
            f = ln_binomial(X1,ii) + ln_binomial(X2,jj) + ln_binomial(X3,kk) - ln_binomial(proj_from,proj_to)
            proj_weights[ii,jj] = np.exp(f)
    
    projection_cache[key] = proj_weights
    return proj_weights
    
def project(F_from, proj_to):
    proj_from = len(F_from)-1
    if proj_to == proj_from:
        return F_from
    elif proj_to > proj_from:
        print('sorry, but projection must be to smaller size!')
        return F_from
    else:
        F_proj = np.zeros((proj_to+1,proj_to+1))
        for X1 in range(proj_from):
            for X2 in range(proj_from):
                if F_from.mask[X1,X2] == False:
                    hits = (X1,X2)
                    proj_weights = cached_projection(proj_to,proj_from,hits)
                    F_proj += proj_weights * F_from[X1,X2]
        
        return TriSpectrum(F_proj).fold()


# Try importing cythonized versions of several slow methods. These imports should overwrite the Python code defined above.
try:
    from transition1 import transition1
    from transition2 import transition2
    from transition12 import transition12
    from transition1D import transition1D
except ImportError:
    pass
