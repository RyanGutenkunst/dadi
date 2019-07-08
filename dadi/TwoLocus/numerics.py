"""
methods for numerics of two locus diffusion - grids, transition matrices, etc
"""
import numpy as np
import dadi
from numpy import newaxis as nuax
from scipy.sparse import lil_matrix,identity
import math

from dadi.TwoLocus.TLSpectrum_mod import TLSpectrum

tol = 1e-12

def LD_per_bin(ns):
    """
    LD statistics per bin for a TLSpectrum.

    ns: Number of samples

    Returns (D, r2). Both are TLSpectrum objects in which each entry is the
    value of D or r^2 for that combination of haplotypes.
    """
    temp = np.arange(ns+1, dtype=float)/ns
    # Fancy array arithmetic, to avoid explicity for loops.
    pAB = temp[:,nuax,nuax]
    pAb = temp[nuax,:,nuax]
    paB = temp[nuax,nuax,:]
    pA = pAB + pAb
    pB = pAB + paB
    D = TLSpectrum(pAB - pA*pB)
    r2 = TLSpectrum(D**2/(pA*(1-pA)*pB*(1-pB)))

    return D,r2

def grid(numpts):
    """
    Default grid is uniform, so that grid points lie directly on the boundaries
    Here, delta_x = 1./numpts
    numpts - number of grid points (less one) 
    """
    return np.linspace(0,1,numpts+1)

def grid_dx(x):
    """
    Given 1D grid x, this returns the 1D grid spacing for x
    """
    return (np.concatenate((np.diff(x),np.array([0]))) + np.concatenate((np.array([0]),np.diff(x))))/2

def domain(x):
    """
    Array of same dimension as density function discretization
    Has a 1 if grid point lies in domain, 0 if outside domain
    x - 1D grid
    """
    tol = 1e-12
    U01 = np.ones((len(x),len(x),len(x)))
    XX = x[:,nuax,nuax] + x[nuax,:,nuax] + x[nuax,nuax,:]
    U01[np.where(XX > 1+tol)] = 0
    return U01

def grid_dx3(x,dx):
    """
    For integrating over the 3D domain (see int3)
    x - 1D grid
    dx - 1D grid spacing
    """
    DX = dx[:,nuax,nuax]*dx[nuax,:,nuax]*dx[nuax,nuax,:]
    for ii in range(len(x)):
        for jj in range(len(x)):
            DX[ii,jj,len(x)-ii-jj-1] *= 1./2
    return DX

def int3(DX,U):
    """
    Numerically integrate the density function phi
    phi - density function
    DX - 3D grid spacing for integration weights
    """
    return np.sum(DX*U)

"""
Injection of new mutations along axes
"""

def injectA(x,dx,dt,yB,phi,thetaA):
    """
    Injection new derived mutations A onto background of B/b
    x - 1D grid
    dx - 1D grid spacing
    dt - integration time step
    yA - biallelic frequency spectrum integrated by dadi
    phi - density function
    thetaA - scaled mutation rate
    """
    phi[0,1,1:-1] += dt/dx[1] / x[1]**2 * yB[1:-1] * (1-x[1:-1]) * thetaA/2.
    phi[1,0,1:-1] += dt/dx[1] / x[1]**2 * yB[1:-1] * x[1:-1] * thetaA/2.
    phi[1,0,0] += dt/dx[1] * 2 / x[1]**2 * yB[1] * x[1] * thetaA/2. 
    return phi

def injectB(x,dx,dt,yA,phi,thetaB):
    """
    Injection new derived mutations B onto background of A/a
    x - 1D grid
    dx - 1D grid spacing
    dt - integration time step
    yB - biallelic frequency spectrum integrated by dadi
    phi - density function
    thetaB - scaled mutation rate
    """
    phi[0,1:-1,1] += dt/dx[1] / x[1]**2 * yA[1:-1] * (1-x[1:-1]) * thetaB/2.
    phi[1,1:-1,0] += dt/dx[1] / x[1]**2 * yA[1:-1] * x[1:-1] * thetaB/2.
    phi[1,0,0] += dt/dx[1] * 2 / x[1]**2 * yA[1] * x[1] * thetaB/2. 
    return phi

"""
Transition matrices for forward integration in bulk of domain.
"""

def transition1(x,dx,U01,gammaA,gammaB,rho,nu,hA=.5,hB=.5):
    """
    Transition matrix for ADI method along first axis
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    P = np.zeros((len(x),len(x),3,len(x)))
    for jj in range(len(x)):
        for kk in range(len(x)):
            if len(np.where(U01[:,jj,kk] == 1)[0]) <= 2:
                continue
            A = np.zeros((len(x),len(x)))
            # drift
            if (jj > 0 or kk > 0) and x[jj] + x[kk] < 1 - tol:
                V = x*(1-x)/nu
                for ii in np.where(U01[:,jj,kk] == 1)[0][:-1]:
                    if ii == 0:
                        A[ii,ii] =  - 1/(2*dx[ii]) * ( -V[ii]/(x[ii+1]-x[ii]) )
                        A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) )
                    elif ii == np.where(U01[:,jj,kk] == 1)[0][:-1][-1]:
                        A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) )
                        A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) )
                    else:
                        A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) )
                        A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) -V[ii]/(x[ii+1]-x[ii]) )
                        A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) )
                
                x2 = x[jj]
                x3 = x[kk]
                M = 2*gammaA * x*(1-x-x2)*(hA+(x+x2)*(1-2*hA)) + 2*gammaB * x*(1-x-x3)*(hB+(x+x3)*(1-2*hB)) - rho/2. * (x*(1-x-x2-x3) - x2*x3)
                for ii in np.where(U01[:,jj,kk] == 1)[0][:-1]:
                    if ii == 0:
                        A[ii,ii] += 1/dx[ii] * ( M[ii] ) / 2
                        A[ii,ii+1] += 1/dx[ii] * ( M[ii+1] ) / 2
                    elif ii == np.where(U01[:,jj,kk] == 1)[0][:-1][-1]:
                        A[ii,ii-1] += 1/dx[ii] * ( - M[ii-1] ) / 2
                        A[ii,ii] += 1/dx[ii] * ( - M[ii] ) / 2
                    else:
                        A[ii,ii-1] += 1/dx[ii] * ( - M[ii-1] ) / 2
                        A[ii,ii] += 0 #1/dx[ii] * ( M[ii] - M[ii-1] ) / 2
                        A[ii,ii+1] += 1/dx[ii] * ( M[ii+1] ) / 2


            if jj == 0 and kk == 0:
                V = x*(1-x)/nu
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
                
                x2 = x[jj]
                x3 = x[kk]
                M = 2*gammaA * x*(1-x-x2)*(hA+(x+x2)*(1-2*hA)) + 2*gammaB * x*(1-x-x3)*(hB+(x+x3)*(1-2*hB)) - rho/2. * (x*(1-x-x2-x3) - x2*x3)
                for ii in range(len(x)):
                    if ii == 0:
                        A[ii,ii] += 1/dx[ii] * ( M[ii] ) / 2
                        A[ii,ii+1] += 1/dx[ii] * ( M[ii+1] ) / 2
                    elif ii == len(x)-1:
                        A[ii,ii-1] += 1/dx[ii]*2 * ( - M[ii-1] ) / 2
                        A[ii,ii] += 1/dx[ii]*2 * ( - M[ii] ) / 2
                    else:
                        A[ii,ii-1] += 1/dx[ii] * ( - M[ii-1] ) / 2
                        A[ii,ii] += 0 #1/dx[ii] * ( M[ii] - M[ii-1] ) / 2
                        A[ii,ii+1] += 1/dx[ii] * ( M[ii+1] ) / 2

            
            P[jj,kk,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
            P[jj,kk,1,:] = np.diagonal(A)
            P[jj,kk,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    return P

def transition2(x,dx,U01,gammaA,gammaB,rho,nu,hA=.5,hB=.5):
    """
    Transition matrix for ADI method along second axis
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    P = np.zeros((len(x),len(x),3,len(x)))
    for ii in range(len(x)):
        for kk in range(len(x)):
            if len(np.where(U01[ii,:,kk] == 1)[0]) <= 2:
                continue
            A = np.zeros((len(x),len(x)))
            # drift
            if (ii > 0 or kk > 0) and x[ii] + x[kk] < 1 - tol:
                V = x*(1-x)/nu
                for jj in np.where(U01[ii,:,kk] == 1)[0][:-1]:
                    if jj == 0:
                        A[jj,jj] =  - 1/(2*dx[jj]) * ( -V[jj]/(x[jj+1]-x[jj]) )
                        A[jj,jj+1] = - 1/(2*dx[jj]) * ( V[jj+1]/(x[jj+1]-x[jj]) )
                    elif jj == np.where(U01[ii,:,kk] == 1)[0][:-1][-1]:
                        A[jj,jj-1] = - 1/(2*dx[jj]) * ( V[jj-1]/(x[jj]-x[jj-1]) )
                        A[jj,jj] = - 1/(2*dx[jj]) * ( -V[jj]/(x[jj]-x[jj-1]) )
                    else:
                        A[jj,jj-1] = - 1/(2*dx[jj]) * ( V[jj-1]/(x[jj]-x[jj-1]) )
                        A[jj,jj] = - 1/(2*dx[jj]) * ( -V[jj]/(x[jj]-x[jj-1]) -V[jj]/(x[jj+1]-x[jj]) )
                        A[jj,jj+1] = - 1/(2*dx[jj]) * ( V[jj+1]/(x[jj+1]-x[jj]) )
                
                x1 = x[ii]
                x3 = x[kk]
                M = 2*gammaA * x*(1-x1-x)*(hA+(x1+x)*(1-2*hA)) - 2*gammaB * x*(x1+x3)*(hB+(x1+x3)*(1-2*hB)) + rho/2. * (x1*(1-x1-x-x3) - x*x3)
                for jj in np.where(U01[ii,:,kk] == 1)[0][:-1]:
                    if jj == 0:
                        A[jj,jj] += 1/dx[jj] * ( M[jj] ) / 2
                        A[jj,jj+1] += 1/dx[jj] * ( M[jj+1] ) / 2
                    elif jj == np.where(U01[ii,:,kk] == 1)[0][:-1][-1]:
                        A[jj,jj-1] += 1/dx[jj] * ( - M[jj-1] ) / 2
                        A[jj,jj] += 1/dx[jj] * ( - M[jj] ) / 2
                    else:
                        A[jj,jj-1] += 1/dx[jj] * ( - M[jj-1] ) / 2
                        A[jj,jj] += 0 #1/dx[jj] * ( M[jj] - M[jj-1] ) / 2
                        A[jj,jj+1] += 1/dx[jj] * ( M[jj+1] ) / 2


            if ii == 0 and kk == 0:
                V = x*(1-x)/nu
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
                
                x1 = x[ii]
                x3 = x[kk]
                M = 2*gammaA * x*(1-x1-x)*(hA+(x1+x)*(1-2*hA)) - 2*gammaB * x*(x1+x3)*(hB+(x1+x3)*(1-2*hB)) + rho/2. * (x1*(1-x1-x-x3) - x*x3)
                for jj in range(len(x)):
                    if jj == 0:
                        A[jj,jj] += 1/dx[jj] * ( M[jj] ) / 2
                        A[jj,jj+1] += 1/dx[jj] * ( M[jj+1] ) / 2
                    elif jj == len(x)-1:
                        A[jj,jj-1] += 1/dx[jj]*2 * ( - M[jj-1] ) / 2
                        A[jj,jj] += 1/dx[jj]*2 * ( - M[jj] ) / 2
                    else:
                        A[jj,jj-1] += 1/dx[jj] * ( - M[jj-1] ) / 2
                        A[jj,jj] += 0 #1/dx[jj] * ( M[jj] - M[jj-1] ) / 2
                        A[jj,jj+1] += 1/dx[jj] * ( M[jj+1] ) / 2

            
            P[ii,kk,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
            P[ii,kk,1,:] = np.diagonal(A)
            P[ii,kk,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    return P


def transition3(x,dx,U01,gammaA,gammaB,rho,nu,hA=.5,hB=.5):
    """
    Transition matrix for ADI method along third axis
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    P = np.zeros((len(x),len(x),3,len(x)))
    for ii in range(len(x)):
        for jj in range(len(x)):
            if len(np.where(U01[ii,jj,:] == 1)[0]) <= 2:
                continue
            A = np.zeros((len(x),len(x)))
            # drift
            if (ii > 0 or jj > 0) and x[ii] + x[jj] < 1 - tol:
                V = x*(1-x)/nu
                for kk in np.where(U01[ii,jj,:] == 1)[0][:-1]:
                    if kk == 0:
                        A[kk,kk] =  - 1/(2*dx[kk]) * ( -V[kk]/(x[kk+1]-x[kk]) )
                        A[kk,kk+1] = - 1/(2*dx[kk]) * ( V[kk+1]/(x[kk+1]-x[kk]) )
                    elif kk == np.where(U01[ii,jj,:] == 1)[0][:-1][-1]:
                        A[kk,kk-1] = - 1/(2*dx[kk]) * ( V[kk-1]/(x[kk]-x[kk-1]) )
                        A[kk,kk] = - 1/(2*dx[kk]) * ( -V[kk]/(x[kk]-x[kk-1]) )
                    else:
                        A[kk,kk-1] = - 1/(2*dx[kk]) * ( V[kk-1]/(x[kk]-x[kk-1]) )
                        A[kk,kk] = - 1/(2*dx[kk]) * ( -V[kk]/(x[kk]-x[kk-1]) -V[kk]/(x[kk+1]-x[kk]) )
                        A[kk,kk+1] = - 1/(2*dx[kk]) * ( V[kk+1]/(x[kk+1]-x[kk]) )
                
                x1 = x[ii]
                x2 = x[jj]
                M = - 2*gammaA * x*(x1+x2)*(hA+(x1+x2)*(1-2*hA)) + 2*gammaB * x*(1-x1-x)*(hB+(x1+x)*(1-2*hB)) + rho/2. * (x1*(1-x1-x2-x) - x2*x)
                for kk in np.where(U01[ii,jj,:] == 1)[0][:-1]:
                    if kk == 0:
                        A[kk,kk] += 1/dx[kk] * ( M[kk] ) / 2
                        A[kk,kk+1] += 1/dx[kk] * ( M[kk+1] ) / 2
                    elif kk == np.where(U01[ii,jj,:] == 1)[0][:-1][-1]:
                        A[kk,kk-1] += 1/dx[kk] * ( - M[kk-1] ) / 2
                        A[kk,kk] += 1/dx[kk] * ( - M[kk] ) / 2
                    else:
                        A[kk,kk-1] += 1/dx[kk] * ( - M[kk-1] ) / 2
                        A[kk,kk] += 0 #1/dx[kk] * ( M[kk] - M[kk-1] ) / 2
                        A[kk,kk+1] += 1/dx[kk] * ( M[kk+1] ) / 2


            if ii == 0 and jj == 0:
                V = x*(1-x)/nu
                for kk in range(len(x)):
                    if kk == 0:
                        A[kk,kk] =  - 1/(2*dx[kk]) * ( -V[kk]/(x[kk+1]-x[kk]) )
                        A[kk,kk+1] = - 1/(2*dx[kk]) * ( V[kk+1]/(x[kk+1]-x[kk]) )
                    elif kk == len(x)-1:
                        A[kk,kk-1] = - 1/(2*dx[kk])*2 * ( V[kk-1]/(x[kk]-x[kk-1]) )
                        A[kk,kk] = - 1/(2*dx[kk])*2 * ( -V[kk]/(x[kk]-x[kk-1]) )
                    else:
                        A[kk,kk-1] = - 1/(2*dx[kk]) * ( V[kk-1]/(x[kk]-x[kk-1]) )
                        A[kk,kk] = - 1/(2*dx[kk]) * ( -V[kk]/(x[kk]-x[kk-1]) -V[kk]/(x[kk+1]-x[kk]) )
                        A[kk,kk+1] = - 1/(2*dx[kk]) * ( V[kk+1]/(x[kk+1]-x[kk]) )
                
                x1 = x[ii]
                x2 = x[jj]
                M = - 2*gammaA * x*(x1+x2)*(hA+(x1+x2)*(1-2*hA)) + 2*gammaB * x*(1-x1-x)*(hB+(x1+x)*(1-2*hB)) + rho/2. * (x1*(1-x1-x2-x) - x2*x)
                for kk in range(len(x)):
                    if kk == 0:
                        A[kk,kk] += 1/dx[kk] * ( M[kk] ) / 2
                        A[kk,kk+1] += 1/dx[kk] * ( M[kk+1] ) / 2
                    elif kk == len(x)-1:
                        A[kk,kk-1] += 1/dx[kk]*2 * ( - M[kk-1] ) / 2
                        A[kk,kk] += 1/dx[kk]*2 * ( - M[kk] ) / 2
                    else:
                        A[kk,kk-1] += 1/dx[kk] * ( - M[kk-1] ) / 2
                        A[kk,kk] += 0 #1/dx[kk] * ( M[kk] - M[kk-1] ) / 2
                        A[kk,kk+1] += 1/dx[kk] * ( M[kk+1] ) / 2

            
            P[ii,jj,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
            P[ii,jj,1,:] = np.diagonal(A)
            P[ii,jj,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    return P

def transition12(x, dx, U01):
    """
    Covariance discretization for slice along axes 1 and 2
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    P = []*len(x)
    for kk in range(len(x)):
        C = lil_matrix((len(x)**2,len(x)**2))
        if kk < len(x) - 4:
            for ii in range(len(x)-1)[:-1]:
                for jj in range(len(x)-1)[:-1]:
                    if U01[ii+2,jj+2,kk] == 1 or U01[ii+1,jj+2,kk] == 1 or U01[ii+2,jj+1,kk] == 1:
                        if ii+1 < len(x)-kk and jj+1 < len(x)-kk and U01[ii+1,jj+1,kk] == 1:
                            C[ii*len(x)+jj,(ii+1)*len(x)+(jj+1)] += 1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii+1]*x[jj+1])
                            
                        if ii+1 < len(x)-kk and jj-1 >= 0 and U01[ii+1,jj-1,kk] == 1:
                            C[ii*len(x)+jj,(ii+1)*len(x)+(jj-1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii+1]*x[jj-1])
                        
                        if ii-1 >= 0 and jj+1 < len(x)-kk and U01[ii-1,jj+1,kk] == 1:
                            C[ii*len(x)+jj,(ii-1)*len(x)+(jj+1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj+1])
                            
                        if ii-1 >= 0 and jj-1 >= 0 and U01[ii-1,jj-1,kk] == 1:
                            C[ii*len(x)+jj,(ii-1)*len(x)+(jj-1)] += 1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj-1])
                    
                    elif U01[ii+1,jj+1,kk] == 1 or U01[ii+1,jj,kk] == 1 or U01[ii,jj+1,kk] == 1:
                        if ii+1 < len(x)-kk and jj-1 >= 0 and U01[ii+1,jj-1,kk] == 1:
                            C[ii*len(x)+jj,(ii+1)*len(x)+(jj-1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii+1]*x[jj-1]) / 2
                        
                        if ii-1 >= 0 and jj+1 < len(x)-kk and U01[ii-1,jj+1,kk] == 1:
                            C[ii*len(x)+jj,(ii-1)*len(x)+(jj+1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj+1]) / 2
                            
                        if ii-1 >= 0 and jj-1 >= 0 and U01[ii-1,jj-1,kk] == 1:
                            C[ii*len(x)+jj,(ii-1)*len(x)+(jj-1)] += 1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj-1])
        
        if kk == 0:
            ii = 0
            jj = len(x)-2-kk
            C[ii*len(x)+jj,(ii+1)*len(x)+(jj-1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii+1]*x[jj-1]) / 2
            
            ii = len(x)-2-kk
            jj = 0
            C[ii*len(x)+jj,(ii-1)*len(x)+(jj+1)] += -1./4 * 1./(dx[ii]*dx[jj]) * (-x[ii-1]*x[jj+1]) / 2
        
        P.append(C)
    return P

def transition13(x, dx, U01):
    """
    Covariance discretization for slice along axes 1 and 3
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    P = []*len(x)
    for jj in range(len(x)):
        C = lil_matrix((len(x)**2,len(x)**2))
        if jj < len(x) - 4:
            for ii in range(len(x)-1)[:-1]:
                for kk in range(len(x)-1)[:-1]:
                    if U01[ii+2,jj,kk+2] == 1 or U01[ii+1,jj,kk+2] == 1 or U01[ii+2,jj,kk+1] == 1:
                        if ii+1 < len(x)-jj and kk+1 < len(x)-jj and U01[ii+1,jj,kk+1] == 1:
                            C[ii*len(x)+kk,(ii+1)*len(x)+(kk+1)] += 1./4 * 1./(dx[ii]*dx[kk]) * (-x[ii+1]*x[kk+1])
                            
                        if ii+1 < len(x)-jj and kk-1 >= 0 and U01[ii+1,jj,kk-1] == 1:
                            C[ii*len(x)+kk,(ii+1)*len(x)+(kk-1)] += -1./4 * 1./(dx[ii]*dx[kk]) * (-x[ii+1]*x[kk-1])
                        
                        if ii-1 >= 0 and kk+1 < len(x)-jj and U01[ii-1,jj,kk+1] == 1:
                            C[ii*len(x)+kk,(ii-1)*len(x)+(kk+1)] += -1./4 * 1./(dx[ii]*dx[kk]) * (-x[ii-1]*x[kk+1])
                            
                        if ii-1 >= 0 and kk-1 >= 0 and U01[ii-1,jj,kk-1] == 1:
                            C[ii*len(x)+kk,(ii-1)*len(x)+(kk-1)] += 1./4 * 1./(dx[ii]*dx[kk]) * (-x[ii-1]*x[kk-1])
                    
                    elif U01[ii+1,jj,kk+1] == 1 or U01[ii+1,jj,kk] == 1 or U01[ii,jj,kk+1] == 1:
                        if ii+1 < len(x)-jj and kk-1 >= 0 and U01[ii+1,jj,kk-1] == 1:
                            C[ii*len(x)+kk,(ii+1)*len(x)+(kk-1)] += -1./4 * 1./(dx[ii]*dx[kk]) * (-x[ii+1]*x[kk-1]) / 2
                        
                        if ii-1 >= 0 and kk+1 < len(x)-jj and U01[ii-1,jj,kk+1] == 1:
                            C[ii*len(x)+kk,(ii-1)*len(x)+(kk+1)] += -1./4 * 1./(dx[ii]*dx[kk]) * (-x[ii-1]*x[kk+1]) / 2
                            
                        if ii-1 >= 0 and kk-1 >= 0 and U01[ii-1,jj,kk-1] == 1:
                            C[ii*len(x)+kk,(ii-1)*len(x)+(kk-1)] += 1./4 * 1./(dx[ii]*dx[kk]) * (-x[ii-1]*x[kk-1])
        
        if jj == 0:
            ii = 0
            kk = len(x)-2-jj
            C[ii*len(x)+kk,(ii+1)*len(x)+(kk-1)] += -1./4 * 1./(dx[ii]*dx[kk]) * (-x[ii+1]*x[kk-1]) / 2
            
            ii = len(x)-2-jj
            kk = 0
            C[ii*len(x)+kk,(ii-1)*len(x)+(kk+1)] += -1./4 * 1./(dx[ii]*dx[kk]) * (-x[ii-1]*x[kk+1]) / 2
        
        P.append(C)
    return P

def transition23(x, dx, U01):
    """
    Covariance discretization for slice along axes 2 and 3
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    P = []*len(x)
    for ii in range(len(x)):
        C = lil_matrix((len(x)**2,len(x)**2))
        if ii < len(x) - 4:
            for jj in range(len(x)-1)[:-1]:
                for kk in range(len(x)-1)[:-1]:
                    if U01[ii,jj+2,kk+2] == 1 or U01[ii,jj+1,kk+2] == 1 or U01[ii,jj+2,kk+1] == 1:
                        if jj+1 < len(x)-ii and kk+1 < len(x)-ii and U01[ii,jj+1,kk+1] == 1:
                            C[jj*len(x)+kk,(jj+1)*len(x)+(kk+1)] += 1./4 * 1./(dx[jj]*dx[kk]) * (-x[jj+1]*x[kk+1])
                            
                        if jj+1 < len(x)-ii and kk-1 >= 0 and U01[ii,jj+1,kk-1] == 1:
                            C[jj*len(x)+kk,(jj+1)*len(x)+(kk-1)] += -1./4 * 1./(dx[jj]*dx[kk]) * (-x[jj+1]*x[kk-1])
                        
                        if jj-1 >= 0 and kk+1 < len(x)-ii and U01[ii,jj-1,kk+1] == 1:
                            C[jj*len(x)+kk,(jj-1)*len(x)+(kk+1)] += -1./4 * 1./(dx[jj]*dx[kk]) * (-x[jj-1]*x[kk+1])
                            
                        if jj-1 >= 0 and kk-1 >= 0 and U01[ii,jj-1,kk-1] == 1:
                            C[jj*len(x)+kk,(jj-1)*len(x)+(kk-1)] += 1./4 * 1./(dx[jj]*dx[kk]) * (-x[jj-1]*x[kk-1])
                    
                    elif U01[ii,jj+1,kk+1] == 1 or U01[ii,jj+1,kk] == 1 or U01[ii,jj,kk+1] == 1:
                        if jj+1 < len(x)-ii and kk-1 >= 0 and U01[ii,jj+1,kk-1] == 1:
                            C[jj*len(x)+kk,(jj+1)*len(x)+(kk-1)] += -1./4 * 1./(dx[jj]*dx[kk]) * (-x[jj+1]*x[kk-1]) / 2
                        
                        if jj-1 >= 0 and kk+1 < len(x)-ii and U01[ii,jj-1,kk+1] == 1:
                            C[jj*len(x)+kk,(jj-1)*len(x)+(kk+1)] += -1./4 * 1./(dx[jj]*dx[kk]) * (-x[jj-1]*x[kk+1]) / 2
                            
                        if jj-1 >= 0 and kk-1 >= 0 and U01[ii,jj-1,kk-1] == 1:
                            C[jj*len(x)+kk,(jj-1)*len(x)+(kk-1)] += 1./4 * 1./(dx[jj]*dx[kk]) * (-x[jj-1]*x[kk-1])
        
        if ii == 0:
            jj = 0
            kk = len(x)-2-ii
            C[jj*len(x)+kk,(jj+1)*len(x)+(kk-1)] += -1./4 * 1./(dx[jj]*dx[kk]) * (-x[jj+1]*x[kk-1]) / 2
            
            jj = len(x)-2-ii
            kk = 0
            C[jj*len(x)+kk,(jj-1)*len(x)+(kk+1)] += -1./4 * 1./(dx[jj]*dx[kk]) * (-x[jj-1]*x[kk+1]) / 2
        
        P.append(C)
    return P


"""
Transition matrices and methods for interaction with non-square face.
Note that this is more complicated than the triallele methods, 
as density can re-enter the domain through recombination; that is, 
the domain boundaries are not necessarily absorbing!
"""

def phi_to_surf(phi,x):
    """
    non-square  mapped to triallelic domain
    note that many of the surface interaction methods could be cythonized for increase in speed
    if x2 or x3 is lost, then only types AB and Ab or AB and aB are left, so either A or B have fixed, and that state is absorbing (no recombination can send you back to the interior
    thus, make sure that x2 lost or x3 lost is the diagonal boundary of the surface domain, so we don't have to worry about that density anymore
    here, we have x3 lost
    """
    surf = np.zeros((len(x),len(x)))
    for ii in range(len(x)): # loop through x1 indices
        for jj in range(len(x)): # loop through x2 indices
            kk = len(x)-1 - ii - jj
            surf[ii,jj] = phi[ii,jj,kk]
    return surf

def surf_to_phi(surf,phi,x):
    """
    move the surf density back to the full phi density function
    """
    for ii in range(len(x)): # loop through x1 indices
        for jj in range(len(x)): # loop through x2 indices
            kk = len(x)-1 - ii - jj
            phi[ii,jj,kk] = surf[ii,jj]
    return phi

def move_density_to_surface(x,dx,dt,gammaA,gammaB,nu,hA=1./2,hB=1./2):
    """
    for each point in the full domain, how much should be lost to the nonsquare surface boundary due to diffusion/selection
    x - grid
    dx - grid spacing
    dt - timestep of integration
    nu - relative population size
    """
    Psurf = np.zeros((len(x),len(x),len(x)))
    if gammaA == 0 and gammaB == 0:
        P1D = transition1D(x,dx,dt,0,nu)
        weights = {}
        for ii in range(len(x))[1:-1]:
            y = np.zeros(len(x))
            y[ii] = 1./dx[ii]
            y = advance1D(y,P1D)
            weights[ii] = y[-1] * dx[-1]
        for ii in range(len(x))[:len(x)-1]:
            for jj in range(len(x))[:len(x)-1-ii]:
                for kk in range(len(x))[:len(x)-1-ii-jj]:
                    if ii+jj+kk == 0:
                        continue
                    Psurf[ii,jj,kk] = weights[ii+jj+kk]
    else:
        for ii in range(len(x))[:len(x)-1]:
            for jj in range(len(x))[:len(x)-1-ii]:
                for kk in range(len(x))[:len(x)-1-ii-jj]:
                    if ii+jj+kk == 0:
                        continue
                    p = x[ii]+x[jj]
                    q = x[ii]+x[kk]
                    #gamma = ((gammaA+gammaB)*x[ii] + gammaA*x[jj] + gammaB*x[kk]) / (x[ii] + x[jj] + x[kk])
                    ### XXX: why is this divided by (x[ii]+x[jj]+x[kk])?? because it's an average, sort of
                    gamma = ((gammaA+gammaB)*x[ii]**2 + 2*(gammaA+hB*gammaB)*x[ii]*x[jj] + 2*(hA*gammaA+gammaB)*x[ii]*x[kk] + (gammaA)*x[jj]**2 + 2*(hA*gammaA+hB*gammaB)*x[jj]*x[kk] + (gammaB)*x[kk]*2 + 2*(hA*gammaA+hB*gammaB)*x[ii]*(1-x[ii]-x[jj]-x[kk]) + 2*(hA*gammaA)*x[jj]*(1-x[ii]-x[jj]-x[kk]) + 2*(hB*gammaB)*x[kk]*(1-x[ii]-x[jj]-x[kk])) / (x[ii]+x[jj]+x[kk])
                    P1D = transition1D(x,dx,dt,gamma,nu)
                    y = np.zeros(len(x))
                    y[ii+jj+kk] = 1./dx[ii+jj+kk]
                    y = advance1D(y,P1D)
                    Psurf[ii,jj,kk] = y[-1]*dx[-1]
    return Psurf

def surface_interaction_b(phi,x,Psurf):
    for ii in range(len(x))[:len(x)-1]:
        for jj in range(len(x))[:len(x)-1-ii]:
            for kk in range(len(x))[:len(x)-1-ii-jj]:
                if ii == 0 and jj == 0:
                    continue
                if ii == 0 and kk == 0:
                    continue
                if jj == 0 and kk == 0:
                    continue
                if ii == 0:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (jj+kk))//2
                    rmdr = (len(x)-1 - (jj+kk))%2
                    if rmdr == 0:
                        phi[ii,jj+dist,kk+dist] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii,jj+dist+1,kk+dist] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj+dist,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj,kk] *= (1-amnt)
                elif jj == 0:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (ii+kk))//2
                    rmdr = (len(x)-1 - (ii+kk))%2
                    if rmdr == 0:
                        phi[ii+dist,jj,kk+dist] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii+dist+1,jj,kk+dist] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii+dist,jj,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj,kk] *= (1-amnt)
                elif kk == 0:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (ii+jj))//2
                    rmdr = (len(x)-1 - (ii+jj))%2
                    if rmdr == 0:
                        phi[ii+dist,jj+dist,kk] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii+dist+1,jj+dist,kk] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii+dist,jj+dist+1,kk] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj,kk] *= (1-amnt)
                else:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (ii+jj+kk))//3
                    rmdr = (len(x)-1 - (ii+jj+kk))%3
                    if rmdr == 0: # in line with boundary grid point
                        phi[ii+dist,jj+dist,kk+dist] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii+dist+1,jj+dist,kk+dist] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist,jj+dist+1,kk+dist] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist,jj+dist,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 2:
                        phi[ii+dist+1,jj+dist+1,kk+dist] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist+1,jj+dist,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist,jj+dist+1,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii,jj,kk] *= (1-amnt)

    return phi

def surface_interaction(phi,x,Psurf):
    """
    density that should be moved to surface
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    for ii in range(len(x))[:len(x)-1]:
        for jj in range(len(x))[:len(x)-1-ii]:
            for kk in range(len(x))[:len(x)-1-ii-jj]:
                if ii == 0 and jj == 0:
                    continue
                if ii == 0 and kk == 0:
                    continue
                if jj == 0 and kk == 0:
                    continue
                if ii == 0:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (jj+kk))//2
                    rmdr = (len(x)-1 - (jj+kk))%2
                    if rmdr == 0:
                        phi[ii,jj+dist,kk+dist] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii,jj+dist+1,kk+dist] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj+dist,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj,kk] *= (1-amnt)
                elif jj == 0:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (ii+kk))//2
                    rmdr = (len(x)-1 - (ii+kk))%2
                    if rmdr == 0:
                        phi[ii+dist,jj,kk+dist] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii+dist+1,jj,kk+dist] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii+dist,jj,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj,kk] *= (1-amnt)
                elif kk == 0:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (ii+jj))//2
                    rmdr = (len(x)-1 - (ii+jj))%2
                    if rmdr == 0:
                        phi[ii+dist,jj+dist,kk] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii+dist+1,jj+dist,kk] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii+dist,jj+dist+1,kk] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj,kk] *= (1-amnt)
                else:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (ii+jj+kk))//3
                    rmdr = (len(x)-1 - (ii+jj+kk))%3
                    if rmdr == 0: # in line with boundary grid point
                        phi[ii+dist,jj+dist,kk+dist] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii+dist+1,jj+dist,kk+dist] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist,jj+dist+1,kk+dist] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist,jj+dist,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 2:
                        phi[ii+dist+1,jj+dist+1,kk+dist] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist+1,jj+dist,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist,jj+dist+1,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii,jj,kk] *= (1-amnt)

    return phi


def surface_recombination(phi,x,rho,dt):
    """
    recombination pushes density from surface back into interior of domain
    """
    if rho == 0:
        return phi
    else:
        for ii in range(len(x)):
            for jj in range(len(x))[1:]:
                kk = len(x)-1 - ii - jj
                if kk > 0:
                    D = -x[jj]*x[kk] # -x2*x3, since x4=0
                    if jj-1 > 0:
                        phi[ii,jj-1,kk] += 1./2 * (-D) * dt * phi[ii,jj,kk] * rho/4. / x[1]
                    else:
                        phi[ii,jj-1,kk] += (-D) * dt * phi[ii,jj,kk] * rho/4. / x[1]
                    if kk-1 > 0:
                        phi[ii,jj,kk-1] += 1./2 * (-D) * dt * phi[ii,jj,kk] * rho/4. / x[1]
                    else:
                        phi[ii,jj,kk-1] += (-D) * dt * phi[ii,jj,kk] * rho/4. / x[1]
                    
                    phi[ii,jj,kk] -= 2 * (-D) * dt * phi[ii,jj,kk] * rho/4. / x[1]
    return phi



# triallele methods
# new gammas: \tilde{gamma1} = gammaA(1-x1-x2)/(1-x1) + gammaB(x2)/(1-x1)
# \tilde{gamma2} = (gammaA-gammaB)(1-x1-x2)/(1-x2) - gammaB(x1)(1-x2)
# use the cythonized methods from the triallele code
"""
forward integration
"""

def advance_adi1(phi,U01,P1,x):
    """
    ADI integration along axis 1 of phi
    phi - density function
    U01 - domain markers
    P1 - ADI transition matrices
    x - grid
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    for jj in range(len(x)):
        for kk in range(len(x)):
            if np.sum(U01[:,jj,kk]) > 1:
                phi[:,jj,kk] = dadi.tridiag.tridiag(P1[jj,kk,0,:],P1[jj,kk,1,:],P1[jj,kk,2,:],phi[:,jj,kk])
    return phi

def advance_adi2(phi,U01,P2,x):
    """
    ADI integration along axis 2 of phi
    phi - density function
    U01 - domain markers
    P2 - ADI transition matrices
    x - grid
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    for ii in range(len(x)):
        for kk in range(len(x)):
            if np.sum(U01[ii,:,kk]) > 1:
                phi[ii,:,kk] = dadi.tridiag.tridiag(P2[ii,kk,0,:],P2[ii,kk,1,:],P2[ii,kk,2,:],phi[ii,:,kk])
    return phi

def advance_adi3(phi,U01,P3,x):
    """
    ADI integration along axis 3 of phi
    phi - density function
    U01 - domain markers
    P3 - ADI transition matrices
    x - grid
    """
    ### XXX: We attempt to import cythonized versions of these methods below. Note: changes should be made to 
    #        this version and the cython version together.
    for ii in range(len(x)):
        for jj in range(len(x)):
            if np.sum(U01[ii,jj,:]) > 1:
                phi[ii,jj,:] = dadi.tridiag.tridiag(P3[ii,jj,0,:],P3[ii,jj,1,:],P3[ii,jj,2,:],phi[ii,jj,:])
    return phi

def advance_adi(phi,U01,P1,P2,P3,x,ii):
    """
    Combined ADI integration of phi
    phi - density function
    U01 - domain markers
    P1/P2/P3 - ADI transition matrices
    x - grid
    """
    if np.mod(ii,3) == 0:
        order = [1,2,3]
    elif np.mod(ii,3) == 1:
        order = [2,3,1]
    else:
        order = [3,1,2]
    for ord in order:
        if ord == 1:
            phi = advance_adi1(phi,U01,P1,x)
        elif ord == 2:
            phi = advance_adi2(phi,U01,P2,x)
        elif ord == 3:
            phi = advance_adi3(phi,U01,P3,x)

    return phi

def advance_cov12(phi,C12,x):
    """
    Explicit integration of covariance term in the 1/2 axes planes
    phi - density function
    C12 - transition matrices
    x - grid
    """
    for kk in range(len(x)):
        C = C12[kk]
        phi[:,:,kk] = ( C * phi[:,:,kk].reshape(len(x)**2) ).reshape(len(x),len(x))
    return phi

def advance_cov13(phi,C13,x):
    """
    Explicit integration of covariance term in the 1/3 axes planes
    phi - density function
    C13 - transition matrices
    x - grid
    """
    for jj in range(len(x)):
        C = C13[jj]
        phi[:,jj,:] = ( C * phi[:,jj,:].reshape(len(x)**2) ).reshape(len(x),len(x))
    return phi


def advance_cov23(phi,C23,x):
    """
    Explicit integration of covariance term in the 2/3 axes planes
    phi - density function
    C23 - transition matrices
    x - grid
    """
    for ii in range(len(x)):
        C = C23[ii]
        phi[ii,:,:] = ( C * phi[ii,:,:].reshape(len(x)**2) ).reshape(len(x),len(x))
    return phi

def advance_cov(phi,C12,C13,C23,x,ii):
    """
    Combined integration for the covariance terms
    phi - density function
    C12/C13/C23 - transition matrices
    x - grid
    """
    if np.mod(ii,3) == 0:
        order = [1,2,3]
    elif np.mod(ii,3) == 1:
        order = [2,3,1]
    else:
        order = [3,1,2]
    for ord in order:
        if ord == 1:
            phi = advance_cov12(phi,C12,x)
        elif ord == 2:
            phi = advance_cov13(phi,C13,x)
        elif ord == 3:
            phi = advance_cov23(phi,C23,x)
    return phi

# methods for integration of the surface - see triallele methods for these (cythonized in triallele code for the transition matrices)

def domain_surf(x):
    """
    Constructs a matrix with the same dimension as the density function discretization, a 1 indicates that the corresponding point is inside the triangular domain or on the boundary, while a 0 indicates that point falls outside the domain
    """
    tol = 1e-12
    U01 = np.ones((len(x),len(x)))
    XX = x[:,nuax] + x[nuax,:]
    U01[np.where(XX > 1+tol)] = 0
    return U01

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

def transition1_surf(x,dx,U01,gammaA,gammaB,rho,nu,hA=.5,hB=.5):
    """
    adi transition matrix for x1 on surface
    See cythonized versions of these in the dadi/Triallele code
    x - grid
    dx - grid spacing
    U01 - surface domain marker
    gammaA/B - scaled selection coefficients
    nu - relative population size
    """
    P = np.zeros((len(x),3,len(x)))
    for jj in range(len(x)):
        A = np.zeros((len(x),len(x)))
        if jj > 0:
            V = x*(1-x)/nu
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
            V = x*(1-x)/nu
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
        
        x2 = x[jj]
        M = 2*gammaA * x*(1-x-x2)*(hA+(x+x2)*(1-2*hA)) + 2*gammaB * x*x2*(hB+(1-x2)*(1-2*hB)) - rho/2. * (-x2*(1-x-x2)) / 2.
        #M = gammaA * x * (1-x-x2) + gammaB * x * x2 - rho/2. * (-x2*(1-x-x2))
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

        P[jj,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
        P[jj,1,:] = np.diagonal(A)
        P[jj,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    return P

def transition2_surf(x,dx,U01,gammaA,gammaB,rho,nu,hA=.5,hB=.5):
    """
    adi transition matrix for x2 on surface
    x - grid
    dx - grid spacing
    U01 - surface domain marker
    gammaA/B - scaled selection coefficients
    nu - relative population size
    """
    P = np.zeros((len(x),3,len(x)))
    for ii in range(len(x)):
        A = np.zeros((len(x),len(x)))
        if ii > 0:
            V = x*(1-x)/nu
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
            V = x*(1-x)/nu
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
        
        x1 = x[ii]
        M = 2*gammaA * x*(1-x1-x)*(hA+(x+x1)*(1-2*hA)) - 2*gammaB * x*(1-x) + rho/2. * (-x * (1-x1-x)) / 2.
        #M = gammaA * x * (1-x1-x) - gammaB * x * (1-x) + rho/2. * (-x * (1-x1-x)) / 2. ### note the (/2.) term at the end, since x3 also would decrease by this amount - we separately handle reentry into domain interior
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
        P[ii,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
        P[ii,1,:] = np.diagonal(A)
        P[ii,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    return P

def transition12_surf(x,dx,U01):
    """
    transition matrix for explicit integration of mixed derivative term along surface
    """
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

def transition1D(x, dx, dt, gamma, nu):
    """
    transition matrix for one dimenional density function
    x - grid
    dx - grid spacing
    dt - timestep of integration
    gamma - scaled selection coefficients
    nu - relative size of population
    """
    P = np.zeros((len(x),len(x)))
    for ii in range(len(x)):
        if ii == 0:
            P[ii,ii] = 1 + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            P[ii,ii+1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii]) + gamma * dt / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
        elif ii == len(x) - 1:
            P[ii,ii-1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1]) - gamma * dt / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            P[ii,ii] = 1 + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1])
        else:
            P[ii,ii-1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1]) - gamma * dt / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            P[ii,ii] = 1 + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1]) + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            P[ii,ii+1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii]) + gamma * dt / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
    return P

def advance_surf_adi1(surf,U01surf,P1surf,x):
    """
    Advance the adi method along first axis one of surface domain
    surf - density function along that surface
    U01surf - domain markers along grid
    P1surf - adi transition matrix
    x - uniform grid
    """
    for jj in range(len(x)):
        if np.sum(U01surf[:,jj]) > 1:
            surf[:,jj] = dadi.tridiag.tridiag(P1surf[jj,0,:],P1surf[jj,1,:],P1surf[jj,2,:],surf[:,jj])
    return surf

def advance_surf_adi2(surf,U01surf,P2surf,x):
    """
    Advance the adi method along second axis one of surface domain
    surf - density function along that surface
    U01surf - domain markers along grid
    P2surf - adi transition matrix
    x - uniform grid
    """
    for ii in range(len(x)):
        if np.sum(U01surf[ii,:]) > 1:
            surf[ii,:] = dadi.tridiag.tridiag(P2surf[ii,0,:],P2surf[ii,1,:],P2surf[ii,2,:],surf[ii,:])
    return surf

def advance_surf_cov(surf,Csurf,x):
    """
    Explicit integration of the covariance term, using scipy's sparse matrix for Csurf
    surf - density function along the surface
    Csurf - transition matrix
    x - grid
    """
    surf = ( Csurf * surf.reshape(len(x)**2)).reshape(len(x),len(x))
    return surf

def move_surf_density_to_bdry(x,surf,P):
    """
    This is an absorbing boundary, so we'll just remove that density and not worry about integrating along that diagonal
    """
    for ii in range(len(x))[1:]:
        for jj in range(len(x))[1:]:
            if x[ii] + x[jj] < 1:
                surf[ii,jj] *= (1-P[ii,jj])
    return surf

def advance1D(u,P):
    """
    Given transition matrix P, use dadi's triagonal solver to integrate in 1D
    u - 1D density function
    P - transition matrix
    """
    a = np.concatenate((np.array([0]),np.diag(P,-1)))
    b = np.diag(P)
    c = np.concatenate((np.diag(P,1),np.array([0])))
    u = dadi.tridiag.tridiag(a,b,c,u)
    return u

def move_surf_to_line(x,dx,dt,gammaA,gammaB,nu,hA=1./2,hB=1./2):
    """
    for each point in the full domain, how much should be lost to the nonsquare surface boundary due to diffusion/selection
    x - grid
    dx - grid spacing
    dt - timestep of integration
    gammaA/gammaB - scaled selection coefficients
    nu - relative population size
    """
    #### XXXX: 11/8/16: assuming that gammaB = 0
    Psurf = np.zeros((len(x),len(x)))
    
    for ii in range(len(x))[1:len(x)-1]:
        for jj in range(len(x))[1:len(x)-1-ii]:
            #gamma = ((gammaA+gammaB)*x[ii] + gammaA*x[jj]) / (x[ii] + x[jj]) ### this should also account for dominance
            gamma = (gammaA*x[ii]**2 + 2*(gammaA)*x[ii]*x[jj] + 2*(hA*gammaA)*x[ii]*(1-x[ii]-x[jj]) + gammaA*x[jj]**2 + 2*(hA*gammaA)*x[jj]*(1-x[ii]-x[jj])) / (x[ii]+x[jj])
            P1D = transition1D(x,dx,dt,gamma,nu)
            y = np.zeros(len(x))
            y[ii+jj] = 1./dx[ii+jj]
            y = advance1D(y,P1D)
            Psurf[ii,jj] = y[-1]*dx[-1]
    return Psurf

def advance_surface(phi,x,P1surf,P2surf,Csurf,Pline,P,U01surf):
    """
    phi - full density function
    x - grid
    P1surf,P2surf - ADI transition matrices for triallele density function
    Csurf - covarance transition matrix
    Pline - transition matrix for diagonal boundary of triallele surface
    P - stores the ammount of density that should be moved from triallele domain (surf) to diagonal line boundary
    """
    # create the triallele domain of the nonsquare surface of the full density function
    surf = phi_to_surf(phi,x) # done
    
    surf = advance_surf_adi1(surf,U01surf,P1surf,x)
    surf = advance_surf_adi2(surf,U01surf,P2surf,x)
    surf = advance_surf_cov(surf,Csurf,x)
    surf = move_surf_density_to_bdry(x,surf,P)
    #surf = advance1D(x,surf,Pline) # this is an absorbing edge, so we'll just leave it alone - density lost!
    
    phi = surf_to_phi(surf,phi,x) # done
    return phi

####
# sampling methods

sample_cache = {}
def sample_cached(phi, ns, x):
    dx = grid_dx(x)
    dx3 = grid_dx3(x,dx)

    if type(ns) == int:
        ns = (ns,)
    else:
        if len(ns) == 1:
            ns = tuple(ns)
        else:
            ns = (ns[0],)

    # cache calculations of several large matrices
    # cache x**ii, x**jj, and x**kk 
    # just cache x[:,nuax,nuax]**ii, and then later use np.swapaxes
    key = (ns, tuple(x))
    if key not in sample_cache:
        this_cache = {}
        for ii in range(ns[0]+1):
            this_cache[ii] = x[:,nuax,nuax]**ii
            this_cache[-ii] = (1 - x[:,nuax,nuax] - x[nuax,:,nuax] - x[nuax,nuax,:]) ** ii
        sample_cache[key] = this_cache
    else:
        this_cache = sample_cache[key]

    F = np.zeros((ns[0]+1,ns[0]+1,ns[0]+1))
    prod_phi = dx3*phi
    for ii in range(len(F)):
        prod_x = prod_phi * this_cache[ii]
        for jj in range(len(F)):
            prod_y = prod_x * np.swapaxes(this_cache[jj],0,1)
            for kk in range(len(F)):
                if ii+jj+kk <= ns[0]:
                    F[ii,jj,kk] = quadrinomial(ns[0],ii,jj,kk) * np.sum(prod_y * np.swapaxes(this_cache[kk],0,2) * this_cache[-(ns[0]-ii-jj-kk)])

    F = TLSpectrum(F)
    return F

def quadrinomial(ns,ii,jj,kk):
    """
    returns ns! / (ii! * jj! * kk! * (ns-ii-jj-kk)!)
    """
    return np.exp(math.lgamma(ns+1) - math.lgamma(ii+1) - math.lgamma(jj+1) - math.lgamma(kk+1) - math.lgamma(ns-ii-jj-kk+1))

def ln_binomial(n,k):
    return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)

def fold_ancestral(F):
    """
    Takes an unfolded two locus frequency spectrum and folds it (assume don't konw ancestral state)
    """
    ns = len(F[:,0,0]) - 1
    for ii in range(ns+1):
        for jj in range(ns+1):
            for kk in range(ns+1):
                if F.mask[ii,jj,kk]:
                    continue
                p = ii + jj
                q = ii + kk
                if p > ns/2 and q > ns/2:
                    # Switch A/a and B/b, so AB becomes ab, Ab becomes aB, etc
                    F[ns-ii-jj-kk,kk,jj] += F[ii,jj,kk]
                    F.mask[ii,jj,kk] = True
                elif p > ns/2:
                    # Switch A/a, so AB -> aB, Ab -> ab, aB -> AB, and ab -> Ab
                    F[kk,ns-ii-jj-kk,ii] += F[ii,jj,kk]
                    F.mask[ii,jj,kk] = True
                elif q > ns/2:
                    # Switch B/b, so AB -> Ab, Ab -> AB, aB -> ab, and ab -> aB
                    F[jj,ii,ns-ii-jj-kk] += F[ii,jj,kk]
                    F.mask[ii,jj,kk] = True
                                        
    return F

def fold_lr(F):
    """
    Takes an unfolded two locus spectrum and folds based on left/right allele
    """
    ns = len(F[:,0,0]) - 1
    for ii in range(ns+1):
        for jj in range(ns+1):
            for kk in range(ns+1):
                if F.mask[ii,jj,kk]:
                    continue
                if kk > jj:
                    F[ii,kk,jj] += F[ii,jj,kk]
                    F[ii,jj,kk] = 0
                    F.mask[ii,jj,kk] = True
    return F
                    
def extrap_dt_pts(temps):
    # in form of temps[dt][numpts]
    dts = sorted(temps.keys())[::-1]
    gridpts = sorted(temps[dts[0]].keys())
    F0 = dadi.Numerics.quadratic_extrap((temps[dts[0]][gridpts[0]],temps[dts[0]][gridpts[1]],temps[dts[0]][gridpts[2]]),(1./gridpts[0],1./gridpts[1],1./gridpts[2]))
    F1 = dadi.Numerics.quadratic_extrap((temps[dts[1]][gridpts[0]],temps[dts[1]][gridpts[1]],temps[dts[1]][gridpts[2]]),(1./gridpts[0],1./gridpts[1],1./gridpts[2]))
    F2 = dadi.Numerics.quadratic_extrap((temps[dts[2]][gridpts[0]],temps[dts[2]][gridpts[1]],temps[dts[2]][gridpts[2]]),(1./gridpts[0],1./gridpts[1],1./gridpts[2]))

    return dadi.Numerics.quadratic_extrap((F0,F1,F2),(dts[0],dts[1],dts[2]))


def array_to_spectrum(phi):
    """
    now handled by TLSpectrum_mod.TLSpectrum
    """
    ns = len(phi)-1
    phi = dadi.Spectrum(phi)
    phi = dadi.Spectrum(phi)
    phi.mask[0,0,0] = True
    phi.mask[0,:,0] = True
    phi.mask[0,0,:] = True
    for ii in range(len(phi)):
        for jj in range(len(phi)):
            for kk in range(len(phi)):
                if ii+jj+kk > ns:
                    phi.mask[ii,jj,kk] = True

    for ii in range(len(phi)):
        phi.mask[ii,ns-ii,0] = True
        phi.mask[ii,0,ns-ii] = True
    
    return phi

def to_single_locus(phi):
    ns = len(phi)-1
    fsA = dadi.Spectrum(np.zeros(ns+1))
    fsB = dadi.Spectrum(np.zeros(ns+1))
    for ii in range(ns+1):
        for jj in range(ns+1-ii):
            for kk in range(ns+1-ii-jj):
                if phi.mask[ii,jj,kk]:
                    continue
                p = ii + jj
                q = ii + kk
                fsA[p] += phi[ii,jj,kk]
                fsB[q] += phi[ii,jj,kk]
    return fsA,fsB

def mean_r2(F):
    ns = len(F)-1
    tot = np.sum(F)
    r2s = []
    weights = []
    for ii in range(ns+1):
        for jj in range(ns+1-ii):
            for kk in range(ns+1-ii-jj):
                if F.mask[ii,jj,kk]:
                    continue
                p = (ii + jj)/float(ns)
                q = (ii + kk)/float(ns)
                pAB = ii/float(ns)
                D = pAB - p*q
                r2s.append(D**2/(p*(1-p)*q*(1-q)))
                weights.append(F[ii,jj,kk])
    return np.sum(np.array(r2s)*np.array(weights)/tot)


# gets too big when I need to cache hundreds of different sample sizes
#projection_cache = {}
def cached_projection(proj_to, proj_from, hits):
    """
    Coefficients for projection from a larger size to smaller
    proj_to: Number of samples to project down to
    proj_from: Number of samples to project from
    hits: Number of derived alleles projecting from - tuple of (n1,n2,n3)
    """
#    key = (proj_to, proj_from, hits)
#    try:
#        return projection_cache[key]
#    except KeyError:
#        pass
    
    X1, X2, X3 = hits
    X4 = proj_from - X1 - X2 - X3
    proj_weights = np.zeros((proj_to+1,proj_to+1,proj_to+1))
    for ii in range(X1+1):
        for jj in range(X2+1):
            for kk in range(X3+1):
                ll = proj_to - ii - jj - kk
                if ll > X4 or ll <0:
                    continue
                f = ln_binomial(X1,ii) + ln_binomial(X2,jj) + ln_binomial(X3,kk) + ln_binomial(X4,ll) - ln_binomial(proj_from,proj_to)
                proj_weights[ii,jj,kk] = np.exp(f)
                
    
#    projection_cache[key] = proj_weights
    return proj_weights
    
def project(F_from, proj_to):
    proj_from = len(F_from)-1
    if proj_to == proj_from:
        return F_from
    elif proj_to > proj_from:
        print('nope!')
        return F_from
    else:
        F_proj = np.zeros((proj_to+1,proj_to+1,proj_to+1))
        for X1 in range(proj_from):
            for X2 in range(proj_from):
                for X3 in range(proj_from):
                    if F_from.mask[X1,X2,X3] == False:
                        hits = (X1,X2,X3)
                        proj_weights = cached_projection(proj_to,proj_from,hits)
                        F_proj += proj_weights * F_from[X1,X2,X3]
                        
        return TLSpectrum(F_proj)

## for projection to genotype spectrum
def pairings(n):
    return math.factorial(n)/(math.factorial(n/2)*2**(n/2))

def binom(n,k):
    return math.factorial(n)/math.factorial(k)/math.factorial(n-k)

def multinomial(n,i,j,k):
    if i+j+k != n:
        return "nein"
    return math.factorial(n)/math.factorial(i)/math.factorial(j)/math.factorial(k)

def genotypes_prob_4(n,colors,hits):
    (nR,nG,nB,nY) = colors
    (nRR,nGG,nBB,nYY,nRG,nRB,nRY,nGB,nGY,nBY) = hits
    if sum(hits) != n/2 or sum(colors) != n: return "nuh uh"
    return pairings(nR-nRG-nRB-nRY)*pairings(nG-nRG-nGB-nGY)*pairings(nB-nRB-nGB-nBY)*pairings(nY-nRY-nGY-nBY)*binom(nR,nRG+nRB+nRY)*multinomial(nRG+nRB+nRY,nRG,nRB,nRY)*binom(nG,nRG+nGB+nGY)*multinomial(nRG+nGB+nGY,nRG,nGB,nGY)*binom(nB,nRB+nGB+nBY)*multinomial(nRB+nGB+nBY,nRB,nGB,nBY)*binom(nY,nRY+nGY+nBY)*multinomial(nRY+nGY+nBY,nRY,nGY,nBY)*math.factorial(nRG)*math.factorial(nRB)*math.factorial(nRY)*math.factorial(nGB)*math.factorial(nGY)*math.factorial(nBY)

def possible_genotypes_4(n,colors):
    nR,nG,nB,nY = colors
    gl = []
    for pure_red in range(nR/2+1):
        for pure_green in range(nG/2+1):
            for pure_blue in range(nB/2+1):
                for pure_yellow in range(nY/2+1):
                    mixed_red = nR-2*pure_red
                    mixed_green = nG-2*pure_green
                    mixed_blue = nB-2*pure_blue
                    mixed_yellow = nY-2*pure_yellow
                    for mixRG in range(min(mixed_red,mixed_green)+1):
                        for mixRB in range(min(mixed_red-mixRG,mixed_blue)+1):
                            for mixRY in range(mixed_red-mixRG-mixRB,min(mixed_red-mixRG-mixRB,mixed_yellow)+1):
                                for mixGB in range(min(mixed_green-mixRG,mixed_blue-mixRB)+1):
                                    for mixGY in range(mixed_green-mixRG-mixGB,min(mixed_green-mixRG-mixGB,mixed_yellow-mixRY)+1):
                                        mixBY1 = mixed_blue-mixRB-mixGB
                                        mixBY2 = mixed_yellow-mixRY-mixGY
                                        if mixBY1 == mixBY2:
                                            gl.append((pure_red,pure_green,pure_blue,pure_yellow,mixRG,mixRB,mixRY,mixGB,mixGY,mixBY1))
    return gl

prob_cache = {} # store possible genotypes and their relative probabilities among all possible genotype sets
def cached_genotype_exact_projection(n,haplotype_counts):
    key = (n,haplotype_counts)
    try:
        return prob_cache[key]
    except KeyError:
        pass
    
    gl = possible_genotypes_4(n,haplotype_counts)
    tot = float(pairings(n))
    probs = []
    prob_cache.setdefault(key,{})
    for hits in gl:
        prob_cache[key][hits] = float(genotypes_prob_4(n,haplotype_counts,hits))/tot
    return prob_cache[key]

def genotype_spectrum_from_F(F):
    n = len(F)-1
    ng = n/2
    tot = pairings(n)
    G = np.zeros((ng+1,ng+1,ng+1,ng+1,ng+1,ng+1,ng+1,ng+1,ng+1))
    for ii in range(n+1):
        for jj in range(n+1):
            for kk in range(n+1):
                if F[ii,jj,kk] > 0:# and F.mask[ii,jj,kk] == False:
                    nAB,nAb,naB,nab = ii,jj,kk,n-ii-jj-kk
                    gen_probs = cached_genotype_exact_projection(n,(nAB,nAb,naB,nab))
                    for gens in gen_probs.keys():
                        g1,g2,g3,g4,g5,g6,g7,g8,g9,g10 = gens
                        prob = gen_probs[gens]
                        G[g1,g2,g3,g4,g5,g6,g7,g8,g9] += F[nAB,nAb,naB]*prob
    return G

def observed_genotype_spectrum_from_F(F):
    n = len(F)-1
    ng = n/2
    G = np.zeros((ng+1,ng+1,ng+1,ng+1,ng+1,ng+1,ng+1,ng+1))
    for ii in range(n+1):
        for jj in range(n+1):
            for kk in range(n+1):
                if F[ii,jj,kk] > 0:# and F.mask[ii,jj,kk] == False:
                    nAB,nAb,naB,nab = ii,jj,kk,n-ii-jj-kk
                    gen_probs = cached_genotype_exact_projection(n,(nAB,nAb,naB,nab))
                    for gens in gen_probs.keys():
                        g1,g2,g3,g4,g5,g6,g7,g8,g9,g10 = gens
                        o1,o2,o3,o4,o5,o6,o7,o8,o9 = g1,g5,g2,g6,g7+g8,g9,g3,g10,g4
                        prob = gen_probs[gens]
                        G[o1,o2,o3,o4,o5,o6,o7,o8] += F[nAB,nAb,naB]*prob
    return G

def observed_genotype_spectrum_dict_from_F(F):
    """
    Gdict has keys n (num individuals in sample), observed genotypes
    """
    n = len(F)-1
    ng = n/2
    Gdict = {}
    Gdict.setdefault(ng,{})
    for ii in range(n+1):
        for jj in range(n+1):
            for kk in range(n+1):
                if F.mask[ii,jj,kk] == False:
                    nAB,nAb,naB,nab = ii,jj,kk,n-ii-jj-kk
                    gen_probs = cached_genotype_exact_projection(n,(nAB,nAb,naB,nab))
                    for gens in gen_probs.keys():
                        g1,g2,g3,g4,g5,g6,g7,g8,g9,g10 = gens
                        o1,o2,o3,o4,o5,o6,o7,o8,o9 = g1,g5,g2,g6,g7+g8,g9,g3,g10,g4
                        prob = gen_probs[gens]
                        try:
                            Gdict[ng][(o1,o2,o3,o4,o5,o6,o7,o8)] += F[nAB,nAb,naB]*prob
                        except KeyError:
                            Gdict[ng][(o1,o2,o3,o4,o5,o6,o7,o8)] = F[nAB,nAb,naB]*prob
    return Gdict

from . import projection_genotypes

genotype_projection_cache = {} # this cache can get very large for n_from large (>~100)
def projection_cache_Gdict(n_from,n_to,hits):
    key = (n_from,n_to,hits)
    try:
        return genotype_projection_cache[key]
    except KeyError:
        pass
    
#    g1,g2,g3,g4,g5,g6,g7,g8 = hits
#    g9 = n_from - g1 - g2 - g3 - g4 - g5 - g6 - g7 - g8
#    weights_to = {}
#    for o1 in range(0,g1+1):
#        for o2 in range(0,g2+1):
#            for o3 in range(0,g3+1):
#                for o4 in range(0,g4+1):
#                    for o5 in range(0,g5+1):
#                        for o6 in range(0,g6+1):
#                            for o7 in range(0,g7+1):
#                                for o8 in range(0,g8+1):
#                                    o9 = n_to-o1-o2-o3-o4-o5-o6-o7-o8
#                                    if o9 < 0 or o9 > g9:
#                                        continue
#                                    else:
#                                        weight = np.exp(ln_binomial(g1,o1) + ln_binomial(g2,o2) + ln_binomial(g3,o3) + ln_binomial(g4,o4) + ln_binomial(g5,o5) + ln_binomial(g6,o6) + ln_binomial(g7,o7) + ln_binomial(g8,o8) + ln_binomial(g9,o9) - ln_binomial(n_from,n_to))
#                                        weights_to[(o1,o2,o3,o4,o5,o6,o7,o8)] = weight
#

    weights_to = projection_genotypes.projection_genotypes(n_from,n_to,hits)
    genotype_projection_cache[key] = weights_to
    return weights_to

def project_Gdict(G,n_from,n_to):
    """ n_from and n_to are the individual counts, not haplotype counts (so ns/2) """
    G_to = {}
    G_to.setdefault(n_to,{})
    for genotypes in G[n_from].keys():
        weights = projection_cache_Gdict(n_from,n_to,genotypes)
        for projected_gens in weights.keys():
            try:
                G_to[n_to][projected_gens] += weights[projected_gens]*G[genotypes]
            except KeyError:
                G_to[n_to][projected_gens] = weights[projected_gens]*G[genotypes]
    return G_to

def misidentification(F,p):
    """
    with probability p, ancestral state is misidentified
    with prob p(1-p) A -> a but B correct
    with prob p(1-p) B -> b but A correct
    with prob p^2 A -> a and B -> b
    """
    F_new = np.zeros(np.shape(F))
    ns = len(F) - 1
    for ii in range(len(F)):
        for jj in range(len(F)):
            for kk in range(len(F)):
                if F.mask[ii,jj,kk] == True:
                    continue
                ll = ns - ii - jj - kk
                weight = F[ii,jj,kk]
                F_new[ii,jj,kk] += weight * (1 - 2*p + p**2)
                F_new[kk,ll,ii] += weight * p * (1-p)
                F_new[jj,ii,ll] += weight * p * (1-p)
                F_new[ll,kk,jj] += weight * p**2
    return TLSpectrum(F_new)

def misidentification_genotype_dict(G,p):
    """
    same misidentification probabilities as misidentification(F,p)
    with prob p(1-p), AA -> aa, aa -> AA, Aa stays the same, B/b stay same
    with prob p(1-p), 
    """
    G_new = {}
    n = G.keys()[0] # first key of G is number of individuals in sample
    G_new.setdefault(n,{})
    for genotypes in G[n].keys():
        G_new[n].setdefault(genotypes,0.0)
    for genotypes in G[n].keys():
        weight = G[n][genotypes]
        g1,g2,g3,g4,g5,g6,g7,g8 = genotypes
        g9 = n-g1-g2-g3-g4-g5-g6-g7-g8
        G_new[n][(g1,g2,g3,g4,g5,g6,g7,g8)] += weight * (1 - 2*p + p**2)
        G_new[n][(g7,g8,g9,g4,g5,g6,g1,g2)] += weight * p * (1-p)
        G_new[n][(g3,g2,g1,g6,g5,g4,g9,g8)] += weight * p * (1-p)
        G_new[n][(g9,g8,g7,g6,g5,g4,g3,g2)] += weight * p**2
    return G_new

def genotype_exp_data_to_arrays(G_exp,G_data):
    """
    return 1D spectra, to perform likelihood calcs
    so given two dicts, one for genotype freq expectations, one for data, returns spectrum arrays of data that can be passed to inference methods
    """
    n_exp = G_exp.keys()[0]
    n_data = G_data.keys()[0]
    if n_exp != n_data:
        return 'must have same number of sampled individuals'
    exp_arr = np.zeros(len(G_exp[n_exp]))
    data_arr = np.zeros(len(G_exp[n_exp]))
    gl = G_exp[n_exp].keys()
    for ii in range(len(gl)):
        g = gl[ii]
        exp_arr[ii] = G_exp[n_exp][g]
        try:
            data_arr[ii] = G_data[n_exp][g]
        except KeyError:
            pass
    
    e = dadi.Spectrum(np.concatenate((np.array([0]),exp_arr,np.array([0]))))
    d = dadi.Spectrum(np.concatenate((np.array([0]),data_arr,np.array([0]))))
    return e,d

def condition(F,i):
    """
    condition on seeing nA = i
    Fcond[i,j]: i - number AB, j - number of aB
    i ranges from 0 to i, j ranges from 0 to n-i
    """
    ns = len(F)-1
    Fcond = np.zeros((i+1,ns-i+1))
    for ii in range(ns):
        for jj in range(ns):
            for kk in range(ns):
                if F.mask[ii,jj,kk] == False:
                    if ii+jj == i:
                        Fcond[ii,kk] = F[ii,jj,kk]
    return Fcond

# Try importing cythonized versions of several slow methods. These imports should overwrite the Python code defined above.
try:
    from .transition1D import transition1D
    from .transition1 import transition1
    from .transition2 import transition2
    from .transition3 import transition3
    from .transition12 import transition12
    from .transition13 import transition13
    from .transition23 import transition23
    from .surface_interaction import surface_interaction
except ImportError:
    print("using numpy versions")
    pass
