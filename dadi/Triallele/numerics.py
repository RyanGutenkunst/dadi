"""
Integration numerics for integration
"""

# RNG: Comments!
# Methods to construct transition matrices, and integrate the model forward in time

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

from bdry_injection import bdry_inj
import transition1D


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
    need the proper DXX, as commented out below
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

def transition1(x,dx,U01,sig1,sig2):
    """
    Implicit transition matrix for the ADI components of the discretization of the diffusion
    Time scaled by 2N, with variance and mean terms x(1-x) and \sigma*x(1-x), resp.
    Store the tridiagonal elements of the matrices, which need to be adjusted by I + nu/dt*P, where I is the identity matrix
    """
    P = np.zeros((len(x),3,len(x)))
    for jj in range(len(x)):
        A = np.zeros((len(x),len(x)))
        if jj > 0 and x[jj] != 1:
            V = x*(1-x)
            V[np.where(U01[:,jj] == 1)[0][:-1][-1]] = 0
            for ii in np.where(U01[:,jj] == 1)[0][:-1]:
                if ii == 0:
                    A[ii,ii] =  - 1/(2*dx[ii]) * ( -V[ii]/(x[ii+1]-x[ii]) )
                    A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) )
                elif ii == np.where(U01[:,jj] == 1)[0][:-1][-1]:
                    A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) ) * 2
                    A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) ) * 2
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
        
        if sig1 != 0:
            ## with two sites, the two seletion coefficients interfere
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

        P[jj,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
        P[jj,1,:] = np.diagonal(A)
        P[jj,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    return P

def transition2(x,dx,U01,sig1,sig2):
    P = np.zeros((len(x),3,len(x)))
    for ii in range(len(x)):
        A = np.zeros((len(x),len(x)))
        if ii > 0 and x[ii] != 1:
            V = x*(1-x)
            V[np.where(U01[ii,:] == 1)[0][:-1][-1]] = 0
            for jj in np.where(U01[ii,:] == 1)[0]:
                if jj == 0:
                    A[jj,jj] =  - 1/(2*dx[jj]) * ( -V[jj]/(x[jj+1]-x[jj]) )
                    A[jj,jj+1] = - 1/(2*dx[jj]) * ( V[jj+1]/(x[jj+1]-x[jj]) )
                elif jj == np.where(U01[ii,:] == 1)[0][:-1][-1]:
                    A[jj,jj-1] = - 1/(2*dx[jj]) * ( V[jj-1]/(x[jj]-x[jj-1]) ) * 2
                    A[jj,jj] = - 1/(2*dx[jj]) * ( -V[jj]/(x[jj]-x[jj-1]) ) * 2
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
        
        if sig2 != 0:
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
        
        P[ii,0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
        P[ii,1,:] = np.diagonal(A)
        P[ii,2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    return P

def transition12(x,dx,U01):
    """
    Transition matrix for the covariance term of the diffusion operator, with term D_{xy} (-x*y*phi)
    As with the ADI components, final transition matrix is given by I + dt/nu*P
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

def transition_line(x,dx,sig1,sig2):
    """
    Transition matrix for loss of ancestral allele, corresponding to states along the diagonal border of the domain
    Selection strength is given by \tilde{sig} = (sig1 - sig2)/(1+sig2) ... probably not. Working on this.
    """
    P = np.zeros((3,len(x)))
    A = np.zeros((len(x),len(x)))
    V = x*(1-x)
    sig = sig1 - sig2
    for ii in range(len(x)):
        if ii == 0:
            A[ii,ii] =  - 1/(dx[ii]) * ( -V[ii]/(x[ii+1]-x[ii]) )
            A[ii,ii+1] = - 1/(dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) ) + sig / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
        elif ii == len(x) - 1:
            A[ii,ii-1] = - 1/(dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) ) - sig / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            A[ii,ii] = - 1/(dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) )
        else:
            A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) ) - sig / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) -V[ii]/(x[ii+1]-x[ii]) )
            A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) ) + sig / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
    
    P[0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
    P[1,:] = np.diagonal(A)
    P[2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
        
    return P

def transition_bdry(x):
    """
    Along the diagonal boundary, integrating forward in time using the ADI method incorrectly pushed density to be absorbed along the boundary instead of diffusing more parallel to the boundary.
    So instead, the bulk of the domain is integrated using the methods above, and we handle the thin strip of the domain along the diagonal boundary separately by applying a split diffusion method parallel and perpendicular to the boundary.
    
    Sept 3, 2015 - this is not what we use. For both better runtime and accuracy, we simply calculate the amount of density for each point that should be lost to the boundary and remove that amount from the grid points
    """
    x_new = np.linspace(0,1,len(x)-1) # we integrate along the diagonal just inside the diagonal boundary where x1+x2=1, so the length of that array is one less that the total grid length
    V = x[:-1] * (1-x[1:])
    Pz = np.zeros((3,len(x_new)))
    A = np.zeros((len(x_new),len(x_new)))
    dx = grid_dx(x_new)
    for ii in range(len(x_new)):
        if ii == 0:
            A[ii,ii] =  - 1/(2*dx[ii]) * ( -V[ii]/(x_new[ii+1]-x_new[ii]) )
            A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x_new[ii+1]-x_new[ii]) )
        elif ii == len(x_new) - 1:
            A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x_new[ii]-x_new[ii-1]) )
            A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x_new[ii]-x_new[ii-1]) )
        else:
            A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x_new[ii]-x_new[ii-1]) )
            A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x_new[ii]-x_new[ii-1]) -V[ii]/(x_new[ii+1]-x_new[ii]) )
            A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x_new[ii+1]-x_new[ii]) )
    
    Pz[0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
    Pz[1,:] = np.diagonal(A)
    Pz[2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    
    Pzperp = np.zeros((3,len(x))) 
    A = np.zeros((len(x),len(x)))
    dx = grid_dx(x)
    V = x*(1-x)
    for ii in range(len(x)):
        if ii == 0:
            A[ii,ii] =  - 1/(2*dx[ii]) * ( -V[ii]/(x[ii+1]-x[ii]) )
            A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) )
        elif ii == len(x) - 1:
            A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) )
            A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) )
        else:
            A[ii,ii-1] = - 1/(2*dx[ii]) * ( V[ii-1]/(x[ii]-x[ii-1]) )
            A[ii,ii] = - 1/(2*dx[ii]) * ( -V[ii]/(x[ii]-x[ii-1]) -V[ii]/(x[ii+1]-x[ii]) )
            A[ii,ii+1] = - 1/(2*dx[ii]) * ( V[ii+1]/(x[ii+1]-x[ii]) )
    
    Pzperp[0,:] = np.concatenate(( np.array([0]), np.diagonal(A,-1) ))
    Pzperp[1,:] = np.diagonal(A)
    Pzperp[2,:] = np.concatenate(( np.diagonal(A,1), np.array([0]) ))
    
    return Pz,Pzperp

# Use tridiag to solve the adi components, and scipy's sparse methods for the covariance term

def advance_adi(U,U01,P1,P2,x,ii):
    if np.mod(ii,2) == 0:
        for jj in range(len(x)):
            if np.sum(U01[:,jj]) > 1:
                U[:,jj] = dadi.tridiag.tridiag(P1[jj,0,:],P1[jj,1,:],P1[jj,2,:],U[:,jj])
        for ii in range(len(x)):
            if np.sum(U01[ii,:]) > 1:
                U[ii,:] = dadi.tridiag.tridiag(P2[ii,0,:],P2[ii,1,:],P2[ii,2,:],U[ii,:])
    else:
        for ii in range(len(x)):
            if np.sum(U01[ii,:]) > 1:
                U[ii,:] = dadi.tridiag.tridiag(P2[ii,0,:],P2[ii,1,:],P2[ii,2,:],U[ii,:])
        for jj in range(len(x)):
            if np.sum(U01[:,jj]) > 1:
                U[:,jj] = dadi.tridiag.tridiag(P1[jj,0,:],P1[jj,1,:],P1[jj,2,:],U[:,jj])
    return U

def advance_adi_x(U,U01,P1,P2,x):
    for jj in range(len(x)):
        if np.sum(U01[:,jj]) > 1:
            U[:,jj] = dadi.tridiag.tridiag(P1[jj,0,:],P1[jj,1,:],P1[jj,2,:],U[:,jj])

    return U

def advance_adi_y(U,U01,P1,P2,x):
    for ii in range(len(x)):
        if np.sum(U01[ii,:]) > 1:
            U[ii,:] = dadi.tridiag.tridiag(P2[ii,0,:],P2[ii,1,:],P2[ii,2,:],U[ii,:])
            
    return U

def advance_cov(U,C,x,dx):
    U = ( C * U.reshape(len(x)**2)).reshape(len(x),len(x))
    return U

def advance_line(U,P_line,x,dx):
    u = np.diag(np.fliplr(U))
    u = dadi.tridiag.tridiag(P_line[0,:],P_line[1,:],P_line[2,:],u)
    for ii in range(len(x)):
        U[ii,len(x)-ii-1] = u[ii]
    return U


def advance_bdry(U,Pz,Pzperp,x,dx):
    """
    U_new is the new domain that is square to the diagonal boundary, and has dimensions (length(x)-1) x length(x)
    
    9/3/15 - no longer used, see comment in transition_bdry function
    """
    x_new = np.linspace(0,1,len(x)-1)
    U_new = np.zeros((len(x_new),len(x)))
    U_new[:,1] = np.diag(np.fliplr(U),1)

    for ii in range(len(x_new)):
        U[ii,len(x_new) - 1 - ii] = 0
        

    U_new[:,1] = dadi.tridiag.tridiag(Pz[0],Pz[1],Pz[2],U_new[:,1])
    for ii in range(len(x_new))[1:-1]:
        U_new[ii,:] = dadi.tridiag.tridiag(Pzperp[0],Pzperp[1],Pzperp[2],U_new[ii,:])

    U[0,len(x)-2] += U_new[0,1]
    U_new[0,1] = 0
    U[len(x)-2,0] += U_new[len(x_new)-1,1]
    U_new[len(x_new)-1,1] = 0

    ## feb 18, 2015 - only push to boundary, not back into interior
    #for ii in range(len(x_new))[1:-1]:
    #    jj = 0
    #    U[ii-jj/2, len(x)-1 - ii - jj/2] += U_new[ii,jj] / 2
    #    U[ii+1-jj/2, len(x) - 1 - ii - jj/2 - 1] += U_new[ii,jj] / 2
    #    U_new[ii,jj] = 0
    #    
    #    jj = 1
    #    U[ii,len(x)-1-ii-jj] += np.sum(U_new[ii,jj:]* dx[jj:]) / dx[jj]
    #    U_new[ii,jj:] = 0
    ##
    """
    Here, return density to original domain.
    If there is a corresponding grid point in the original domain, simply place the density on that grid point.
    If the grid point in the new domain is between two grid points in the original domain, split the density between them.
    For grid points in the new domain that extend beyond the boundary of the original domain, the density is assumed to have fixed along the boundary at those points that it extends beyond.
    """
    
    ### 10/14 this is really eating up most of the run time. transfering density back to the full domain from the boundary method
    
    U = bdry_inj(U,U_new,x,dx,x_new)
#    for ii in range(len(x_new))[1:-1]:
#        for jj in range(len(x))[:2*np.min((len(x_new)-ii-1,ii)) + 1 + 1]: # out to edge of full domain
#            if np.mod(jj,2) == 0: # falls in between points in domain
#                if ii-jj/2 == 0 or len(x)-1 - ii - jj/2 == 0:
#                    U[ii-jj/2, len(x)-1 - ii - jj/2] += U_new[ii,jj] 
#                else:
#                    U[ii-jj/2, len(x)-1 - ii - jj/2] += U_new[ii,jj] / 2
#                
#                if ii+1-jj/2 == 0 or len(x) - 1 - ii - jj/2 - 1 == 0:
#                    U[ii+1-jj/2, len(x) - 1 - ii - jj/2 - 1] += U_new[ii,jj] 
#                else:
#                    U[ii+1-jj/2, len(x) - 1 - ii - jj/2 - 1] += U_new[ii,jj] / 2
#                U_new[ii,jj] = 0
#            elif jj == 2*np.min((len(x_new)-ii-1,ii)) + 1: # for points beyond edge of domain, sum and fix on domain boundary
#                if ii > len(x)/2:
#                    U[len(x)-1-jj,0] += np.sum(U_new[ii,jj:]* dx[jj:]) / dx[jj] * 2
#                else:
#                    U[0,len(x)-1-jj] += np.sum(U_new[ii,jj:]* dx[jj:]) / dx[jj] * 2
#                U_new[ii,jj:] = 0
#            else: # falls on point
#                U[ii - (jj-1)/2,len(x) - 1 - ii - (jj+1)/2] += U_new[ii,jj]
#                U_new[ii,jj] = 0
    
    ### ^ slow ^ ####
    
    return U

## 1D methods - could be coopted from 

def transition1D_py(x,dx,dt,sig,nu):
    P = np.zeros((len(x),len(x)))
    for ii in range(len(x)):
        if ii == 0:
            P[ii,ii] = 1 + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            P[ii,ii+1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii]) + sig * dt/nu / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
        elif ii == len(x) - 1:
            P[ii,ii-1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1]) - sig * dt / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            P[ii,ii] = 1 + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1])
        else:
            P[ii,ii-1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1]) - sig * dt / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            P[ii,ii] = 1 + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1]) + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            P[ii,ii+1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii]) + sig * dt / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
    return P

def remove_diag_density_weights(x,dt,nu,sig1,sig2):
    dx = grid_dx(x)
    P1D = transition1D.transition1D(x,dx,dt,0.0,nu)
    P = np.zeros((len(x),len(x)))
    for dd in np.where(x<.5)[0][1:]:
        y = np.zeros(len(x))
        y[dd] = 1./dx[dd]
        y = advance1D(y,P1D)
        prob = y[0]*dx[0]
        for ii in range(len(x))[:-dd]:
            y = np.zeros(len(x))
            y[np.min((ii,len(x)-ii-dd-1))] += 1./dx[np.min((ii,len(x)-ii-dd-1))]
            y = advance1D(y,P1D)
            prob2 = y[0]*dx[0]
            P[ii,len(x)-1-dd-ii] = prob
            if dd == 1:
                P[ii,len(x)-1-dd-ii] += prob2
    P[:,0] = 0
    P[0,:] = 0
    return P

def remove_diag_density_weights_nonneutral(x,dt,nu,sig1,sig2):
    """
    Numerically determine the amount of density that should be lost to the diagonal boundary.
    Numerically integrate 1D array with initial point mass at z0, where z0 is the frequency x+y, integrated for time step dt.
    We then check the fraction of density that is lost to z=1.
    If sig1 or sig2 are nonzero, estimate the selection pressure on z as sig = sig1*x/(x+y) + sig2*y/(x+y)
    """
    dx = grid_dx(x)
    P = np.zeros((len(x),len(x)))
    for ii in range(len(x)):
        for jj in range(len(x)):
            if x[ii]+x[jj] < 1.0 and x[ii]+x[jj] > .75:
                sig = sig1*x[ii]/(x[ii]+x[jj]) + sig2*x[jj]/(x[ii]+x[jj]) 
                P1D = transition1D.transition1D(x,dx,dt,sig,nu)
                y = np.zeros(len(x))
                y[ii+jj] = 1./dx[ii+jj]
                y = advance1D(y,P1D)
                prob = y[-1]*dx[-1]
                P[ii,jj] = prob
                if ii+jj == len(x)-2:
                    if ii==1:
                        P1D = transition1D.transition1D(x,dx,dt,sig1,nu)
                        y = np.zeros(len(x))
                        y[ii] = 1./dx[ii]
                        y = advance1D(y,P1D)
                        prob2 = y[0]*dx[0]
                        P[ii,jj] += prob2
                    elif jj==1:
                        P1D = transition1D.transition1D(x,dx,dt,sig2,nu)
                        y = np.zeros(len(x))
                        y[jj] = 1./dx[jj]
                        y = advance1D(y,P1D)
                        prob2 = y[0]*dx[0]
                        P[ii,jj] += prob2
    
    P[:,0] = 0
    P[0,:] = 0
    return P

def advance1D(u,P):
    a = np.concatenate((np.array([0]),np.diag(P,-1)))
    b = np.diag(P)
    c = np.concatenate((np.diag(P,1),np.array([0])))
    u = dadi.tridiag.tridiag(a,b,c,u)
    return u

def sample1D(x,u,samples):
    dx = grid_dx(x)
    F = np.zeros(samples-1)
    for i in range(len(F))[1:-1]:
        j = i+1
        F[i] = math.factorial(samples)/(math.factorial(samples-j)*math.factorial(j)) * np.sum( u * x**j * (1-x)**(samples-j) * dx )
    return F

def sample(phi, ns, x, DXX):
    """
    For sampling the density function, we integrate against the trinomial distribution
    """
    if type(ns) == int:
        ns = tuple([ns])
    if not type(ns) == int:
        if len(ns) == 1:
            ns = tuple(ns)
        else:
            ns = tuple([ns[0]])
    #dx = grid_dx(x)
    F = np.zeros((ns[0]+1,ns[0]+1))
    for ii in range(len(F)):
        for jj in range(len(F)):
            if ii+jj < ns[0] and ii != 0 and jj != 0:
                #F[ii,jj] = math.factorial(ns)/(math.factorial(ii)*math.factorial(jj)*math.factorial(ns-ii-jj)) * int2(x, dx, phi*x[:,nuax]**ii*x[nuax,:]**jj*(1-x[:,nuax]-x[nuax,:])**(ns-ii-jj) )
                F[ii,jj] = trinomial(ns[0],ii,jj) * np.sum(DXX * phi*x[:,nuax]**ii*x[nuax,:]**jj*(1-x[:,nuax]-x[nuax,:])**(ns[0]-ii-jj) )
    return F

sample_cache = {}
def sample_cached(phi, ns, x, DXX):
    if type(ns) == int:
        ns = (ns,)
    else:
        if len(ns) == 1:
            ns = tuple(ns)
        else:
            ns = (ns[0],)
    
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
    F = Spectrum(F)
    F.extrap_x = x[1]
    F.mask[:,0] = True
    F.mask[0,:] = True
    for ii in range(len(F))[1:]:
        F.mask[ii,len(F)-ii-1:] = True
    return F

def trinomial(ns,ii,jj):
    """
    Return ns!/(ii! * jj! * (ns-ii-jj)!) for large values
    """
    return np.exp(math.lgamma(ns+1) - math.lgamma(ii+1) - math.lgamma(jj+1) - math.lgamma(ns-ii-jj+1))

def misidentification(Spectrum, p):
    """
    Given folded spectrum, and probability p that one of the derived alleles is the actual ancestral allele
    Then refold to return folded spectrum
    """
    F = np.zeros((len(Spectrum),len(Spectrum)))
    for ii in range(len(Spectrum))[1:-1]:
        for jj in range(len(Spectrum))[1:ii+1]:
            if ii+jj < len(Spectrum):
                F[ii,jj] += (1 - p) * Spectrum[ii,jj]
                F[len(Spectrum)-1-ii-jj,jj] += p/2. * Spectrum[ii,jj]
                F[ii,len(Spectrum)-1-ii-jj] += p/2. * Spectrum[ii,jj]
    
    return fold(Spectrum(F))
    
def fold(spectrum):
    """
    Given a frequency spectrum over the full domain, fold into a spectrum with major and minor derived alleles
    """
    spectrum = Spectrum(spectrum)
    if spectrum.mask[1,2] == True:
        print "error: trying to fold a spectrum that is already folded"
        return spectrum
    else:
        spectrum = (spectrum + np.transpose(spectrum))
        for ii in range(len(spectrum)):
            spectrum[ii,ii] /= 2
        spectrum.mask[0,:] = True
        spectrum.mask[:,0] = True
        for ii in range(len(spectrum)):
            spectrum.mask[ii,ii+1:] = True
            spectrum.mask[ii,len(spectrum)-1-ii:] = True
        return spectrum


def sample_bootstrap(data):
    
    # create cdf for sampling from data
    cdf = np.zeros(len(data)**2)
    for ii in range(len(data)**2)[1:]:
        if data.mask[ii/len(data),ii%len(data)] == False:
            cdf[ii] = cdf[ii-1] + data[ii/len(data),ii%len(data)]
        else:
            cdf[ii] = cdf[ii-1]
    
    cdf = cdf/cdf[-1]
        
    # 
    fs = np.zeros((len(data),len(data)))
    for ii in range(int(np.sum(data))):
        jj = np.min(np.where(np.random.rand() < cdf))
        fs[jj/len(data),jj%len(data)] += 1
    #
    fs = Spectrum(fs)
    fs.mask[:,0] = True
    fs.mask[0,:] = True
    for ii in range(len(fs)):
        for jj in range(len(fs)):
            if ii < jj or ii + jj >= len(fs)-1:
                fs.mask[ii,jj] = True
                
    return fs


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
