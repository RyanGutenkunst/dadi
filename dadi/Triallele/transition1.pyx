## cythonize transition1 and transition2 in numerics
import numpy as np
cimport numpy as np

def transition1(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] dx,np.ndarray[np.float64_t, ndim=2] U01, np.float64_t sig1, np.float64_t sig2):
    """
    Implicit transition matrix for the ADI components of the discretization of the diffusion
    Time scaled by 2N, with variance and mean terms x(1-x) and \sigma*x(1-x), resp.
    Store the tridiagonal elements of the matrices, which need to be adjusted by I + dt*P, where I is the identity matrix
    """
    cdef int ii
    cdef int jj
    cdef np.ndarray[np.float64_t, ndim=3] PV = np.zeros((len(x),3,len(x)))
    cdef np.ndarray[np.float64_t, ndim=3] PM = np.zeros((len(x),3,len(x)))
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((len(x),len(x)))
    cdef np.ndarray[np.float64_t, ndim=1] V = x*(1-x)
    cdef np.ndarray[np.float64_t, ndim=1] M = x*(1-x)
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
