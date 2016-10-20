## cythonize transition1 and transition2 in numerics
import numpy as np
cimport numpy as np

def transition2(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] dx,np.ndarray[np.float64_t, ndim=2] U01, np.float64_t sig1, np.float64_t sig2):
    cdef int ii
    cdef int jj
    cdef np.ndarray[np.float64_t, ndim=3] PV = np.zeros((len(x),3,len(x)))
    cdef np.ndarray[np.float64_t, ndim=3] PM = np.zeros((len(x),3,len(x)))
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((len(x),len(x)))
    cdef np.ndarray[np.float64_t, ndim=1] V = x*(1-x)
    cdef np.ndarray[np.float64_t, ndim=1] M = x*(1-x)
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
