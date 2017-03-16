import numpy as np
cimport numpy as np

def transition1(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] dx, np.ndarray[np.float64_t, ndim=3] U01, gammaA, gammaB, rho, nu, hA=.5, hB=.5):
    """
    Transition matrix for ADI method along first axis
    """
    cdef int ii
    cdef int jj
    cdef int kk
    cdef np.ndarray[np.float64_t, ndim=4] P = np.zeros((len(x),len(x),3,len(x)))
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((len(x),len(x)))
    cdef np.ndarray[np.float64_t, ndim=1] V = x*(1-x)
    cdef np.ndarray[np.float64_t, ndim=1] M = x*(1-x)
    tol = 1e-12
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
