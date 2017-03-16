import numpy as np
cimport numpy as np

def transition3(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] dx, U01, gammaA, gammaB, rho, nu, hA=.5, hB=.5):
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
