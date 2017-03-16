import numpy as np
cimport numpy as np

def transition2(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] dx, U01, gammaA, gammaB, rho, nu, hA=.5, hB=.5):
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
