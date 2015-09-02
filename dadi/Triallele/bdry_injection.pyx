### to convert using cython
import numpy as np
cimport numpy as np

def bdry_inj(np.ndarray[np.float64_t, ndim=2] U, np.ndarray[np.float64_t, ndim=2] U_new, np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] dx, np.ndarray[np.float64_t, ndim=1] x_new):
    cdef int ii
    cdef int jj
    #assert U.dtype == DTYPE and U_new.dtype == DTYPE and x.dtype == DTYPE and dx.dtype == DTYPE and x_new.dtype == DTYPE
    for ii in range(len(x_new))[1:-1]:
        for jj in range(len(x))[:2*min((len(x_new)-ii-1,ii)) + 1 + 1]: # out to edge of full domain
            if jj%2 == 0: # falls in between points in domain
                if ii-jj/2 == 0 or len(x)-1 - ii - jj/2 == 0:
                    U[ii-jj/2, len(x)-1 - ii - jj/2] += U_new[ii,jj] 
                else:
                    U[ii-jj/2, len(x)-1 - ii - jj/2] += U_new[ii,jj] / 2
                
                if ii+1-jj/2 == 0 or len(x) - 1 - ii - jj/2 - 1 == 0:
                    U[ii+1-jj/2, len(x) - 1 - ii - jj/2 - 1] += U_new[ii,jj] 
                else:
                    U[ii+1-jj/2, len(x) - 1 - ii - jj/2 - 1] += U_new[ii,jj] / 2
                #U_new[ii,jj] = 0
            elif jj == 2*min((len(x_new)-ii-1,ii)) + 1: # for points beyond edge of domain, sum and fix on domain boundary
                if ii > len(x)/2:
                    U[len(x)-1-jj,0] += sum(U_new[ii,jj:]* dx[jj:]) / dx[jj] * 2
                else:
                    U[0,len(x)-1-jj] += sum(U_new[ii,jj:]* dx[jj:]) / dx[jj] * 2
                #U_new[ii,jj:] = 0
            else: # falls on point
                U[ii - (jj-1)/2,len(x) - 1 - ii - (jj+1)/2] += U_new[ii,jj]
                #U_new[ii,jj] = 0
    return U
