import numpy as np
cimport numpy as np
from scipy.sparse import lil_matrix

def transition13(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] dx, np.ndarray[np.float64_t, ndim=3] U01):
    """
    Covariance discretization for slice along axes 1 and 2
    """
    cdef int ii
    cdef int jj
    cdef int kk
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
