import numpy as np
cimport numpy as np
from scipy.sparse import lil_matrix

def transition12(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] dx, np.ndarray[np.float64_t, ndim=3] U01):
    """
    Covariance discretization for slice along axes 1 and 2
    """
    cdef int ii
    cdef int jj
    cdef int kk
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

