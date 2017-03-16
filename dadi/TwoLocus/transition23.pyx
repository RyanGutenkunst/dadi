import numpy as np
cimport numpy as np
from scipy.sparse import lil_matrix

def transition23(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] dx, np.ndarray[np.float64_t, ndim=3] U01):
    """
    Covariance discretization for slice along axes 1 and 2
    """
    cdef int ii
    cdef int jj
    cdef int kk
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
