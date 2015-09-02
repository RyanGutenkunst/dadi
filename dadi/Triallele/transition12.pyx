import numpy as np
cimport numpy as np
from scipy.sparse import lil_matrix


def transition12(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] dx,np.ndarray[np.float64_t, ndim=2] U01):
    """
    Transition matrix for the covariance term of the diffusion operator, with term D_{xy} (-x*y*phi)
    As with the ADI components, final transition matrix is given by I + dt/nu*P
    """
    cdef int ii
    cdef int jj
        
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

