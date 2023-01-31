import numpy as np
cimport numpy as np

cdef extern from "tridiag.h":
    void c_tridiag "tridiag" (double *a, double *b, double *c, double *r, double *u,
        int n)

# Type checking arguments introduced noticable overhead in this (fast) function.
#def tridiag_cython(np.ndarray[np.float64_t, ndim=1] a not None, 
#                    np.ndarray[np.float64_t, ndim=1] b not None,
#                    np.ndarray[np.float64_t, ndim=1] c not None,
#                    np.ndarray[np.float64_t, ndim=1] r not None):

def tridiag(np.ndarray a, np.ndarray b, np.ndarray c, np.ndarray r):
    cdef np.ndarray u = np.empty(a.size, dtype=np.float64)
    c_tridiag(<double*> a.data, <double*> b.data, <double*> c.data, 
            <double*> r.data, <double*> u.data, a.size)
    return u
