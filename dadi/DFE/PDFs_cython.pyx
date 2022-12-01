import numpy as np
cimport numpy as np

cdef extern from "PDFs.c":
    void c_biv_lognormal "biv_lognormal"(double *xx, double *yy, double *params, int n, int m,
        int Nparams, double *output)
    void c_biv_ind_gamma "biv_ind_gamma" (double *xx, double *yy, double *params, int n, int m,
        int Nparams, double *output)

def biv_lognormal(np.ndarray xx, np.ndarray yy, np.ndarray params):
    cdef np.ndarray[np.float64_t, ndim=2] zz = np.empty((xx.size, yy.size), dtype=np.float64)
    c_biv_lognormal(<double*> xx.data, <double*> yy.data, <double*> params.data, 
        xx.size, yy.size, params.size, <double*> zz.data)
    return zz

def biv_ind_gamma(np.ndarray xx, np.ndarray yy, np.ndarray params):
    cdef np.ndarray[np.float64_t, ndim=2] zz = np.empty((xx.size, yy.size), dtype=np.float64)
    c_biv_ind_gamma(<double*> xx.data, <double*> yy.data, <double*> params.data, 
        xx.size, yy.size, params.size, <double*> zz.data)
    return zz