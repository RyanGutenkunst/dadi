#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np

# =========================================================
# SHARED AND DIPLOID C FUNCTIONS
# =========================================================
cdef extern from "integration_shared.h":
    double Vfunc(double x, double nu)
    double Mfunc1D(double x, double gamma, double h)
    void compute_dx(double *xx, int N, double *dx)
    void compute_dfactor(double *dx, int N, double *dfactor)
    void compute_xInt(double *xx, int N, double *xInt)
    void compute_delj(double *dx, double *MInt, double *VInt,
            int N, double *delj, int use_delj_trick)
    void compute_abc_nobc(double *dx, double *dfactor, 
            double *delj, double *MInt, double *V, double dt, int N,
            double *a, double *b, double *c)
            
# =========================================================
# C FUNCTIONS FOR POLYPLOIDS
# =========================================================
cdef extern from "integration_shared_poly.h":
    double Vfunc_auto(double x, double nu)
    double Mfunc1D_auto(double x, double gam1, double gam2, double gam3, double gam4)

# =========================================================
# C TRIDIAGONAL MATRIX SOLVER
# =========================================================
cdef extern from "tridiag.h":
    void tridiag(double *a, double *b, double *c, double *r, double *u, int n)

# =========================================================
# CYTHON 1D INTEGRATION FUNCTION 
# =========================================================
cdef void c_implicit_1Dx(double[:] phi, double[:] xx, double nu, double[:] sel_vec, 
                        double dt, int use_delj_trick, int[:] ploidy):

    # since we only need the sel_vec to accommodate a certain max number of params, 
    # we can put n params into the first n slots of sel_vec
    # where n is the number of params for the given ploidy
    # this saves us from creating very long arrays or dealing with variable length arrays
    # Here, sel_vec = [param1, param2, param3, param4] b/c we only consider diploids and autotetraploids

    cdef int L = xx.shape[0] # number of grid points
    cdef int ii # loop index

    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] dx = np.empty(L-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(L, dtype=np.float64) 
    cdef double[:] xInt = np.empty(L-1, dtype=np.float64)
    cdef double[:] delj = np.empty(L-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(L-1, dtype=np.float64)
    cdef double[:] V = np.empty(L, dtype=np.float64)
    cdef double[:] VInt = np.empty(L-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(L, dtype=np.float64)
    cdef double[:] b = np.empty(L, dtype=np.float64)
    cdef double[:] c = np.empty(L, dtype=np.float64)
    cdef double[:] r = np.empty(L, dtype=np.float64)
    ### weights/coefficients for scaling
    cdef int is_diploid = ploidy[0]
    cdef int is_auto = ploidy[1]

    # call the C functions for the grid spacings and other numerical details
    compute_dx(&xx[0], L, &dx[0])
    compute_dfactor(&dx[0], L, &dfactor[0])
    compute_xInt(&xx[0], L, &xInt[0])
    
    # branch on ploidy type here
    if is_diploid:
        Mfirst = Mfunc1D(xx[0], sel_vec[0], sel_vec[1]) 
        Mlast =  Mfunc1D(xx[L-1], sel_vec[0], sel_vec[1]) 

        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu)

        for ii in range(0, L-1):
            MInt[ii] = Mfunc1D(xInt[ii], sel_vec[0], sel_vec[1]) 
            VInt[ii] = Vfunc(xInt[ii], nu)

        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)

        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
        if Mfirst <= 0:
            b[0] += (0.5/nu - Mfirst)*2/dx[0] 
        if Mlast >= 0:
            b[L-1] += -(-0.5/nu - Mlast)*2/dx[L-2] 

    if is_auto:
        Mfirst = Mfunc1D_auto(xx[0], sel_vec[0], sel_vec[1], sel_vec[2], sel_vec[3])
        Mlast = Mfunc1D_auto(xx[L-1], sel_vec[0], sel_vec[1], sel_vec[2], sel_vec[3])
    
        for ii in range(0, L):
            V[ii] = Vfunc_auto(xx[ii], nu)
    
        for ii in range(0, L-1):
            MInt[ii] = Mfunc1D_auto(xInt[ii], sel_vec[0], sel_vec[1], sel_vec[2], sel_vec[3])
            VInt[ii] = Vfunc_auto(xInt[ii], nu)

        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)

        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
    
        if Mfirst <= 0:
            b[0] +=(0.25/nu - Mfirst)*2/dx[0]
        if Mlast >= 0:
            b[L-1] += -(-0.25/nu - Mlast)*2/dx[L-2]

    # calculate the RHS of the tridiagonal problem
    for ii in range(0, L):
        r[ii] = phi[ii]/dt

    # solve the tridiagonal matrix
    tridiag(&a[0], &b[0], &c[0], &r[0], &phi[0], L)

### ==========================================================================
### MAKE THE INTEGRATION FUNCTION CALLABLE FROM PYTHON
### ==========================================================================  
def implicit_1Dx(np.ndarray[double, ndim=1] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 double nu, 
                 np.ndarray[double, ndim=1] sel_vec, 
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy):
    """
    Implicit 1D integration function for 1D diffusion equation.
    This version uses branching outside the loops, so it only computes the
    diploid OR auto functions, not both.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx : numpy array (float64) 
        Grid points
    nu : float
        Population size parameter
    sel_vec : numpy array (float64)
        Selection parameters [gamma, h, gam1, gam2, gam3, gam4]
    dt : float
        Time step
    use_delj_trick : int
        Whether to use delj optimization (0 or 1)
    ploidy : numpy array (int)
        Ploidy coefficients [diploid_coeff, auto_coeff]
        Either 0 or 1
    
    Returns:
    --------
    numpy array : Modified phi array
    """
    # Call the cdef function with memory views
    c_implicit_1Dx(phi, xx, nu, sel_vec, dt, use_delj_trick, ploidy)
    return phi
