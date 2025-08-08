#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np

# =========================================================
# SHARED AND DIPLOID C FUNCTIONS
# =========================================================
cdef extern from "integration_shared.h":
    double Vfunc(double x, double nu)
    double Mfunc2D(double x, double y, double mxy, double gamma, double h)
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
    double Mfunc2D_auto(double x, double y, double mxy, double gam1, double gam2, double gam3, double gam4)
    double Mfunc2D_allo_a(double x, double y, double mxy, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22)
    double Mfunc2D_allo_b(double x, double y, double mxy, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22)

# =========================================================
# C TRIDIAGONAL MATRIX SOLVER
# =========================================================
# for higher dimensions, we want to preallocate memory outside of the spatial loop(s)
# so, we will use this set of functions instead of the regular tridiagonal solver
cdef extern from "tridiag.h":
    void tridiag_malloc(int n)
    void tridiag_premalloc(double *a, double *b, double *c, double *r, double *u, int n)
    void tridiag_free()

# =========================================================
# CYTHON 2D INTEGRATION FUNCTIONS
# =========================================================

cdef void c_implicit_2Dx(double[:,:] phi, double[:] xx, double[:] yy, 
                        double nu1, double m12, double[:] s1, 
                        double dt, int use_delj_trick, int[:] ploidy,
                        double[:] dx, double[:] dfactor, double[:] xInt, double[:] delj, 
                        double[:] MInt, double[:] V, double[:] VInt, 
                        double[:] a, double[:] b, double[:] c, double[:] r, double[:] temp):
    
    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int ii # loop index for x
    cdef int jj # loop index for y
    cdef double y # single y value from the grid yy
    cdef double Mfirst, Mlast
    ### weights/coefficients for scaling
    cdef int is_diploid = ploidy[0]
    cdef int is_auto = ploidy[1]
    cdef int is_alloa = ploidy[2]
    cdef int is_allob = ploidy[3]

    # compute the x step size and intermediate x values
    compute_dx(&xx[0], L, &dx[0])
    compute_dfactor(&dx[0], L, &dfactor[0])
    compute_xInt(&xx[0], L, &xInt[0])
    # dynamic allocation of memory for tridiag
    tridiag_malloc(L)

    # branch on ploidy
    if is_diploid:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y values
        for jj in range(M):
            y = yy[jj]

            Mfirst = Mfunc2D(xx[0], y, m12, s1[0], s1[1])
            Mlast = Mfunc2D(xx[L-1], y, m12, s1[0], s1[1])
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D(xInt[ii], y, m12, s1[0], s1[1]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if Mfirst <= 0:
                b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
            if Mlast >= 0:
                b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

            for ii in range(0, L):
                r[ii] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj] = temp[ii]

    elif is_auto:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_auto(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_auto(xInt[ii], nu1)
        # loop through y values
        for jj in range(M):
            y = yy[jj]

            Mfirst = Mfunc2D_auto(xx[0], y, m12, s1[0], s1[1], s1[2], s1[3])
            Mlast = Mfunc2D_auto(xx[L-1], y, m12, s1[0], s1[1], s1[2], s1[3])   
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D_auto(xInt[ii], y, m12, s1[0], s1[1], s1[2], s1[3]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if Mfirst <= 0:
                b[0] += (0.25/nu1 - Mfirst)*2/dx[0] 
            if Mlast >= 0:
                b[L-1] += -(-0.25/nu1 - Mlast)*2/dx[L-2]

            for ii in range(0, L):
                r[ii] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj] = temp[ii]

    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y values
        for jj in range(M):
            y = yy[jj]

            Mfirst = Mfunc2D_allo_a(xx[0], y, m12, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7])
            Mlast = Mfunc2D_allo_a(xx[L-1], y, m12, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7])   
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D_allo_a(xInt[ii], y, m12, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if Mfirst <= 0:
                b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
            if Mlast >= 0:
                b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

            for ii in range(0, L):
                r[ii] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj] = temp[ii]
    
    elif is_allob:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y values
        for jj in range(M):
            y = yy[jj]

            Mfirst = Mfunc2D_allo_b(xx[0], y, m12, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7])
            Mlast = Mfunc2D_allo_b(xx[L-1], y, m12, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7])   
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D_allo_b(xInt[ii], y, m12, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if Mfirst <= 0:
                b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
            if Mlast >= 0:
                b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

            for ii in range(0, L):
                r[ii] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj] = temp[ii]
    
    tridiag_free()
            
cdef void c_implicit_2Dy(double[:,:] phi, double[:] xx, double[:] yy, 
                        double nu2, double m21, double[:] s2, 
                        double dt, int use_delj_trick, int[:] ploidy,
                        double[:] dy, double[:] dfactor, double[:] yInt, double[:] delj, 
                        double[:] MInt, double[:] V, double[:] VInt, 
                        double[:] a, double[:] b, double[:] c, double[:] r, double[:] temp):
    
    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int ii # loop index for x
    cdef int jj # loop index for y
    cdef double x # single x value from the grid xx
    cdef double Mfirst, Mlast
    ### weights/coefficients for scaling
    cdef int is_diploid = ploidy[0]
    cdef int is_auto = ploidy[1]
    cdef int is_alloa = ploidy[2]
    cdef int is_allob = ploidy[3]

    # compute the y step size and intermediate y values
    compute_dx(&yy[0], M, &dy[0])
    compute_dfactor(&dy[0], M, &dfactor[0])
    compute_xInt(&yy[0], M, &yInt[0])
    # dynamic allocation of memory for tridiag
    tridiag_malloc(M)

    # branch on ploidy
    if is_diploid:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x values
        for ii in range(L):
            x = xx[ii]

            Mfirst = Mfunc2D(yy[0], x, m21, s2[0], s2[1])
            Mlast = Mfunc2D(yy[M-1], x, m21, s2[0], s2[1])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D(yInt[jj], x, m21, s2[0], s2[1]) 
            compute_delj(&dy[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if Mfirst <= 0:
                b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
            if Mlast >= 0:
                b[L-1] += -(-0.5/nu2 - Mlast)*2/dy[L-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_auto(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_auto(yInt[jj], nu2)
        # loop through x values
        for ii in range(L):
            x = xx[ii]

            Mfirst = Mfunc2D_auto(yy[0], x, m21, s2[0], s2[1], s2[2], s2[3])
            Mlast = Mfunc2D_auto(yy[M-1], x, m21, s2[0], s2[1], s2[2], s2[3])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D_auto(yInt[jj], x, m21, s2[0], s2[1], s2[2], s2[3])
            compute_delj(&dy[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if Mfirst <= 0:
                b[0] += (0.25/nu2 - Mfirst)*2/dy[0] 
            if Mlast >= 0:
                b[L-1] += -(-0.25/nu2 - Mlast)*2/dy[L-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]
    
    if is_alloa:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x values
        for ii in range(L):
            x = xx[ii]

            Mfirst = Mfunc2D_allo_a(yy[0], x, m21, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5], s2[6], s2[7])
            Mlast = Mfunc2D_allo_a(yy[M-1], x, m21, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5], s2[6], s2[7])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D_allo_a(yInt[jj], x, m21, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5], s2[6], s2[7]) 
            compute_delj(&dy[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if Mfirst <= 0:
                b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
            if Mlast >= 0:
                b[L-1] += -(-0.5/nu2 - Mlast)*2/dy[L-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]
    
    elif is_allob:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x values
        for ii in range(L):
            x = xx[ii]

            Mfirst = Mfunc2D_allo_b(yy[0], x, m21, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5], s2[6], s2[7])
            Mlast = Mfunc2D_allo_b(yy[M-1], x, m21, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5], s2[6], s2[7])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D_allo_b(yInt[jj], x, m21, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5], s2[6], s2[7])
            compute_delj(&dy[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if Mfirst <= 0:
                b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
            if Mlast >= 0:
                b[L-1] += -(-0.5/nu2 - Mlast)*2/dy[L-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]
    tridiag_free()

# =========================================================
# MAKE THE INTEGRATION FUNCTIONS CALLABLE FROM PYTHON
# =========================================================

def implicit_2Dx(np.ndarray[double, ndim=1] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 double nu1, 
                 double m12,
                 np.ndarray[double, ndim=1] s1, 
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy,
                 np.ndarray[double, ndim=1] dx, 
                 np.ndarray[double, ndim=1] dfactor, 
                 np.ndarray[double, ndim=1] xInt, 
                 np.ndarray[double, ndim=1] delj, 
                 np.ndarray[double, ndim=1] MInt, 
                 np.ndarray[double, ndim=1] V, 
                 np.ndarray[double, ndim=1] VInt, 
                 np.ndarray[double, ndim=1] a, 
                 np.ndarray[double, ndim=1] b, 
                 np.ndarray[double, ndim=1] c, 
                 np.ndarray[double, ndim=1] r,
                 np.ndarray[double, ndim=1] temp):
    """
    Implicit 2D integration function for x direction of 2D diffusion equation.
    This version uses pre-allocated memory views for all intermediate arrays.
    (Hence the extra parameters at the end.)
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx : numpy array (float64) 
        Grid points
    yy : numpy array (float64) 
        Grid points
    nu1 : float
        Population size for pop1
    s1 : numpy array (float64)
        Selection params for pop1
    dt : float
        Time step
    use_delj_trick : int
        Whether to use delj optimization (0 or 1)
    ploidy : numpy array (int)
        Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob]
    dx : numpy array (float64)
        Grid spacings
    dfactor : numpy array (float64)
        Factors for scaling
    xInt : numpy array(float64)
        Integration points
    delj : numpy array (float64)
        Values of delj
    MInt : numpy array (float64)
        Values of MInt
    V : numpy array (float64)
        Values of V
    VInt : numpy array (float64)
        Values of VInt
    a : numpy array (float64)
        a, b, c, r, and temp arrays are for tridiagonal matrix solver
    b : numpy array (float64)
    c : numpy array (float64)
    r : numpy array (float64)
    temp : numpy array (float64)

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    # Call the cdef function with memory views
    c_implicit_2Dx(phi, xx, yy, nu1, m12, s1, dt, use_delj_trick, ploidy,
                    dx, dfactor, xInt, delj, MInt, V, VInt, 
                    a, b, c, r, temp)
    return phi         

def implicit_2Dy(np.ndarray[double, ndim=1] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 double nu2, 
                 double m21,
                 np.ndarray[double, ndim=1] s2, 
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy,
                 np.ndarray[double, ndim=1] dy, 
                 np.ndarray[double, ndim=1] dfactor, 
                 np.ndarray[double, ndim=1] yInt, 
                 np.ndarray[double, ndim=1] delj, 
                 np.ndarray[double, ndim=1] MInt, 
                 np.ndarray[double, ndim=1] V, 
                 np.ndarray[double, ndim=1] VInt, 
                 np.ndarray[double, ndim=1] a, 
                 np.ndarray[double, ndim=1] b, 
                 np.ndarray[double, ndim=1] c, 
                 np.ndarray[double, ndim=1] r,
                 np.ndarray[double, ndim=1] temp):
    """
    Implicit 2D integration function for y direction of 2D diffusion equation.
    This version uses pre-allocated memory views for all intermediate arrays.
    (Hence the extra parameters at the end.)
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx : numpy array (float64) 
        Grid points
    yy : numpy array (float64) 
        Grid points
    nu2 : float
        Population size for pop2
    s2 : numpy array (float64)
        Selection params for pop2
    dt : float
        Time step
    use_delj_trick : int
        Whether to use delj optimization (0 or 1)
    ploidy : numpy array (int)
        Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob]
    dy : numpy array (float64)
        Grid spacings
    dfactor : numpy array (float64)
        Factors for scaling
    yInt : numpy array(float64)
        Integration points
    delj : numpy array (float64)
        Values of delj
    MInt : numpy array (float64)
        Values of MInt
    V : numpy array (float64)
        Values of V
    VInt : numpy array (float64)
        Values of VInt
    a : numpy array (float64)
        a, b, c, r, and temp arrays are for tridiagonal matrix solver
    b : numpy array (float64)
    c : numpy array (float64)
    r : numpy array (float64)
    temp : numpy array (float64)

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    # Call the cdef function with memory views
    c_implicit_2Dy(phi, xx, yy, nu2, m21, s2, dt, use_delj_trick, ploidy,
                    dy, dfactor, yInt, delj, MInt, V, VInt, 
                    a, b, c, r, temp)
    return phi                            
