import numpy as np

# =========================================================
# SHARED AND DIPLOID C FUNCTIONS
# =========================================================
cdef extern from "integration_shared.h":
    double Vfunc(double x, double nu)
    double Mfunc1D(double x, double gamma, double h)
    double Mfunc2D(double x, double y, double mxy, double gamma, double h)
    double Mfunc3D(double x, double y, double z, double mxy, double mxz, double gamma, double h)
    double Mfunc4D(double x, double y, double z, double a, 
                   double mxy, double mxz, double mxa, double gamma, double h)
    double Mfunc5D(double x, double y, double z, double a, double b,
                   double mxy, double mxz, double mxa, double mxb, double gamma, double h)
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
    double Vfunc_tetra(double x, double nu)
    double Mfunc1D_auto(double x, double gam1, double gam2, double gam3, double gam4)
    double Mfunc2D_auto(double x, double y, double mxy, double gam1, double gam2, double gam3, double gam4)
    double Mfunc3D_auto(double x, double y, double z, double mxy, double mxz, 
                        double gam1, double gam2, double gam3, double gam4)
    double Mfunc4D_auto(double x, double y, double z, double a, 
                        double mxy, double mxz, double mxa, 
                        double gam1, double gam2, double gam3, double gam4)
    double Mfunc5D_auto(double x, double y, double z, double a, double b,
                        double mxy, double mxz, double mxa, double mxb,
                        double gam1, double gam2, double gam3, double gam4)

    double Mfunc2D_allo_a(double x, double y, double exy, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22)
    double Mfunc2D_allo_b(double x, double y, double exy, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22)
    double Mfunc3D_allo_a(double x, double y, double z, double exy, double mxz, 
                          double g01, double g02, double g10, double g11, 
                          double g12, double g20, double g21, double g22)
    double Mfunc3D_allo_b(double x, double y, double z, double exy, double mxz, 
                          double g01, double g02, double g10, double g11, 
                          double g12, double g20, double g21, double g22)
    double Mfunc4D_allo_a(double x, double y, double z, double a, 
                          double exy, double mxz, double mxa, 
                          double g01, double g02, double g10, double g11, 
                          double g12, double g20, double g21, double g22)
    double Mfunc4D_allo_b(double x, double y, double z, double a, 
                          double exy, double mxz, double mxa, 
                          double g01, double g02, double g10, double g11, 
                          double g12, double g20, double g21, double g22)
    double Mfunc5D_allo_a(double x, double y, double z, double a, double b,
                          double exy, double mxz, double mxa, double mxb,
                          double g01, double g02, double g10, double g11, 
                          double g12, double g20, double g21, double g22)
    double Mfunc5D_allo_b(double x, double y, double z, double a, double b,
                          double exy, double mxz, double mxa, double mxb,
                          double g01, double g02, double g10, double g11, 
                          double g12, double g20, double g21, double g22)

    double Vfunc_hex(double x, double nu)
    double Mfunc1D_autohex(double x, double g1, double g2, double g3, double g4, double g5, double g6)
    double Mfunc2D_autohex(double x, double y, double mxy, double g1, double g2, double g3, double g4, double g5, double g6)
    double Mfunc3D_autohex(double x, double y, double z, double mxy, double mxz, 
                       double g1, double g2, double g3, double g4, double g5, double g6)
    double Mfunc4D_autohex(double x, double y, double z, double a, double mxy, double mxz, double mxa, 
                       double g1, double g2, double g3, double g4, double g5, double g6)
    double Mfunc5D_autohex(double x, double y, double z, double a, double b,
                       double mxy, double mxz, double mxa, double mxb,  
                       double g1, double g2, double g3, double g4, double g5, double g6)

    # the naming convention for these functions specifies the overall ploidy (hex) 
    # and then the subgenome ploidy (tetra or dip)
    # so, MfuncND_hex_tetra is the mean func. for the tetraploid subgenome of a 4+2 hexaploid
    double Mfunc2D_hex_tetra(double x, double y, double exy, double g01, double g02, 
                             double g10, double g11, double g12, double g20, double g21, double g22, 
                             double g30, double g31, double g32, double g40, double g41, double g42)
    double Mfunc2D_hex_dip(double x, double y, double exy, double g01, double g02, 
                           double g10, double g11, double g12, double g20, double g21, double g22, 
                           double g30, double g31, double g32, double g40, double g41, double g42)
    double Mfunc3D_hex_tetra(double x, double y, double z, double exy, double mxz, 
                            double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                            double g30, double g31, double g32, double g40, double g41, double g42)
    double Mfunc3D_hex_dip(double x, double y, double z, double exy, double mxz, 
                           double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22,                    
                           double g30, double g31, double g32, double g40, double g41, double g42)
    double Mfunc4D_hex_tetra(double x, double y, double z, double a, double exy, double mxz, double mxa, 
                            double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                            double g30, double g31, double g32, double g40, double g41, double g42)
    double Mfunc4D_hex_dip(double x, double y, double z, double a, double exy, double mxz, double mxa, 
                           double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                           double g30, double g31, double g32, double g40, double g41, double g42)
    double Mfunc5D_hex_tetra(double x, double y, double z, double a, double b, double exy, double mxz, double mxa, double mxb, 
                             double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22,
                             double g30, double g31, double g32, double g40, double g41, double g42)
    double Mfunc5D_hex_dip(double x, double y, double z, double a, double b, double exy, double mxz, double mxa, double mxb, 
                           double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22,
                           double g30, double g31, double g32, double g40, double g41, double g42)

    double Mfunc3D_hex_a(double x, double y, double z, double exy, double exz, 
                         double g001, double g002, double g010, double g011, double g012, double g020, double g021, double g022,
                         double g100, double g101, double g102, double g110, double g111, double g112, double g120, double g121, double g122,
                         double g200, double g201, double g202, double g210, double g211, double g212, double g220, double g221, double g222)
    double Mfunc3D_hex_b(double x, double y, double z, double exy, double exz, 
                         double g001, double g002, double g010, double g011, double g012, double g020, double g021, double g022,
                         double g100, double g101, double g102, double g110, double g111, double g112, double g120, double g121, double g122,
                         double g200, double g201, double g202, double g210, double g211, double g212, double g220, double g221, double g222)
    double Mfunc3D_hex_c(double x, double y, double z, double exy, double exz, 
                         double g001, double g002, double g010, double g011, double g012, double g020, double g021, double g022,
                         double g100, double g101, double g102, double g110, double g111, double g112, double g120, double g121, double g122,
                         double g200, double g201, double g202, double g210, double g211, double g212, double g220, double g221, double g222)
    
    double Mfunc4D_hex_a(double x, double y, double z, double a, double exy, double exz, double mxa,
                         double g001, double g002, double g010, double g011, double g012, double g020, double g021, double g022,
                         double g100, double g101, double g102, double g110, double g111, double g112, double g120, double g121, double g122,
                         double g200, double g201, double g202, double g210, double g211, double g212, double g220, double g221, double g222)
    double Mfunc4D_hex_b(double x, double y, double z, double a, double exy, double exz, double mxa,
                         double g001, double g002, double g010, double g011, double g012, double g020, double g021, double g022,
                         double g100, double g101, double g102, double g110, double g111, double g112, double g120, double g121, double g122,
                         double g200, double g201, double g202, double g210, double g211, double g212, double g220, double g221, double g222)
    double Mfunc4D_hex_c(double x, double y, double z, double a, double exy, double exz, double mxa,
                         double g001, double g002, double g010, double g011, double g012, double g020, double g021, double g022,
                         double g100, double g101, double g102, double g110, double g111, double g112, double g120, double g121, double g122,
                         double g200, double g201, double g202, double g210, double g211, double g212, double g220, double g221, double g222)
    
    double Mfunc5D_hex_a(double x, double y, double z, double a, double b, double exy, double exz, double mxa, double mxb,
                         double g001, double g002, double g010, double g011, double g012, double g020, double g021, double g022,
                         double g100, double g101, double g102, double g110, double g111, double g112, double g120, double g121, double g122,
                         double g200, double g201, double g202, double g210, double g211, double g212, double g220, double g221, double g222)
    double Mfunc5D_hex_b(double x, double y, double z, double a, double b, double exy, double exz, double mxa, double mxb,
                         double g001, double g002, double g010, double g011, double g012, double g020, double g021, double g022,
                         double g100, double g101, double g102, double g110, double g111, double g112, double g120, double g121, double g122,
                         double g200, double g201, double g202, double g210, double g211, double g212, double g220, double g221, double g222)
    double Mfunc5D_hex_c(double x, double y, double z, double a, double b, double exy, double exz, double mxa, double mxb,  
                         double g001, double g002, double g010, double g011, double g012, double g020, double g021, double g022,
                         double g100, double g101, double g102, double g110, double g111, double g112, double g120, double g121, double g122,
                         double g200, double g201, double g202, double g210, double g211, double g212, double g220, double g221, double g222)
    
# =========================================================
# C TRIDIAGONAL MATRIX SOLVER
# =========================================================
cdef extern from "tridiag.h":
    void tridiag(double *a, double *b, double *c, double *r, double *u, int n)
    void tridiag_malloc(int n)
    void tridiag_premalloc(double *a, double *b, double *c, double *r, double *u, int n)
    void tridiag_free()

# =========================================================
# CYTHON 1D INTEGRATION FUNCTION 
# =========================================================
def implicit_1Dx(double[:] phi, double[:] xx, double nu, double[:] s, 
                        double dt, int use_delj_trick, int[:] ploidy):
    """
    Implicit 1D integration function for 1D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx: numpy array (float64)
        discrete numerical grid for spatial dimension
    nu: Population size
    s: vector of selection parameters
    dt: Time step
    use_delj_trick: Whether to use delj optimization (0 or 1)
    ploidy: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]
    
    Returns:
    --------
    phi: Modified phi array
    """
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
    ### specify ploidy of the x direction
    cdef double is_diploid = ploidy[0]
    cdef double is_auto = ploidy[1]
    cdef double is_autohex = ploidy[4]

    # compute step size and intermediate values
    compute_dx(&xx[0], L, &dx[0])
    compute_dfactor(&dx[0], L, &dfactor[0])
    compute_xInt(&xx[0], L, &xInt[0])

    # branch on ploidy type here
    if is_diploid:
        Mfirst = Mfunc1D(xx[0], s[0], s[1]) 
        Mlast =  Mfunc1D(xx[L-1], s[0], s[1]) 

        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu)

        for ii in range(0, L-1):
            MInt[ii] = Mfunc1D(xInt[ii], s[0], s[1]) 
            VInt[ii] = Vfunc(xInt[ii], nu)

        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
        
        if Mfirst <= 0:
            b[0] += (0.5/nu - Mfirst)*2/dx[0] 
        if Mlast >= 0:
            b[L-1] += -(-0.5/nu - Mlast)*2/dx[L-2]

    elif is_auto:
        Mfirst = Mfunc1D_auto(xx[0], s[0], s[1], s[2], s[3])
        Mlast = Mfunc1D_auto(xx[L-1], s[0], s[1], s[2], s[3])
    
        for ii in range(0, L):
            V[ii] = Vfunc_tetra(xx[ii], nu)
    
        for ii in range(0, L-1):
            MInt[ii] = Mfunc1D_auto(xInt[ii], s[0], s[1], s[2], s[3])
            VInt[ii] = Vfunc_tetra(xInt[ii], nu)

        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)

        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
    
        if Mfirst <= 0:
            b[0] +=(0.25/nu - Mfirst)*2/dx[0]
        if Mlast >= 0:
            b[L-1] += -(-0.25/nu - Mlast)*2/dx[L-2]

    elif is_autohex:
        Mfirst = Mfunc1D_autohex(xx[0], s[0], s[1], s[2], s[3], s[4], s[5])
        Mlast = Mfunc1D_autohex(xx[L-1], s[0], s[1], s[2], s[3], s[4], s[5])
    
        for ii in range(0, L):
            V[ii] = Vfunc_hex(xx[ii], nu)
    
        for ii in range(0, L-1):
            MInt[ii] = Mfunc1D_autohex(xInt[ii], s[0], s[1], s[2], s[3], s[4], s[5])
            VInt[ii] = Vfunc_hex(xInt[ii], nu)

        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)

        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
    
        if Mfirst <= 0:
            b[0] +=(1/(6*nu) - Mfirst)*2/dx[0]
        if Mlast >= 0:
            b[L-1] += -(-1/(6*nu) - Mlast)*2/dx[L-2]

    # calculate the RHS of the tridiagonal problem
    for ii in range(0, L):
        r[ii] = phi[ii]/dt

    # solve the tridiagonal matrix
    tridiag(&a[0], &b[0], &c[0], &r[0], &phi[0], L)

    return np.asarray(phi)

# =========================================================
# CYTHON 2D INTEGRATION FUNCTIONS - TEMPORAL PARAMS
# =========================================================
def implicit_2Dx(double[:,:] phi, double[:] xx, double[:] yy, 
                        double nu1, double m12, double[:] s1, 
                        double dt, int use_delj_trick, int[:] ploidy1):
    """
    Implicit 2D integration function for x direction of 2D diffusion equation.
    
    Parameters:
    -----------
    phi: numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu1: Population size for pop1
    m12: Migration rate to pop1 from pop2
    s1: vector of selection parameters for pop1
    dt: Time step
    use_delj_trick: Whether to use delj optimization (0 or 1)
    ploidy1: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """

    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x dim
    cdef int M = yy.shape[0] # number of grid points in y dim
    cdef int ii # loop index for x dim
    cdef int jj # loop index for y dim
    cdef double y # single y value from the grid yy
    
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
    cdef double[:] temp = np.empty(L, dtype=np.float64)
    ### specify ploidy of the x direction
    cdef int is_diploid = ploidy1[0]
    cdef int is_auto = ploidy1[1]
    cdef int is_alloa = ploidy1[2]
    cdef int is_allob = ploidy1[3]
    cdef int is_autohex = ploidy1[4]
    cdef int is_hex_tetra = ploidy1[5]
    cdef int is_hex_dip = ploidy1[6]

    # compute step size and intermediate values
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

            Mfirst = Mfunc2D(xx[0], y, m12, s1[0],s1[1])
            Mlast = Mfunc2D(xx[L-1], y, m12, s1[0],s1[1])
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D(xInt[ii], y, m12, s1[0],s1[1]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if y==0 and Mfirst <= 0:
                b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
            if y==1 and Mlast >= 0:
                b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

            for ii in range(0, L):
                r[ii] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj] = temp[ii]

    elif is_auto:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_tetra(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_tetra(xInt[ii], nu1)
        # loop through y values
        for jj in range(M):
            y = yy[jj]

            Mfirst = Mfunc2D_auto(xx[0], y, m12, s1[0],s1[1],s1[2],s1[3])
            Mlast = Mfunc2D_auto(xx[L-1], y, m12, s1[0],s1[1],s1[2],s1[3])   
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D_auto(xInt[ii], y, m12, s1[0],s1[1],s1[2],s1[3]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if y==0 and Mfirst <= 0:
                b[0] += (0.25/nu1 - Mfirst)*2/dx[0] 
            if y==1 and Mlast >= 0:
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

            Mfirst = Mfunc2D_allo_a(xx[0], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
            Mlast = Mfunc2D_allo_a(xx[L-1], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])   
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D_allo_a(xInt[ii], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if y==0 and Mfirst <= 0:
                b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
            if y==1 and Mlast >= 0:
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

            Mfirst = Mfunc2D_allo_b(xx[0], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
            Mlast = Mfunc2D_allo_b(xx[L-1], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])   
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D_allo_b(xInt[ii], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if y==0 and Mfirst <= 0:
                b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
            if y==1 and Mlast >= 0:
                b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

            for ii in range(0, L):
                r[ii] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj] = temp[ii]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_hex(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_hex(xInt[ii], nu1)
        # loop through y values
        for jj in range(M):
            y = yy[jj]

            Mfirst = Mfunc2D_autohex(xx[0], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
            Mlast = Mfunc2D_autohex(xx[L-1], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D_autohex(xInt[ii], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if y==0 and Mfirst <= 0:
                b[0] += (1/(6*nu1) - Mfirst)*2/dx[0] 
            if y==1 and Mlast >= 0:
                b[L-1] += -(-1/(6*nu1) - Mlast)*2/dx[L-2]

            for ii in range(0, L):
                r[ii] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj] = temp[ii]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_tetra(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_tetra(xInt[ii], nu1)
        # loop through y values
        for jj in range(M):
            y = yy[jj]

            Mfirst = Mfunc2D_hex_tetra(xx[0], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
            Mlast = Mfunc2D_hex_tetra(xx[L-1], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])   
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D_hex_tetra(xInt[ii], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if y==0 and Mfirst <= 0:
                b[0] += (0.25/nu1 - Mfirst)*2/dx[0] 
            if y==1 and Mlast >= 0:
                b[L-1] += -(-0.25/nu1 - Mlast)*2/dx[L-2]

            for ii in range(0, L):
                r[ii] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj] = temp[ii]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y values
        for jj in range(M):
            y = yy[jj]

            Mfirst = Mfunc2D_hex_dip(xx[0], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
            Mlast = Mfunc2D_hex_dip(xx[L-1], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])   
            for ii in range(0, L-1):
                MInt[ii] = Mfunc2D_hex_dip(xInt[ii], y, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13]) 
            compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
            compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
            if y==0 and Mfirst <= 0:
                b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
            if y==1 and Mlast >= 0:
                b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

            for ii in range(0, L):
                r[ii] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj] = temp[ii]
    
    tridiag_free()

    return np.asarray(phi)
            
def implicit_2Dy(double[:,:] phi, double[:] xx, double[:] yy, 
                        double nu2, double m21, double[:] s2, 
                        double dt, int use_delj_trick, int[:] ploidy2):
    """
    Implicit 2D integration function for y direction of 2D diffusion equation.
    
    Parameters:
    -----------
    phi: numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu2: Population size for pop2
    m21: Migration rate to pop2 from pop1
    s2: vector of selection parameters for pop2
    dt: Time step
    use_delj_trick: Whether to use delj optimization (0 or 1)
    ploidy2: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi: modified phi after integration in y direction
    """

    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int ii # loop index for x
    cdef int jj # loop index for y
    cdef double x # single x value from the grid xx
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] dy = np.empty(M-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(M, dtype=np.float64) 
    cdef double[:] yInt = np.empty(M-1, dtype=np.float64)
    cdef double[:] delj = np.empty(M-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(M-1, dtype=np.float64)
    cdef double[:] V = np.empty(M, dtype=np.float64)
    cdef double[:] VInt = np.empty(M-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(M, dtype=np.float64)
    cdef double[:] b = np.empty(M, dtype=np.float64)
    cdef double[:] c = np.empty(M, dtype=np.float64)
    cdef double[:] r = np.empty(M, dtype=np.float64)
    cdef double[:] temp = np.empty(M, dtype=np.float64)
    ### specify ploidy of the y direction
    cdef int is_diploid = ploidy2[0]
    cdef int is_auto = ploidy2[1]
    cdef int is_alloa = ploidy2[2]
    cdef int is_allob = ploidy2[3]
    cdef int is_autohex = ploidy2[4]
    cdef int is_hex_tetra = ploidy2[5]
    cdef int is_hex_dip = ploidy2[6]    

    # compute step size and intermediate values
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

            Mfirst = Mfunc2D(yy[0], x, m21, s2[0],s2[1])
            Mlast = Mfunc2D(yy[M-1], x, m21, s2[0],s2[1])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D(yInt[jj], x, m21, s2[0],s2[1]) 
            compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
            if x==0 and Mfirst <= 0:
                b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
            if x==1 and Mlast >= 0:
                b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_tetra(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_tetra(yInt[jj], nu2)
        # loop through x values
        for ii in range(L):
            x = xx[ii]

            Mfirst = Mfunc2D_auto(yy[0], x, m21, s2[0],s2[1],s2[2],s2[3])
            Mlast = Mfunc2D_auto(yy[M-1], x, m21, s2[0],s2[1],s2[2],s2[3])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D_auto(yInt[jj], x, m21, s2[0],s2[1],s2[2],s2[3])
            compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
            if x==0 and Mfirst <= 0:
                b[0] += (0.25/nu2 - Mfirst)*2/dy[0] 
            if x==1 and Mlast >= 0:
                b[M-1] += -(-0.25/nu2 - Mlast)*2/dy[M-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]
    
    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x values
        for ii in range(L):
            x = xx[ii]

            Mfirst = Mfunc2D_allo_a(yy[0], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
            Mlast = Mfunc2D_allo_a(yy[M-1], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D_allo_a(yInt[jj], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7]) 
            compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
            if x==0 and Mfirst <= 0:
                b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
            if x==1 and Mlast >= 0:
                b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
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

            Mfirst = Mfunc2D_allo_b(yy[0], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
            Mlast = Mfunc2D_allo_b(yy[M-1], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D_allo_b(yInt[jj], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
            compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
            if x==0 and Mfirst <= 0:
                b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
            if x==1 and Mlast >= 0:
                b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_hex(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_hex(yInt[jj], nu2)
        # loop through x values
        for ii in range(L):
            x = xx[ii]

            Mfirst = Mfunc2D_autohex(yy[0], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
            Mlast = Mfunc2D_autohex(yy[M-1], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D_autohex(yInt[jj], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
            compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
            if x==0 and Mfirst <= 0:
                b[0] += (1/(6*nu2) - Mfirst)*2/dy[0] 
            if x==1 and Mlast >= 0:
                b[M-1] += -(-1/(6*nu2) - Mlast)*2/dy[M-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_tetra(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_tetra(yInt[jj], nu2)
        # loop through x values
        for ii in range(L):
            x = xx[ii]

            Mfirst = Mfunc2D_hex_tetra(yy[0], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
            Mlast = Mfunc2D_hex_tetra(yy[M-1], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D_hex_tetra(yInt[jj], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
            compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
            if x==0 and Mfirst <= 0:
                b[0] += (0.25/nu2 - Mfirst)*2/dy[0] 
            if x==1 and Mlast >= 0:
                b[M-1] += -(-0.25/nu2 - Mlast)*2/dy[M-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x values
        for ii in range(L):
            x = xx[ii]

            Mfirst = Mfunc2D_hex_dip(yy[0], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
            Mlast = Mfunc2D_hex_dip(yy[M-1], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
            for jj in range(0, M-1):
                MInt[jj] = Mfunc2D_hex_dip(yInt[jj], x, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
            compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
            compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
            if x==0 and Mfirst <= 0:
                b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
            if x==1 and Mlast >= 0:
                b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

            for jj in range(0, M):
                r[jj] = phi[ii, jj]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
            for jj in range(0, M):
                phi[ii, jj] = temp[jj]

    tridiag_free()

    return np.asarray(phi)

# =========================================================
# CYTHON 2D INTEGRATION FUNCTIONS - CONSTANT PARAMS    
# =========================================================
def implicit_precalc_2Dx(double[:,:] phi, double[:,:] ax, double[:,:] bx,
                                 double[:,:] cx, double dt):
    """
    Implicit 2D integration function for x direction of 2D diffusion equation.
    Uses arrays pre-computed in Python for a, b, c.

    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    ax : numpy array (float64)
        a, b, c, arrays are for tridiagonal matrix solver
    bx : numpy array (float64)
    cx : numpy array (float64)
    dt : float
        Time step

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    
    cdef int ii, jj
    cdef int L = phi.shape[0]
    cdef int M = phi.shape[1]

    # create memory views for the tridiagonal solver
    cdef double[:] a = np.empty(L, dtype=np.float64)
    cdef double[:] b = np.empty(L, dtype=np.float64)
    cdef double[:] c = np.empty(L, dtype=np.float64)
    cdef double[:] r = np.empty(L, dtype=np.float64)
    cdef double[:] temp = np.empty(L, dtype=np.float64)

    tridiag_malloc(L)

    for jj in range(0, M):
        for ii in range(0, L):
            a[ii] = ax[ii, jj]
            b[ii] = bx[ii, jj] + 1/dt
            c[ii] = cx[ii, jj]
            r[ii] = phi[ii, jj]/dt
        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
        for ii in range(0, L):
            phi[ii, jj] = temp[ii]

    tridiag_free()

    return np.asarray(phi, dtype = np.float64)

def implicit_precalc_2Dy(double[:,:] phi, double[:,:] ay, double[:,:] by,
                                 double[:,:] cy, double dt):
    """
    Implicit 2D integration function for y direction of 2D diffusion equation.
    Uses arrays pre-computed in Python for a, b, c.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    ay : numpy array (float64)
        a, b, c, arrays are for tridiagonal matrix solver
    by : numpy array (float64)
    cy : numpy array (float64)
    dt : float
        Time step

    Returns:
    --------
    phi : modified phi after integration in y direction
    """

    cdef int ii, jj
    cdef int L = phi.shape[0]
    cdef int M = phi.shape[1]

    # create memory views for the tridiagonal solver
    cdef double[:] a = np.empty(M, dtype=np.float64)
    cdef double[:] b = np.empty(M, dtype=np.float64)
    cdef double[:] c = np.empty(M, dtype=np.float64)
    cdef double[:] r = np.empty(M, dtype=np.float64)
    cdef double[:] temp = np.empty(M, dtype=np.float64)

    tridiag_malloc(M)

    for ii in range(0, L):
        for jj in range(0, M):
            a[jj] = ay[ii, jj]
            b[jj] = by[ii, jj] + 1/dt
            c[jj] = cy[ii, jj]
            r[jj] = phi[ii, jj]/dt
        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
        for jj in range(0, M):
            phi[ii, jj] = temp[jj] 
    tridiag_free()

    return np.asarray(phi, dtype=np.float64)

# =========================================================
# CYTHON 3D INTEGRATION FUNCTIONS - TEMPORAL PARAMS
# =========================================================
def implicit_3Dx(double[:,:,:] phi, double[:] xx, double[:] yy, double[:] zz,
                        double nu1, double m12, double m13, double[:] s1, 
                        double dt, int use_delj_trick, int[:] ploidy1):
    """
    Implicit 3D integration function for x direction of 3D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu1: Population size for pop1
    m12: Migration rate to pop1 from pop2
    m13: Migration rate to pop1 from pop3
    s1: vector of selection parameters for pop1
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy1: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """

    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x dim
    cdef int M = yy.shape[0] # number of grid points in y dim
    cdef int N = zz.shape[0] # number of grid points in z dim
    cdef int ii, jj, kk # loop indices
    cdef double y, z # single values from y and z grids
    
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
    cdef double[:] temp = np.empty(L, dtype=np.float64)
    ### specify ploidy of the x direction
    cdef int is_diploid = ploidy1[0]
    cdef int is_auto = ploidy1[1]
    cdef int is_autohex = ploidy1[4]
    cdef int is_hex_a = ploidy1[7]
    # note: we don't support alloa, allob, hex_tetra, or hex_dip as being the first dimension of a 3D model
    # we also only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order, 
    # hence no hex_b or hex_c models here

    # compute step size and intermediate values
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
        # loop through y and z dimensions
        for jj in range(M):
            for kk in range(N):
                y = yy[jj]
                z = zz[kk]

                Mfirst = Mfunc3D(xx[0], y,z, m12,m13, s1[0],s1[1])
                Mlast = Mfunc3D(xx[L-1], y,z, m12,m13, s1[0],s1[1])
                for ii in range(0, L-1):
                    MInt[ii] = Mfunc3D(xInt[ii], y,z, m12,m13, s1[0],s1[1]) 
                compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                if y==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                if y==1 and z==1 and Mlast >= 0:
                    b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                for ii in range(0, L):
                    r[ii] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                for ii in range(0, L):
                    phi[ii, jj, kk] = temp[ii]

    elif is_auto:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_tetra(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_tetra(xInt[ii], nu1)
        # loop through y and z dimensions
        for jj in range(M):
            for kk in range(N):
                y = yy[jj]
                z = zz[kk]

                Mfirst = Mfunc3D_auto(xx[0], y,z, m12,m13, s1[0],s1[1],s1[2],s1[3])
                Mlast = Mfunc3D_auto(xx[L-1], y,z, m12,m13, s1[0],s1[1],s1[2],s1[3])
                for ii in range(0, L-1):
                    MInt[ii] = Mfunc3D_auto(xInt[ii], y,z, m12,m13, s1[0],s1[1],s1[2],s1[3]) 
                compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                if y==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.25/nu1 - Mfirst)*2/dx[0] 
                if y==1 and z==1 and Mlast >= 0:
                    b[L-1] += -(-0.25/nu1 - Mlast)*2/dx[L-2]

                for ii in range(0, L):
                    r[ii] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                for ii in range(0, L):
                    phi[ii, jj, kk] = temp[ii]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_hex(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_hex(xInt[ii], nu1)
        # loop through y and z dimensions
        for jj in range(M):
            for kk in range(N):
                y = yy[jj]
                z = zz[kk]

                Mfirst = Mfunc3D_autohex(xx[0], y,z, m12,m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
                Mlast = Mfunc3D_autohex(xx[L-1], y,z, m12,m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
                for ii in range(0, L-1):
                    MInt[ii] = Mfunc3D_autohex(xInt[ii], y,z, m12,m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
                compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                if y==0 and z==0 and Mfirst <= 0:
                    b[0] += (1/(6*nu1) - Mfirst)*2/dx[0] 
                if y==1 and z==1 and Mlast >= 0:
                    b[L-1] += -(-1/(6*nu1) - Mlast)*2/dx[L-2]

                for ii in range(0, L):
                    r[ii] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                for ii in range(0, L):
                    phi[ii, jj, kk] = temp[ii]

    elif is_hex_a:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y and z dimensions
        for jj in range(M):
            for kk in range(N):
                y = yy[jj]
                z = zz[kk]

                Mfirst = Mfunc3D_hex_a(xx[0], y,z, m12,m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],
                                       s1[12],s1[13],s1[14],s1[15],s1[16],s1[17],s1[18],s1[19],s1[20],s1[21],s1[22],s1[23],s1[24],s1[25])
                Mlast = Mfunc3D_hex_a(xx[L-1], y,z, m12,m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],
                                       s1[12],s1[13],s1[14],s1[15],s1[16],s1[17],s1[18],s1[19],s1[20],s1[21],s1[22],s1[23],s1[24],s1[25])
                for ii in range(0, L-1):
                    MInt[ii] = Mfunc3D_hex_a(xInt[ii], y,z, m12,m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],
                                             s1[12],s1[13],s1[14],s1[15],s1[16],s1[17],s1[18],s1[19],s1[20],s1[21],s1[22],s1[23],s1[24],s1[25])
                compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                if y==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                if y==1 and z==1 and Mlast >= 0:
                    b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                for ii in range(0, L):
                    r[ii] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                for ii in range(0, L):
                    phi[ii, jj, kk] = temp[ii]
               
    tridiag_free()

    return np.asarray(phi)
            
def implicit_3Dy(double[:,:,:] phi, double[:] xx, double[:] yy, double[:] zz,
                        double nu2, double m21, double m23, double[:] s2, 
                        double dt, int use_delj_trick, int[:] ploidy2):
    """
    Implicit 3D integration function for y direction of 3D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu2: Population size for pop2
    m21: Migration rate to pop2 from pop1
    m23: Migration rate to pop2 from pop3
    s2: vector of selection parameters for pop2
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy2: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int N = zz.shape[0] # number of grid points in z direction
    cdef int ii, jj, kk # loop indices
    cdef double x, z # single values from x and z grids
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] dy = np.empty(M-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(M, dtype=np.float64) 
    cdef double[:] yInt = np.empty(M-1, dtype=np.float64)
    cdef double[:] delj = np.empty(M-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(M-1, dtype=np.float64)
    cdef double[:] V = np.empty(M, dtype=np.float64)
    cdef double[:] VInt = np.empty(M-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(M, dtype=np.float64)
    cdef double[:] b = np.empty(M, dtype=np.float64)
    cdef double[:] c = np.empty(M, dtype=np.float64)
    cdef double[:] r = np.empty(M, dtype=np.float64)
    cdef double[:] temp = np.empty(M, dtype=np.float64)
    ### specify ploidy of the y direction
    cdef int is_diploid = ploidy2[0]
    cdef int is_auto = ploidy2[1]
    cdef int is_alloa = ploidy2[2]
    cdef int is_allob = ploidy2[3]
    cdef int is_autohex = ploidy2[4]
    cdef int is_hex_tetra = ploidy2[5]
    cdef int is_hex_dip = ploidy2[6]
    cdef int is_hex_b = ploidy2[8]

    # compute step size and intermediate values
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
        # loop through x and z dimensions
        for ii in range(L):
            for kk in range(N):
                x = xx[ii]
                z = zz[kk]
                
                Mfirst = Mfunc3D(yy[0], x,z, m21,m23, s2[0],s2[1])
                Mlast = Mfunc3D(yy[M-1], x,z, m21,m23, s2[0],s2[1])  
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D(yInt[jj], x,z, m21,m23, s2[0],s2[1])
                compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                if x==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                if x==1 and z==1 and Mlast >= 0:
                    b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                for jj in range(0, M):
                    r[jj] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                for jj in range(0, M):
                    phi[ii, jj, kk] = temp[jj]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_tetra(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_tetra(yInt[jj], nu2)
        # loop through x and z values
        for ii in range(L):
            for kk in range(N):
                x = xx[ii]
                z = zz[kk]
                
                Mfirst = Mfunc3D_auto(yy[0], x,z, m21,m23, s2[0],s2[1],s2[2],s2[3])
                Mlast = Mfunc3D_auto(yy[M-1], x,z, m21,m23, s2[0],s2[1],s2[2],s2[3])
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_auto(yInt[jj], x,z, m21,m23, s2[0],s2[1],s2[2],s2[3])
                compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                if x==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.25/nu2 - Mfirst)*2/dy[0] 
                if x==1 and z==1 and Mlast >= 0:
                    b[M-1] += -(-0.25/nu2 - Mlast)*2/dy[M-2]

                for jj in range(0, M):
                    r[jj] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                for jj in range(0, M):
                    phi[ii, jj, kk] = temp[jj]
    
    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x and z values
        for ii in range(L):
            for kk in range(N):
                x = xx[ii]
                z = zz[kk]
                ### Note: the order of migration params and grids being passed here is different 
                # This is for consistency with the allo cases where the first two dimensions passed
                # to Mfunc need to be the allo subgenomes and the subgenomes are always passed as y and z.
                Mfirst = Mfunc3D_allo_a(yy[0], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                Mlast = Mfunc3D_allo_a(yy[M-1], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_allo_a(yInt[jj], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                if x==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                if x==1 and z==1 and Mlast >= 0:
                    b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                for jj in range(0, M):
                    r[jj] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                for jj in range(0, M):
                    phi[ii, jj, kk] = temp[jj]
    
    elif is_allob:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x and z values
        for ii in range(L):
            for kk in range(N):
                x = xx[ii]
                z = zz[kk]
                # see note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_allo_b(yy[0], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                Mlast = Mfunc3D_allo_b(yy[M-1], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_allo_b(yInt[jj], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                if x==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                if x==1 and z==1 and Mlast >= 0:
                    b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                for jj in range(0, M):
                    r[jj] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                for jj in range(0, M):
                    phi[ii, jj, kk] = temp[jj]
    
    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_hex(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_hex(yInt[jj], nu2)
        # loop through x and z values
        for ii in range(L):
            for kk in range(N):
                x = xx[ii]
                z = zz[kk]
                
                Mfirst = Mfunc3D_autohex(yy[0], x,z, m21,m23, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
                Mlast = Mfunc3D_autohex(yy[M-1], x,z, m21,m23, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_autohex(yInt[jj], x,z, m21,m23, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
                compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                if x==0 and z==0 and Mfirst <= 0:
                    b[0] += (1/(6*nu2) - Mfirst)*2/dy[0] 
                if x==1 and z==1 and Mlast >= 0:
                    b[M-1] += -(-1/(6*nu2) - Mlast)*2/dy[M-2]

                for jj in range(0, M):
                    r[jj] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                for jj in range(0, M):
                    phi[ii, jj, kk] = temp[jj]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_tetra(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_tetra(yInt[jj], nu2)
        # loop through x and z values
        for ii in range(L):
            for kk in range(N):
                x = xx[ii]
                z = zz[kk]
                # see note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_hex_tetra(yy[0], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                Mlast = Mfunc3D_hex_tetra(yy[M-1], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_hex_tetra(yInt[jj], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                if x==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.25/nu2 - Mfirst)*2/dy[0] 
                if x==1 and z==1 and Mlast >= 0:
                    b[M-1] += -(-0.25/nu2 - Mlast)*2/dy[M-2]

                for jj in range(0, M):
                    r[jj] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                for jj in range(0, M):
                    phi[ii, jj, kk] = temp[jj]
    
    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x and z values
        for ii in range(L):
            for kk in range(N):
                x = xx[ii]
                z = zz[kk]
                # see note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_hex_dip(yy[0], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                Mlast = Mfunc3D_hex_dip(yy[M-1], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_hex_dip(yInt[jj], z,x, m23,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                if x==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                if x==1 and z==1 and Mlast >= 0:
                    b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                for jj in range(0, M):
                    r[jj] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                for jj in range(0, M):
                    phi[ii, jj, kk] = temp[jj]

    elif is_hex_b:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x and z dimensions
        for ii in range(L):
            for kk in range(N):
                x = xx[ii]
                z = zz[kk]
                
                Mfirst = Mfunc3D_hex_b(yy[0], x,z, m21,m23, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                       s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25])
                Mlast = Mfunc3D_hex_b(yy[M-1], x,z, m21,m23, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                       s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25]) 
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_hex_b(yInt[jj], x,z, m21,m23, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                             s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25])
                compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                if x==0 and z==0 and Mfirst <= 0:
                    b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                if x==1 and z==1 and Mlast >= 0:
                    b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                for jj in range(0, M):
                    r[jj] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                for jj in range(0, M):
                    phi[ii, jj, kk] = temp[jj]

    tridiag_free()

    return np.asarray(phi)

def implicit_3Dz(double[:,:,:] phi, double[:] xx, double[:] yy, double[:] zz,
                        double nu3, double m31, double m32, double[:] s3, 
                        double dt, int use_delj_trick, int[:] ploidy3):
    """
    Implicit 3D integration function for z direction of 3D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu3: Population size for pop3
    m31: Migration rate to pop3 from pop1
    m32: Migration rate to pop3 from pop2
    s3: vector of selection parameters for pop3
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy3: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """

    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int N = zz.shape[0] # number of grid points in z direction
    cdef int ii, jj, kk # loop indices
    cdef double x, y # single values from x and y grids
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] dz = np.empty(N-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(N, dtype=np.float64) 
    cdef double[:] zInt = np.empty(N-1, dtype=np.float64)
    cdef double[:] delj = np.empty(N-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(N-1, dtype=np.float64)
    cdef double[:] V = np.empty(N, dtype=np.float64)
    cdef double[:] VInt = np.empty(N-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(N, dtype=np.float64)
    cdef double[:] b = np.empty(N, dtype=np.float64)
    cdef double[:] c = np.empty(N, dtype=np.float64)
    cdef double[:] r = np.empty(N, dtype=np.float64)
    cdef double[:] temp = np.empty(N, dtype=np.float64)
    ### specify ploidy of the z direction
    cdef int is_diploid = ploidy3[0]
    cdef int is_auto = ploidy3[1]
    cdef int is_alloa = ploidy3[2]
    cdef int is_allob = ploidy3[3]
    cdef int is_autohex = ploidy3[4]
    cdef int is_hex_tetra = ploidy3[5]
    cdef int is_hex_dip = ploidy3[6]
    cdef int is_hex_c = ploidy3[9]

    # compute step size and intermediate values
    compute_dx(&zz[0], N, &dz[0])
    compute_dfactor(&dz[0], N, &dfactor[0])
    compute_xInt(&zz[0], N, &zInt[0])
    # dynamic allocation of memory for tridiag
    tridiag_malloc(N)

    # branch on ploidy
    if is_diploid:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x and y dimensions
        for ii in range(L):
            for jj in range(M):
                x = xx[ii]
                y = yy[jj]
                
                Mfirst = Mfunc3D(zz[0], x,y, m31,m32, s3[0],s3[1])
                Mlast = Mfunc3D(zz[N-1], x,y, m31,m32, s3[0],s3[1])  
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D(zInt[kk], x,y, m31,m32, s3[0],s3[1])
                compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                if x==0 and y==0 and Mfirst <= 0:
                    b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                if x==1 and y==1 and Mlast >= 0:
                    b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                for kk in range(0, N):
                    r[kk] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                for kk in range(0, N):
                    phi[ii, jj, kk] = temp[kk]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc_tetra(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc_tetra(zInt[kk], nu3)
        # loop through x and y dimensions
        for ii in range(L):
            for jj in range(M):
                x = xx[ii]
                y = yy[jj]
                
                Mfirst = Mfunc3D_auto(zz[0], x,y, m31,m32, s3[0],s3[1],s3[2],s3[3])
                Mlast = Mfunc3D_auto(zz[N-1], x,y, m31,m32, s3[0],s3[1],s3[2],s3[3])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_auto(zInt[kk], x,y, m31,m32, s3[0],s3[1],s3[2],s3[3])
                compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                if x==0 and y==0 and Mfirst <= 0:
                    b[0] += (0.25/nu3 - Mfirst)*2/dz[0] 
                if x==1 and y==1 and Mlast >= 0:
                    b[N-1] += -(-0.25/nu3 - Mlast)*2/dz[N-2]

                for kk in range(0, N):
                    r[kk] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                for kk in range(0, N):
                    phi[ii, jj, kk] = temp[kk]

    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x and y dimensions
        for ii in range(L):
            for jj in range(M):
                x = xx[ii]
                y = yy[jj]
                ### Note: the order of migration params and grids being passed here is different 
                # This is for consistency with the allo cases where the first two dimensions passed
                # to Mfunc need to be the allo subgenomes and the subgenomes are always passed as y and z.
                Mfirst = Mfunc3D_allo_a(zz[0], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                Mlast = Mfunc3D_allo_a(zz[N-1], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_allo_a(zInt[kk], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                if x==0 and y==0 and Mfirst <= 0:
                    b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                if x==1 and y==1 and Mlast >= 0:
                    b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                for kk in range(0, N):
                    r[kk] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                for kk in range(0, N):
                    phi[ii, jj, kk] = temp[kk]

    elif is_allob:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x and y dimensions
        for ii in range(L):
            for jj in range(M):
                x = xx[ii]
                y = yy[jj]
                # See note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_allo_b(zz[0], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                Mlast = Mfunc3D_allo_b(zz[N-1], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_allo_b(zInt[kk], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                if x==0 and y==0 and Mfirst <= 0:
                    b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                if x==1 and y==1 and Mlast >= 0:
                    b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                for kk in range(0, N):
                    r[kk] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                for kk in range(0, N):
                    phi[ii, jj, kk] = temp[kk]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc_hex(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc_hex(zInt[kk], nu3)
        # loop through x and y dimensions
        for ii in range(L):
            for jj in range(M):
                x = xx[ii]
                y = yy[jj]
                
                Mfirst = Mfunc3D_autohex(zz[0], x,y, m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5])
                Mlast = Mfunc3D_autohex(zz[N-1], x,y, m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_autohex(zInt[kk], x,y, m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5])
                compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                if x==0 and y==0 and Mfirst <= 0:
                    b[0] += (1/(6*nu3) - Mfirst)*2/dz[0] 
                if x==1 and y==1 and Mlast >= 0:
                    b[N-1] += -(-1/(6*nu3) - Mlast)*2/dz[N-2]

                for kk in range(0, N):
                    r[kk] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                for kk in range(0, N):
                    phi[ii, jj, kk] = temp[kk]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc_tetra(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc_tetra(zInt[kk], nu3)
        # loop through x and y dimensions
        for ii in range(L):
            for jj in range(M):
                x = xx[ii]
                y = yy[jj]
                # see note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_hex_tetra(zz[0], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                Mlast = Mfunc3D_hex_tetra(zz[N-1], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_hex_tetra(zInt[kk], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                if x==0 and y==0 and Mfirst <= 0:
                    b[0] += (0.25/nu3 - Mfirst)*2/dz[0] 
                if x==1 and y==1 and Mlast >= 0:
                    b[N-1] += -(-0.25/nu3 - Mlast)*2/dz[N-2]

                for kk in range(0, N):
                    r[kk] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                for kk in range(0, N):
                    phi[ii, jj, kk] = temp[kk]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x and y dimensions
        for ii in range(L):
            for jj in range(M):
                x = xx[ii]
                y = yy[jj]
                # see note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_hex_dip(zz[0], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                Mlast = Mfunc3D_hex_dip(zz[N-1], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_hex_dip(zInt[kk], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                if x==0 and y==0 and Mfirst <= 0:
                    b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                if x==1 and y==1 and Mlast >= 0:
                    b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                for kk in range(0, N):
                    r[kk] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                for kk in range(0, N):
                    phi[ii, jj, kk] = temp[kk]

    elif is_hex_c:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x and y dimensions
        for ii in range(L):
            for jj in range(M):
                x = xx[ii]
                y = yy[jj]
                # see note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_hex_c(zz[0], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],
                                       s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25])
                Mlast = Mfunc3D_hex_c(zz[N-1], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],
                                      s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_hex_c(zInt[kk], y,x, m32,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],
                                             s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25])
                compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                if x==0 and y==0 and Mfirst <= 0:
                    b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                if x==1 and y==1 and Mlast >= 0:
                    b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                for kk in range(0, N):
                    r[kk] = phi[ii, jj, kk]/dt
                tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                for kk in range(0, N):
                    phi[ii, jj, kk] = temp[kk]

    tridiag_free()

    return np.asarray(phi)

### ==========================================================================
### CYTHON 3D INTEGRATION FUNCTIONS - CONSTANT PARAMS
### ==========================================================================

def implicit_precalc_3Dx(double[:,:,:] phi, double[:,:,:] ax, double[:,:,:] bx,
                                 double[:,:,:] cx, double dt):
    """
    Implicit 3D integration function for x direction of 3D diffusion equation.
    Uses arrays pre-computed in Python for a, b, c.

    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    ax : numpy array (float64)
        a, b, c, arrays are for tridiagonal matrix solver
    bx : numpy array (float64)
    cx : numpy array (float64)
    dt : Time step

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    
    cdef int ii, jj, kk
    cdef int L = phi.shape[0]
    cdef int M = phi.shape[1]
    cdef int N = phi.shape[2]

    # create memory views for the tridiagonal solver
    cdef double[:] a = np.empty(L, dtype=np.float64)
    cdef double[:] b = np.empty(L, dtype=np.float64)
    cdef double[:] c = np.empty(L, dtype=np.float64)
    cdef double[:] r = np.empty(L, dtype=np.float64)
    cdef double[:] temp = np.empty(L, dtype=np.float64)

    tridiag_malloc(L)

    for jj in range(0, M):
        for kk in range(0, N):
            for ii in range(0, L):
                a[ii] = ax[ii, jj, kk]
                b[ii] = bx[ii, jj, kk] + 1/dt
                c[ii] = cx[ii, jj, kk]
                r[ii] = phi[ii, jj, kk]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
            for ii in range(0, L):
                phi[ii, jj, kk] = temp[ii]

    tridiag_free()

    return np.asarray(phi)

def implicit_precalc_3Dy(double[:,:,:] phi, double[:,:,:] ay, double[:,:,:] by,
                                 double[:,:,:] cy, double dt):
    """
    Implicit 3D integration function for y direction of 3D diffusion equation.
    Uses arrays pre-computed in Python for a, b, c.

    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    ay : numpy array (float64)
        a, b, c, arrays are for tridiagonal matrix solver
    by : numpy array (float64)
    cy : numpy array (float64)
    dt : Time step

    Returns:
    --------
    phi : modified phi after integration in y direction
    """
    
    cdef int ii, jj, kk
    cdef int L = phi.shape[0]
    cdef int M = phi.shape[1]
    cdef int N = phi.shape[2]

    # create memory views for the tridiagonal solver
    cdef double[:] a = np.empty(M, dtype=np.float64)
    cdef double[:] b = np.empty(M, dtype=np.float64)
    cdef double[:] c = np.empty(M, dtype=np.float64)
    cdef double[:] r = np.empty(M, dtype=np.float64)
    cdef double[:] temp = np.empty(M, dtype=np.float64)

    tridiag_malloc(M)

    for ii in range(0, L):
        for kk in range(0, N):
            for jj in range(0, M):
                a[jj] = ay[ii, jj, kk]
                b[jj] = by[ii, jj, kk] + 1/dt
                c[jj] = cy[ii, jj, kk]
                r[jj] = phi[ii, jj, kk]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
            for jj in range(0, M):
                phi[ii, jj, kk] = temp[jj]

    tridiag_free()

    return np.asarray(phi)

def implicit_precalc_3Dz(double[:,:,:] phi, double[:,:,:] az, double[:,:,:] bz,
                                 double[:,:,:] cz, double dt):
    """
    Implicit 3D integration function for z direction of 3D diffusion equation.
    Uses arrays pre-computed in Python for a, b, c.

    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    az : numpy array (float64)
        a, b, c, arrays are for tridiagonal matrix solver
    bz : numpy array (float64)
    cz : numpy array (float64)
    dt : Time step

    Returns:
    --------
    phi : modified phi after integration in z direction
    """
    
    cdef int ii, jj, kk
    cdef int L = phi.shape[0]
    cdef int M = phi.shape[1]
    cdef int N = phi.shape[2]

    # create memory views for the tridiagonal solver
    cdef double[:] a = np.empty(N, dtype=np.float64)
    cdef double[:] b = np.empty(N, dtype=np.float64)
    cdef double[:] c = np.empty(N, dtype=np.float64)
    cdef double[:] r = np.empty(N, dtype=np.float64)
    cdef double[:] temp = np.empty(N, dtype=np.float64)

    tridiag_malloc(N)

    for ii in range(0, L):
        for jj in range(0, M):
            for kk in range(0, N):
                a[kk] = az[ii, jj, kk]
                b[kk] = bz[ii, jj, kk] + 1/dt
                c[kk] = cz[ii, jj, kk]
                r[kk] = phi[ii, jj, kk]/dt
            tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
            for kk in range(0, N):
                phi[ii, jj, kk] = temp[kk]

    tridiag_free()

    return np.asarray(phi)

# =========================================================
# CYTHON 4D INTEGRATION FUNCTIONS
# =========================================================
def implicit_4Dx(double[:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa,
                        double nu1, double m12, double m13, double m14, double[:] s1, 
                        double dt, int use_delj_trick, int[:] ploidy1):
    """
    Implicit 4D integration function for x direction of 4D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz, aa: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu1: Population size for pop1
    m12: Migration rate to pop1 from pop2
    m13: Migration rate to pop1 from pop3
    m14: Migration rate to pop1 from pop4
    s1: vector of selection parameters for pop1
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """

    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x dim
    cdef int M = yy.shape[0] # number of grid points in y dim
    cdef int N = zz.shape[0] # number of grid points in z dim
    cdef int O = aa.shape[0] # number of grid points in a dim
    cdef int ii, jj, kk, ll # loop indices
    cdef double y, z, a_ # single values from y, z, and a grids; note a is reserved for the tridiag solver, hence a_
    
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
    cdef double[:] temp = np.empty(L, dtype=np.float64)
    ### specify ploidy of the x direction
    cdef int is_diploid = ploidy1[0]
    cdef int is_auto = ploidy1[1]
    cdef int is_alloa = ploidy1[2]
    cdef int is_allob = ploidy1[3]
    cdef int is_autohex = ploidy1[4]
    cdef int is_hex_tetra = ploidy1[5] 
    cdef int is_hex_dip = ploidy1[6]

    # we also only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order as the last three pops, 
    # hence no hex_a, hex_b, or hex_c models here

    # compute step size and intermediate values
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
        # loop through y, z, and a dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    y = yy[jj]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D(xx[0], y,z,a_, m12,m13,m14, s1[0],s1[1])
                    Mlast = Mfunc4D(xx[L-1], y,z,a_, m12,m13,m14, s1[0],s1[1])
                    for ii in range(0, L-1):
                        MInt[ii] = Mfunc4D(xInt[ii], y,z,a_, m12,m13,m14, s1[0],s1[1]) 
                    compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                    if y==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                    if y==1 and z==1 and a_==1 and Mlast >= 0:
                        b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                    for ii in range(0, L):
                        r[ii] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                    for ii in range(0, L):
                        phi[ii, jj, kk, ll] = temp[ii]

    elif is_auto:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_tetra(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_tetra(xInt[ii], nu1)
        # loop through y, z, and a dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    y = yy[jj]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_auto(xx[0], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3])
                    Mlast = Mfunc4D_auto(xx[L-1], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3])
                    for ii in range(0, L-1):
                        MInt[ii] = Mfunc4D_auto(xInt[ii], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3])
                    compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                    if y==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.25/nu1 - Mfirst)*2/dx[0] 
                    if y==1 and z==1 and a_==1 and Mlast >= 0:
                        b[L-1] += -(-0.25/nu1 - Mlast)*2/dx[L-2]

                    for ii in range(0, L):
                        r[ii] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                    for ii in range(0, L):
                        phi[ii, jj, kk, ll] = temp[ii]

    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y, z, and a dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    y = yy[jj]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_allo_a(xx[0], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                    Mlast = Mfunc4D_allo_a(xx[L-1], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                    for ii in range(0, L-1):
                        MInt[ii] = Mfunc4D_allo_a(xInt[ii], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                    compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                    if y==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                    if y==1 and z==1 and a_==1 and Mlast >= 0:
                        b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                    for ii in range(0, L):
                        r[ii] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                    for ii in range(0, L):
                        phi[ii, jj, kk, ll] = temp[ii]
    
    elif is_allob:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y, z, and a dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    y = yy[jj]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_allo_b(xx[0], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                    Mlast = Mfunc4D_allo_b(xx[L-1], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                    for ii in range(0, L-1):
                        MInt[ii] = Mfunc4D_allo_b(xInt[ii], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                    compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                    if y==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                    if y==1 and z==1 and a_==1 and Mlast >= 0:
                        b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                    for ii in range(0, L):
                        r[ii] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                    for ii in range(0, L):
                        phi[ii, jj, kk, ll] = temp[ii]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_hex(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_hex(xInt[ii], nu1)
        # loop through y, z, and a dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    y = yy[jj]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_autohex(xx[0], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
                    Mlast = Mfunc4D_autohex(xx[L-1], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
                    for ii in range(0, L-1):
                        MInt[ii] = Mfunc4D_autohex(xInt[ii], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
                    compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                    if y==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (1/(6*nu1) - Mfirst)*2/dx[0] 
                    if y==1 and z==1 and a_==1 and Mlast >= 0:
                        b[L-1] += -(-1/(6*nu1) - Mlast)*2/dx[L-2]

                    for ii in range(0, L):
                        r[ii] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                    for ii in range(0, L):
                        phi[ii, jj, kk, ll] = temp[ii]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_tetra(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_tetra(xInt[ii], nu1)
        # loop through y, z, and a dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    y = yy[jj]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_hex_tetra(xx[0], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                    Mlast = Mfunc4D_hex_tetra(xx[L-1], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                    for ii in range(0, L-1):
                        MInt[ii] = Mfunc4D_hex_tetra(xInt[ii], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                    compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                    if y==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.25/nu1 - Mfirst)*2/dx[0] 
                    if y==1 and z==1 and a_==1 and Mlast >= 0:
                        b[L-1] += -(-0.25/nu1 - Mlast)*2/dx[L-2]

                    for ii in range(0, L):
                        r[ii] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                    for ii in range(0, L):
                        phi[ii, jj, kk, ll] = temp[ii]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y, z, and a dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    y = yy[jj]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_hex_dip(xx[0], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                    Mlast = Mfunc4D_hex_dip(xx[L-1], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                    for ii in range(0, L-1):
                        MInt[ii] = Mfunc4D_hex_dip(xInt[ii], y,z,a_, m12,m13,m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                    compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                    if y==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                    if y==1 and z==1 and a_==1 and Mlast >= 0:
                        b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                    for ii in range(0, L):
                        r[ii] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                    for ii in range(0, L):
                        phi[ii, jj, kk, ll] = temp[ii]
    
    tridiag_free()

    return np.asarray(phi)
            
def implicit_4Dy(double[:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa,
                        double nu2, double m21, double m23, double m24, double[:] s2, 
                        double dt, int use_delj_trick, int[:] ploidy2):
    """
    Implicit 4D integration function for y direction of 4D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz, aa: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu2: Population size for pop2
    m21: Migration rate to pop2 from pop1
    m23: Migration rate to pop2 from pop3
    m24: Migration rate to pop2 from pop4
    s2: vector of selection parameters for pop2
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy2: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in y direction
    """

    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int N = zz.shape[0] # number of grid points in z direction
    cdef int O = aa.shape[0] # number of grid points in a direction
    cdef int ii, jj, kk, ll # loop indices
    cdef double x, z, a_ # single values from x, z, and a grids
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] dy = np.empty(M-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(M, dtype=np.float64) 
    cdef double[:] yInt = np.empty(M-1, dtype=np.float64)
    cdef double[:] delj = np.empty(M-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(M-1, dtype=np.float64)
    cdef double[:] V = np.empty(M, dtype=np.float64)
    cdef double[:] VInt = np.empty(M-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(M, dtype=np.float64)
    cdef double[:] b = np.empty(M, dtype=np.float64)
    cdef double[:] c = np.empty(M, dtype=np.float64)
    cdef double[:] r = np.empty(M, dtype=np.float64)
    cdef double[:] temp = np.empty(M, dtype=np.float64)
    ### specify ploidy of the y direction
    cdef int is_diploid = ploidy2[0]
    cdef int is_auto = ploidy2[1]
    cdef int is_alloa = ploidy2[2]
    cdef int is_allob = ploidy2[3]
    cdef int is_autohex = ploidy2[4]
    cdef int is_hex_tetra = ploidy2[5]
    cdef int is_hex_dip = ploidy2[6]
    cdef int is_hex_a = ploidy2[7]

    # we only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order as the last three pops,
    # hence no hex_b or hex_c models here

    # compute step size and intermediate values
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
        # loop through x, z, and a dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    x = xx[ii]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D(yy[0], x,z,a_, m21,m23,m24, s2[0],s2[1])
                    Mlast = Mfunc4D(yy[M-1], x,z,a_, m21,m23,m24, s2[0],s2[1])  
                    for jj in range(0, M-1):
                        MInt[jj] = Mfunc4D(yInt[jj], x,z,a_, m21,m23,m24, s2[0],s2[1])
                    compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                    if x==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                    if x==1 and z==1 and a_==1 and Mlast >= 0:
                        b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                    for jj in range(0, M):
                        r[jj] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                    for jj in range(0, M):
                        phi[ii, jj, kk, ll] = temp[jj]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_tetra(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_tetra(yInt[jj], nu2)
        # loop through x, z, and a dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    x = xx[ii]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_auto(yy[0], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3])
                    Mlast = Mfunc4D_auto(yy[M-1], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3])
                    for jj in range(0, M-1):
                        MInt[jj] = Mfunc4D_auto(yInt[jj], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3])
                    compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                    if x==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.25/nu2 - Mfirst)*2/dy[0] 
                    if x==1 and z==1 and a_==1 and Mlast >= 0:
                        b[M-1] += -(-0.25/nu2 - Mlast)*2/dy[M-2]

                    for jj in range(0, M):
                        r[jj] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                    for jj in range(0, M):
                        phi[ii, jj, kk, ll] = temp[jj]
    
    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x, z, and a dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    x = xx[ii]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_allo_a(yy[0], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                    Mlast = Mfunc4D_allo_a(yy[M-1], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                    for jj in range(0, M-1):
                        MInt[jj] = Mfunc4D_allo_a(yInt[jj], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                    compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                    if x==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                    if x==1 and z==1 and a_==1 and Mlast >= 0:
                        b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                    for jj in range(0, M):
                        r[jj] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                    for jj in range(0, M):
                        phi[ii, jj, kk, ll] = temp[jj]
    
    elif is_allob:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x, z, and a dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    x = xx[ii]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_allo_b(yy[0], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                    Mlast = Mfunc4D_allo_b(yy[M-1], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                    for jj in range(0, M-1):
                        MInt[jj] = Mfunc4D_allo_b(yInt[jj], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                    compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                    if x==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                    if x==1 and z==1 and a_==1 and Mlast >= 0:
                        b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                    for jj in range(0, M):
                        r[jj] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                    for jj in range(0, M):
                        phi[ii, jj, kk, ll] = temp[jj]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_hex(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_hex(yInt[jj], nu2)
        # loop through x, z, and a dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    x = xx[ii]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_autohex(yy[0], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
                    Mlast = Mfunc4D_autohex(yy[M-1], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
                    for jj in range(0, M-1):
                        MInt[jj] = Mfunc4D_autohex(yInt[jj], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
                    compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                    if x==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (1/(6*nu2) - Mfirst)*2/dy[0] 
                    if x==1 and z==1 and a_==1 and Mlast >= 0:
                        b[M-1] += -(-1/(6*nu2) - Mlast)*2/dy[M-2]

                    for jj in range(0, M):
                        r[jj] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                    for jj in range(0, M):
                        phi[ii, jj, kk, ll] = temp[jj]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_tetra(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_tetra(yInt[jj], nu2)
        # loop through x, z, and a dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    x = xx[ii]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_hex_tetra(yy[0], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                    Mlast = Mfunc4D_hex_tetra(yy[M-1], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                    for jj in range(0, M-1):
                        MInt[jj] = Mfunc4D_hex_tetra(yInt[jj], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                    compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                    if x==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.25/nu2 - Mfirst)*2/dy[0] 
                    if x==1 and z==1 and a_==1 and Mlast >= 0:
                        b[M-1] += -(-0.25/nu2 - Mlast)*2/dy[M-2]

                    for jj in range(0, M):
                        r[jj] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                    for jj in range(0, M):
                        phi[ii, jj, kk, ll] = temp[jj]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x, z, and a dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    x = xx[ii]
                    z = zz[kk]
                    a_ = aa[ll]

                    Mfirst = Mfunc4D_hex_dip(yy[0], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                    Mlast = Mfunc4D_hex_dip(yy[M-1], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                    for jj in range(0, M-1):
                        MInt[jj] = Mfunc4D_hex_dip(yInt[jj], x,z,a_, m21,m23,m24, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                    compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                    if x==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                    if x==1 and z==1 and a_==1 and Mlast >= 0:
                        b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                    for jj in range(0, M):
                        r[jj] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                    for jj in range(0, M):
                        phi[ii, jj, kk, ll] = temp[jj]

    elif is_hex_a:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x, z, and a dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    x = xx[ii]
                    z = zz[kk]
                    a_ = aa[ll]
                    # Note the order of migration params and grids being passed here is different
                    # This is to support a 2+2+2 hexaploid which must be passed as the last three pops/dimensions of phi
                    # and the Mfuncs are written so that the subgenome dimensions and migration/exchange params are first
                    Mfirst = Mfunc4D_hex_a(yy[0], z,a_,x, m23,m24,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                           s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25])
                    Mlast = Mfunc4D_hex_a(yy[M-1], z,a_,x, m23,m24,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                           s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25]) 
                    for jj in range(0, M-1):
                        MInt[jj] = Mfunc4D_hex_a(yInt[jj], z,a_,x, m23,m24,m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                                 s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25])
                    compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                    if x==0 and z==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                    if x==1 and z==1 and a_==1 and Mlast >= 0:
                        b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                    for jj in range(0, M):
                        r[jj] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                    for jj in range(0, M):
                        phi[ii, jj, kk, ll] = temp[jj]

    tridiag_free()

    return np.asarray(phi)

def implicit_4Dz(double[:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa,
                        double nu3, double m31, double m32, double m34, double[:] s3, 
                        double dt, int use_delj_trick, int[:] ploidy3):
    """
    Implicit 4D integration function for z direction of 4D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz, aa: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu3: Population size for pop3
    m31: Migration rate to pop3 from pop1
    m32: Migration rate to pop3 from pop2
    m34: Migration rate to pop3 from pop4
    s3: vector of selection parameters for pop3
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy3: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in z direction
    """

    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int N = zz.shape[0] # number of grid points in z direction
    cdef int O = aa.shape[0] # number of grid points in a direction
    cdef int ii, jj, kk, ll # loop indices
    cdef double x, y, a_ # single values from x, y, and a grids
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] dz = np.empty(N-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(N, dtype=np.float64) 
    cdef double[:] zInt = np.empty(N-1, dtype=np.float64)
    cdef double[:] delj = np.empty(N-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(N-1, dtype=np.float64)
    cdef double[:] V = np.empty(N, dtype=np.float64)
    cdef double[:] VInt = np.empty(N-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(N, dtype=np.float64)
    cdef double[:] b = np.empty(N, dtype=np.float64)
    cdef double[:] c = np.empty(N, dtype=np.float64)
    cdef double[:] r = np.empty(N, dtype=np.float64)
    cdef double[:] temp = np.empty(N, dtype=np.float64)
    ### specify ploidy of the z direction
    cdef int is_diploid = ploidy3[0]
    cdef int is_auto = ploidy3[1]
    cdef int is_alloa = ploidy3[2]
    cdef int is_allob = ploidy3[3]
    cdef int is_autohex = ploidy3[4]
    cdef int is_hex_tetra = ploidy3[5]
    cdef int is_hex_dip = ploidy3[6]
    cdef int is_hex_b = ploidy3[8]

    # we only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order as the last three pops,
    # hence no hex_a or hex_c models here

    # compute step size and intermediate values
    compute_dx(&zz[0], N, &dz[0])
    compute_dfactor(&dz[0], N, &dfactor[0])
    compute_xInt(&zz[0], N, &zInt[0])
    # dynamic allocation of memory for tridiag
    tridiag_malloc(N)

    # branch on ploidy
    if is_diploid:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x, y, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    x = xx[ii]
                    y = yy[jj]
                    a_ = aa[ll]
                
                    Mfirst = Mfunc4D(zz[0], x,y,a_, m31,m32,m34, s3[0],s3[1])
                    Mlast = Mfunc4D(zz[N-1], x,y,a_, m31,m32,m34, s3[0],s3[1])  
                    for kk in range(0, N-1):
                        MInt[kk] = Mfunc4D(zInt[kk], x,y,a_, m31,m32,m34, s3[0],s3[1])
                    compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                    if x==1 and y==1 and a_==1 and Mlast >= 0:
                        b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                    for kk in range(0, N):
                        r[kk] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                    for kk in range(0, N):
                        phi[ii, jj, kk, ll] = temp[kk]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc_tetra(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc_tetra(zInt[kk], nu3)
        # loop through x, y, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    x = xx[ii]
                    y = yy[jj]
                    a_ = aa[ll]
                
                    Mfirst = Mfunc4D_auto(zz[0], x,y,a_, m31,m32,m34, s3[0],s3[1],s3[2],s3[3])
                    Mlast = Mfunc4D_auto(zz[N-1], x,y,a_, m31,m32,m34, s3[0],s3[1],s3[2],s3[3])
                    for kk in range(0, N-1):
                        MInt[kk] = Mfunc4D_auto(zInt[kk], x,y,a_, m31,m32,m34, s3[0],s3[1],s3[2],s3[3])
                    compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.25/nu3 - Mfirst)*2/dz[0] 
                    if x==1 and y==1 and a_==1 and Mlast >= 0:
                        b[N-1] += -(-0.25/nu3 - Mlast)*2/dz[N-2]

                    for kk in range(0, N):
                        r[kk] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                    for kk in range(0, N):
                        phi[ii, jj, kk, ll] = temp[kk]

    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x, y, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    x = xx[ii]
                    y = yy[jj]
                    a_ = aa[ll]
                    ### Note: the order of migration params and grids being passed here is different 
                    # This is for consistency with the allo cases where the first two dimensions passed
                    # to Mfunc need to be the allo subgenomes and the subgenomes are always passed 
                    # to the integrator as z and a.
                    Mfirst = Mfunc4D_allo_a(zz[0], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                    Mlast = Mfunc4D_allo_a(zz[N-1], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                    for kk in range(0, N-1):
                        MInt[kk] = Mfunc4D_allo_a(zInt[kk], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                    compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                    if x==1 and y==1 and a_==1 and Mlast >= 0:
                        b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                    for kk in range(0, N):
                        r[kk] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                    for kk in range(0, N):
                        phi[ii, jj, kk, ll] = temp[kk]

    elif is_allob:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x, y, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    x = xx[ii]
                    y = yy[jj]
                    a_ = aa[ll]
                    # see note above about the order of the params passed to Mfuncs here
                    Mfirst = Mfunc4D_allo_b(zz[0], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                    Mlast = Mfunc4D_allo_b(zz[N-1], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                    for kk in range(0, N-1):
                        MInt[kk] = Mfunc4D_allo_b(zInt[kk], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                    compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                    if x==1 and y==1 and a_==1 and Mlast >= 0:
                        b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                    for kk in range(0, N):
                        r[kk] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                    for kk in range(0, N):
                        phi[ii, jj, kk, ll] = temp[kk]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc_hex(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc_hex(zInt[kk], nu3)
        # loop through x, y, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    x = xx[ii]
                    y = yy[jj]
                    a_ = aa[ll]
                
                    Mfirst = Mfunc4D_autohex(zz[0], x,y,a_, m31,m32,m34, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5])
                    Mlast = Mfunc4D_autohex(zz[N-1], x,y,a_, m31,m32,m34, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5])
                    for kk in range(0, N-1):
                        MInt[kk] = Mfunc4D_autohex(zInt[kk], x,y,a_, m31,m32,m34, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5])
                    compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and a_==0 and Mfirst <= 0:
                        b[0] += (1/(6*nu3) - Mfirst)*2/dz[0] 
                    if x==1 and y==1 and a_==1 and Mlast >= 0:
                        b[N-1] += -(-1/(6*nu3) - Mlast)*2/dz[N-2]

                    for kk in range(0, N):
                        r[kk] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                    for kk in range(0, N):
                        phi[ii, jj, kk, ll] = temp[kk]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc_tetra(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc_tetra(zInt[kk], nu3)
        # loop through x, y, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    x = xx[ii]
                    y = yy[jj]
                    a_ = aa[ll]
                    # see note above about the order of the params passed to Mfuncs here
                    Mfirst = Mfunc4D_hex_tetra(zz[0], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                    Mlast = Mfunc4D_hex_tetra(zz[N-1], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                    for kk in range(0, N-1):
                        MInt[kk] = Mfunc4D_hex_tetra(zInt[kk], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                    compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.25/nu3 - Mfirst)*2/dz[0] 
                    if x==1 and y==1 and a_==1 and Mlast >= 0:
                        b[N-1] += -(-0.25/nu3 - Mlast)*2/dz[N-2]

                    for kk in range(0, N):
                        r[kk] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                    for kk in range(0, N):
                        phi[ii, jj, kk, ll] = temp[kk]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x, y, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    x = xx[ii]
                    y = yy[jj]
                    a_ = aa[ll]
                    # see note above about the order of the params passed to Mfuncs here
                    Mfirst = Mfunc4D_hex_dip(zz[0], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                    Mlast = Mfunc4D_hex_dip(zz[N-1], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                    for kk in range(0, N-1):
                        MInt[kk] = Mfunc4D_hex_dip(zInt[kk], a_,x,y, m34,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13])
                    compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                    if x==1 and y==1 and a_==1 and Mlast >= 0:
                        b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                    for kk in range(0, N):
                        r[kk] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                    for kk in range(0, N):
                        phi[ii, jj, kk, ll] = temp[kk]

    elif is_hex_b:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x, y, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    x = xx[ii]
                    y = yy[jj]
                    a_ = aa[ll]
                    # Note the order of migration params and grids being passed here is different
                    # This is to support a 2+2+2 hexaploid which must be passed as the last three pops/dimensions of phi
                    # and the Mfuncs are written so that the subgenome dimensions and migration/exchange params are first
                    Mfirst = Mfunc4D_hex_b(zz[0], y,a_,x, m32,m34,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],
                                           s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25])
                    Mlast = Mfunc4D_hex_b(zz[N-1], y,a_,x, m32,m34,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],
                                           s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25])  
                    for kk in range(0, N-1):
                        MInt[kk] = Mfunc4D_hex_b(zInt[kk], y,a_,x, m32,m34,m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],
                                                 s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25])
                    compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                    compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and a_==0 and Mfirst <= 0:
                        b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                    if x==1 and y==1 and a_==1 and Mlast >= 0:
                        b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                    for kk in range(0, N):
                        r[kk] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                    for kk in range(0, N):
                        phi[ii, jj, kk, ll] = temp[kk]

    tridiag_free()

    return np.asarray(phi)

def implicit_4Da(double[:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa,
                        double nu4, double m41, double m42, double m43, double[:] s4, 
                        double dt, int use_delj_trick, int[:] ploidy4):
    """
    Implicit 4D integration function for a direction of 4D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz, aa: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu4: Population size for pop4
    m41: Migration rate to pop4 from pop1
    m42: Migration rate to pop4 from pop2
    m43: Migration rate to pop4 from pop3
    s4: vector of selection parameters for pop4
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy4: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in a direction
    """
    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int N = zz.shape[0] # number of grid points in z direction
    cdef int O = aa.shape[0] # number of grid points in a direction
    cdef int ii, jj, kk, ll # loop indices
    cdef double x, y, z # single values from x, y, and z grids
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] da = np.empty(O-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(O, dtype=np.float64) 
    cdef double[:] aInt = np.empty(O-1, dtype=np.float64)
    cdef double[:] delj = np.empty(O-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(O-1, dtype=np.float64)
    cdef double[:] V = np.empty(O, dtype=np.float64)
    cdef double[:] VInt = np.empty(O-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(O, dtype=np.float64)
    cdef double[:] b = np.empty(O, dtype=np.float64)
    cdef double[:] c = np.empty(O, dtype=np.float64)
    cdef double[:] r = np.empty(O, dtype=np.float64)
    cdef double[:] temp = np.empty(O, dtype=np.float64)
    ### specify ploidy of the a direction
    cdef int is_diploid = ploidy4[0]
    cdef int is_auto = ploidy4[1]
    cdef int is_alloa = ploidy4[2]
    cdef int is_allob = ploidy4[3]
    cdef int is_autohex = ploidy4[4]
    cdef int is_hex_tetra = ploidy4[5]    
    cdef int is_hex_dip = ploidy4[6]
    cdef int is_hex_c = ploidy4[9]

    # we only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order as the last three pops, 
    # hence no hex_a or hex_b models here

    # compute step size and intermediate values
    compute_dx(&aa[0], O, &da[0])
    compute_dfactor(&da[0], O, &dfactor[0])
    compute_xInt(&aa[0], O, &aInt[0])
    # dynamic allocation of memory for tridiag
    tridiag_malloc(O)

    # branch on ploidy
    if is_diploid:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, and z dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    x = xx[ii]
                    y = yy[jj]
                    z = zz[kk]
                
                    Mfirst = Mfunc4D(aa[0], x,y,z, m41,m42,m43, s4[0],s4[1])
                    Mlast = Mfunc4D(aa[O-1], x,y,z, m41,m42,m43, s4[0],s4[1])  
                    for ll in range(0, O-1):
                        MInt[ll] = Mfunc4D(aInt[ll], x,y,z, m41,m42,m43, s4[0],s4[1])
                    compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                    compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and z==0 and Mfirst <= 0:
                        b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                    if x==1 and y==1 and z==1 and Mlast >= 0:
                        b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                    for ll in range(0, O):
                        r[ll] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                    for ll in range(0, O):
                        phi[ii, jj, kk, ll] = temp[ll]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc_tetra(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc_tetra(aInt[ll], nu4)
        # loop through x, y, and z dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    x = xx[ii]
                    y = yy[jj]
                    z = zz[kk]
            
                    Mfirst = Mfunc4D_auto(aa[0], x,y,z, m41,m42,m43, s4[0],s4[1],s4[2],s4[3])
                    Mlast = Mfunc4D_auto(aa[O-1], x,y,z, m41,m42,m43, s4[0],s4[1],s4[2],s4[3])
                    for ll in range(0, O-1):
                        MInt[ll] = Mfunc4D_auto(aInt[ll], x,y,z, m41,m42,m43, s4[0],s4[1],s4[2],s4[3])
                    compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                    compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and z==0 and Mfirst <= 0:
                        b[0] += (0.25/nu4 - Mfirst)*2/da[0] 
                    if x==1 and y==1 and z==1 and Mlast >= 0:
                        b[O-1] += -(-0.25/nu4 - Mlast)*2/da[O-2]

                    for ll in range(0, O):
                        r[ll] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                    for ll in range(0, O):
                        phi[ii, jj, kk, ll] = temp[ll]
        
    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, and z dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    x = xx[ii]
                    y = yy[jj]
                    z = zz[kk]
                    ### Note: the order of migration params and grids being passed here is different 
                    # This is for consistency with the allo cases where the first two dimensions passed
                    # to Mfunc need to be the allo subgenomes and the subgenomes are always passed 
                    # to the integrator as z and a.
                    Mfirst = Mfunc4D_allo_a(aa[0], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                    Mlast = Mfunc4D_allo_a(aa[O-1], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                    for ll in range(0, O-1):
                        MInt[ll] = Mfunc4D_allo_a(aInt[ll], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                    compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                    compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and z==0 and Mfirst <= 0:
                        b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                    if x==1 and y==1 and z==1 and Mlast >= 0:
                        b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                    for ll in range(0, O):
                        r[ll] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                    for ll in range(0, O):
                        phi[ii, jj, kk, ll] = temp[ll]

    elif is_allob:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, and z dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    x = xx[ii]
                    y = yy[jj]
                    z = zz[kk]
                    # see note above about the order of the params passed to Mfuncs here
                    Mfirst = Mfunc4D_allo_b(aa[0], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                    Mlast = Mfunc4D_allo_b(aa[O-1], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                    for ll in range(0, O-1):
                        MInt[ll] = Mfunc4D_allo_b(aInt[ll], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                    compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                    compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and z==0 and Mfirst <= 0:
                        b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                    if x==1 and y==1 and z==1 and Mlast >= 0:
                        b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                    for ll in range(0, O):
                        r[ll] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                    for ll in range(0, O):
                        phi[ii, jj, kk, ll] = temp[ll]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc_hex(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc_hex(aInt[ll], nu4)
        # loop through x, y, and z dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    x = xx[ii]
                    y = yy[jj]
                    z = zz[kk]
            
                    Mfirst = Mfunc4D_autohex(aa[0], x,y,z, m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5])
                    Mlast = Mfunc4D_autohex(aa[O-1], x,y,z, m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5])
                    for ll in range(0, O-1):
                        MInt[ll] = Mfunc4D_autohex(aInt[ll], x,y,z, m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5])
                    compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                    compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and z==0 and Mfirst <= 0:
                        b[0] += (1/(6*nu4) - Mfirst)*2/da[0] 
                    if x==1 and y==1 and z==1 and Mlast >= 0:
                        b[O-1] += -(-1/(6*nu4) - Mlast)*2/da[O-2]

                    for ll in range(0, O):
                        r[ll] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                    for ll in range(0, O):
                        phi[ii, jj, kk, ll] = temp[ll]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc_tetra(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc_tetra(aInt[ll], nu4)
        # loop through x, y, and z dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    x = xx[ii]
                    y = yy[jj]
                    z = zz[kk]
                    # see note above about the order of the params passed to Mfuncs here
                    Mfirst = Mfunc4D_hex_tetra(aa[0], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                    Mlast = Mfunc4D_hex_tetra(aa[O-1], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                    for ll in range(0, O-1):
                        MInt[ll] = Mfunc4D_hex_tetra(aInt[ll], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                    compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                    compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and z==0 and Mfirst <= 0:
                        b[0] += (0.25/nu4 - Mfirst)*2/da[0] 
                    if x==1 and y==1 and z==1 and Mlast >= 0:
                        b[O-1] += -(-0.25/nu4 - Mlast)*2/da[O-2]

                    for ll in range(0, O):
                        r[ll] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                    for ll in range(0, O):
                        phi[ii, jj, kk, ll] = temp[ll]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, and z dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    x = xx[ii]
                    y = yy[jj]
                    z = zz[kk]
                    # see note above about the order of the params passed to Mfuncs here
                    Mfirst = Mfunc4D_hex_dip(aa[0], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                    Mlast = Mfunc4D_hex_dip(aa[O-1], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                    for ll in range(0, O-1):
                        MInt[ll] = Mfunc4D_hex_dip(aInt[ll], z,x,y, m43,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                    compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                    compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and z==0 and Mfirst <= 0:
                        b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                    if x==1 and y==1 and z==1 and Mlast >= 0:
                        b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                    for ll in range(0, O):
                        r[ll] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                    for ll in range(0, O):
                        phi[ii, jj, kk, ll] = temp[ll]

    elif is_hex_c:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, and z dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    x = xx[ii]
                    y = yy[jj]
                    z = zz[kk]
                    # Note the order of migration params and grids being passed here is different
                    # This is to support a 2+2+2 hexaploid which must be passed as the last three pops/dimensions of phi
                    # and the Mfuncs are written so that the subgenome dimensions and migration/exchange params are first
                    Mfirst = Mfunc4D_hex_c(aa[0], y,z,x, m42,m43,m41, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],
                                           s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25])
                    Mlast = Mfunc4D_hex_c(aa[O-1], y,z,x, m42,m43,m41, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],
                                           s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25])
                    for ll in range(0, O-1):
                        MInt[ll] = Mfunc4D_hex_c(aInt[ll], y,z,x, m42,m43,m41, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],
                                           s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25])
                    compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                    compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                    if x==0 and y==0 and z==0 and Mfirst <= 0:
                        b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                    if x==1 and y==1 and z==1 and Mlast >= 0:
                        b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                    for ll in range(0, O):
                        r[ll] = phi[ii, jj, kk, ll]/dt
                    tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                    for ll in range(0, O):
                        phi[ii, jj, kk, ll] = temp[ll]
        
    tridiag_free()

    return np.asarray(phi)

# =========================================================
# CYTHON 5D INTEGRATION FUNCTIONS
# =========================================================
def implicit_5Dx(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu1, double m12, double m13, double m14, double m15, double[:] s1, 
                        double dt, int use_delj_trick, int[:] ploidy1):
    """
    Implicit 5D integration function for x direction of 5D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz, aa, bb: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu1: Population size for pop1
    m12: Migration rate to pop1 from pop2
    m13: Migration rate to pop1 from pop3
    m14: Migration rate to pop1 from pop4
    m15: Migration rate to pop1 from pop5
    s1: vector of selection parameters for pop1
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x dim
    cdef int M = yy.shape[0] # number of grid points in y dim
    cdef int N = zz.shape[0] # number of grid points in z dim
    cdef int O = aa.shape[0] # number of grid points in a dim
    cdef int P = bb.shape[0] # number of grid points in b dim
    cdef int ii, jj, kk, ll, mm # loop indices
    cdef double y, z, a_, b_ # single values from y, z, a, and b grids 
    # note a and b is reserved for the tridiag solver, hence a_ and b_
    
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
    cdef double[:] temp = np.empty(L, dtype=np.float64)
    ### specify ploidy of the x direction
    cdef int is_diploid = ploidy1[0]
    cdef int is_auto = ploidy1[1]
    cdef int is_alloa = ploidy1[2]
    cdef int is_allob = ploidy1[3]
    cdef int is_autohex = ploidy1[4]
    cdef int is_hex_tetra = ploidy1[5]
    cdef int is_hex_dip = ploidy1[6]

    # we only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order as the last three pops, 
    # hence no hex_a, hex_b, or hex_c models here

    # compute step size and intermediate values
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
        # loop through y, z, a, and b dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):        
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D(xx[0], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1])
                        Mlast = Mfunc5D(xx[L-1], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1])
                        for ii in range(0, L-1):
                            MInt[ii] = Mfunc5D(xInt[ii], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1]) 
                        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                        if y==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                        if y==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                        for ii in range(0, L):
                            r[ii] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                        for ii in range(0, L):
                            phi[ii, jj, kk, ll, mm] = temp[ii]

    elif is_auto:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_tetra(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_tetra(xInt[ii], nu1)
        # loop through y, z, a, and b dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):        
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_auto(xx[0], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3])
                        Mlast = Mfunc5D_auto(xx[L-1], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3])
                        for ii in range(0, L-1):
                            MInt[ii] = Mfunc5D_auto(xInt[ii], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3]) 
                        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                        if y==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.25/nu1 - Mfirst)*2/dx[0] 
                        if y==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[L-1] += -(-0.25/nu1 - Mlast)*2/dx[L-2]

                        for ii in range(0, L):
                            r[ii] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                        for ii in range(0, L):
                            phi[ii, jj, kk, ll, mm] = temp[ii]

    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y, z, a, and b dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):        
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_allo_a(xx[0], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                        Mlast = Mfunc5D_allo_a(xx[L-1], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                        for ii in range(0, L-1):
                            MInt[ii] = Mfunc5D_allo_a(xInt[ii], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                        if y==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                        if y==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                        for ii in range(0, L):
                            r[ii] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                        for ii in range(0, L):
                            phi[ii, jj, kk, ll, mm] = temp[ii]
    
    elif is_allob:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y, z, a, and b dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):        
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_allo_b(xx[0], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                        Mlast = Mfunc5D_allo_b(xx[L-1], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                        for ii in range(0, L-1):
                            MInt[ii] = Mfunc5D_allo_b(xInt[ii], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                        if y==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                        if y==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                        for ii in range(0, L):
                            r[ii] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                        for ii in range(0, L):
                            phi[ii, jj, kk, ll, mm] = temp[ii]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_hex(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_hex(xInt[ii], nu1)
        # loop through y, z, a, and b dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):        
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_autohex(xx[0], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
                        Mlast = Mfunc5D_autohex(xx[L-1], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5])
                        for ii in range(0, L-1):
                            MInt[ii] = Mfunc5D_autohex(xInt[ii], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5]) 
                        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                        if y==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (1/(6*nu1) - Mfirst)*2/dx[0] 
                        if y==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[L-1] += -(-1/(6*nu1) - Mlast)*2/dx[L-2]

                        for ii in range(0, L):
                            r[ii] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                        for ii in range(0, L):
                            phi[ii, jj, kk, ll, mm] = temp[ii]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc_tetra(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc_tetra(xInt[ii], nu1)
        # loop through y, z, a, and b dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):        
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_hex_tetra(xx[0], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                        Mlast = Mfunc5D_hex_tetra(xx[L-1], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                        for ii in range(0, L-1):
                            MInt[ii] = Mfunc5D_hex_tetra(xInt[ii], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13]) 
                        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                        if y==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.25/nu1 - Mfirst)*2/dx[0] 
                        if y==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[L-1] += -(-0.25/nu1 - Mlast)*2/dx[L-2]

                        for ii in range(0, L):
                            r[ii] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                        for ii in range(0, L):
                            phi[ii, jj, kk, ll, mm] = temp[ii]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        # loop through y, z, a, and b dimensions
        for jj in range(M):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):        
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_hex_dip(xx[0], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                        Mlast = Mfunc5D_hex_dip(xx[L-1], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13])
                        for ii in range(0, L-1):
                            MInt[ii] = Mfunc5D_hex_dip(xInt[ii], y,z,a_,b_, m12,m13,m14,m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13]) 
                        compute_delj(&dx[0], &MInt[0], &VInt[0], L, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dx[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, L, &a[0], &b[0], &c[0])
                        if y==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu1 - Mfirst)*2/dx[0] 
                        if y==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[L-1] += -(-0.5/nu1 - Mlast)*2/dx[L-2]

                        for ii in range(0, L):
                            r[ii] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], L)
                        for ii in range(0, L):
                            phi[ii, jj, kk, ll, mm] = temp[ii]
    
    tridiag_free()

    return np.asarray(phi)
            
def implicit_5Dy(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu2, double m21, double m23, double m24, double m25, double[:] s2, 
                        double dt, int use_delj_trick, int[:] ploidy2):
    """
    Implicit 5D integration function for y direction of 5D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz, aa, bb: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu2: Population size for pop2
    m21: Migration rate to pop2 from pop1
    m23: Migration rate to pop2 from pop3
    m24: Migration rate to pop2 from pop4
    m25: Migration rate to pop2 from pop5   
    s2: vector of selection parameters for pop2
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy2: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in y direction
    """
    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int N = zz.shape[0] # number of grid points in z direction
    cdef int O = aa.shape[0] # number of grid points in a direction
    cdef int P = bb.shape[0] # number of grid points in b direction
    cdef int ii, jj, kk, ll, mm # loop indices
    cdef double x, z, a_, b_ # single values from x, z, a, and b grids
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] dy = np.empty(M-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(M, dtype=np.float64) 
    cdef double[:] yInt = np.empty(M-1, dtype=np.float64)
    cdef double[:] delj = np.empty(M-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(M-1, dtype=np.float64)
    cdef double[:] V = np.empty(M, dtype=np.float64)
    cdef double[:] VInt = np.empty(M-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(M, dtype=np.float64)
    cdef double[:] b = np.empty(M, dtype=np.float64)
    cdef double[:] c = np.empty(M, dtype=np.float64)
    cdef double[:] r = np.empty(M, dtype=np.float64)
    cdef double[:] temp = np.empty(M, dtype=np.float64)
    ### specify ploidy of the y direction
    cdef int is_diploid = ploidy2[0]
    cdef int is_auto = ploidy2[1]
    cdef int is_alloa = ploidy2[2]
    cdef int is_allob = ploidy2[3]
    cdef int is_autohex = ploidy2[4]
    cdef int is_hex_tetra = ploidy2[5]
    cdef int is_hex_dip = ploidy2[6]

    # we only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order as the last three pops, 
    # hence no hex_a, hex_b, or hex_c models here

    # compute step size and intermediate values
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
        # loop through x, z, a, and b dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D(yy[0], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1])
                        Mlast = Mfunc5D(yy[M-1], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1])  
                        for jj in range(0, M-1):
                            MInt[jj] = Mfunc5D(yInt[jj], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1])
                        compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                        if x==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                        if x==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                        for jj in range(0, M):
                            r[jj] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                        for jj in range(0, M):
                            phi[ii, jj, kk, ll, mm] = temp[jj]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_tetra(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_tetra(yInt[jj], nu2)
        # loop through x, z, a, and b dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_auto(yy[0], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3])
                        Mlast = Mfunc5D_auto(yy[M-1], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3])
                        for jj in range(0, M-1):
                            MInt[jj] = Mfunc5D_auto(yInt[jj], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3])
                        compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                        if x==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.25/nu2 - Mfirst)*2/dy[0] 
                        if x==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[M-1] += -(-0.25/nu2 - Mlast)*2/dy[M-2]

                        for jj in range(0, M):
                            r[jj] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                        for jj in range(0, M):
                            phi[ii, jj, kk, ll, mm] = temp[jj]
    
    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x, z, a, and b dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_allo_a(yy[0], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                        Mlast = Mfunc5D_allo_a(yy[M-1], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                        for jj in range(0, M-1):
                            MInt[jj] = Mfunc5D_allo_a(yInt[jj], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                        compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                        if x==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                        if x==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                        for jj in range(0, M):
                            r[jj] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                        for jj in range(0, M):
                            phi[ii, jj, kk, ll, mm] = temp[jj]
    
    elif is_allob:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x, z, a, and b dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_allo_b(yy[0], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                        Mlast = Mfunc5D_allo_b(yy[M-1], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                        for jj in range(0, M-1):
                            MInt[jj] = Mfunc5D_allo_b(yInt[jj], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                        compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                        if x==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                        if x==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                        for jj in range(0, M):
                            r[jj] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                        for jj in range(0, M):
                            phi[ii, jj, kk, ll, mm] = temp[jj]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_hex(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_hex(yInt[jj], nu2)
        # loop through x, z, a, and b dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_autohex(yy[0], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
                        Mlast = Mfunc5D_autohex(yy[M-1], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
                        for jj in range(0, M-1):
                            MInt[jj] = Mfunc5D_autohex(yInt[jj], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5])
                        compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                        if x==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (1/(6*nu2) - Mfirst)*2/dy[0] 
                        if x==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[M-1] += -(-1/(6*nu2) - Mlast)*2/dy[M-2]

                        for jj in range(0, M):
                            r[jj] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                        for jj in range(0, M):
                            phi[ii, jj, kk, ll, mm] = temp[jj]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc_tetra(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc_tetra(yInt[jj], nu2)
        # loop through x, z, a, and b dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_hex_tetra(yy[0], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                        Mlast = Mfunc5D_hex_tetra(yy[M-1], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                        for jj in range(0, M-1):
                            MInt[jj] = Mfunc5D_hex_tetra(yInt[jj], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                        compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                        if x==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.25/nu2 - Mfirst)*2/dy[0] 
                        if x==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[M-1] += -(-0.25/nu2 - Mlast)*2/dy[M-2]

                        for jj in range(0, M):
                            r[jj] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                        for jj in range(0, M):
                            phi[ii, jj, kk, ll, mm] = temp[jj]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for jj in range(0, M):
            V[jj] = Vfunc(yy[jj], nu2)
        for jj in range(0, M-1):
            VInt[jj] = Vfunc(yInt[jj], nu2)
        # loop through x, z, a, and b dimensions
        for ii in range(L):
            for kk in range(N):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        z = zz[kk]
                        a_ = aa[ll]
                        b_ = bb[mm]

                        Mfirst = Mfunc5D_hex_dip(yy[0], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                        Mlast = Mfunc5D_hex_dip(yy[M-1], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                        for jj in range(0, M-1):
                            MInt[jj] = Mfunc5D_hex_dip(yInt[jj], x,z,a_,b_, m21,m23,m24,m25, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13])
                        compute_delj(&dy[0], &MInt[0], &VInt[0], M, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dy[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, M, &a[0], &b[0], &c[0])
                        if x==0 and z==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu2 - Mfirst)*2/dy[0] 
                        if x==1 and z==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[M-1] += -(-0.5/nu2 - Mlast)*2/dy[M-2]

                        for jj in range(0, M):
                            r[jj] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], M)
                        for jj in range(0, M):
                            phi[ii, jj, kk, ll, mm] = temp[jj]

    tridiag_free()

    return np.asarray(phi)

def implicit_5Dz(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu3, double m31, double m32, double m34, double m35, double[:] s3, 
                        double dt, int use_delj_trick, int[:] ploidy3):
    """
    Implicit 5D integration function for z direction of 5D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz, aa, bb: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu3: Population size for pop3
    m31: Migration rate to pop3 from pop1
    m32: Migration rate to pop3 from pop2
    m34: Migration rate to pop3 from pop4
    m35: Migration rate to pop3 from pop5
    s3: vector of selection parameters for pop3
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy3: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in z direction
    """
    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int N = zz.shape[0] # number of grid points in z direction
    cdef int O = aa.shape[0] # number of grid points in a direction
    cdef int P = bb.shape[0] # number of grid points in b direction
    cdef int ii, jj, kk, ll, mm # loop indices
    cdef double x, y, a_, b_ # single values from x, y, a, and b grids
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] dz = np.empty(N-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(N, dtype=np.float64) 
    cdef double[:] zInt = np.empty(N-1, dtype=np.float64)
    cdef double[:] delj = np.empty(N-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(N-1, dtype=np.float64)
    cdef double[:] V = np.empty(N, dtype=np.float64)
    cdef double[:] VInt = np.empty(N-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(N, dtype=np.float64)
    cdef double[:] b = np.empty(N, dtype=np.float64)
    cdef double[:] c = np.empty(N, dtype=np.float64)
    cdef double[:] r = np.empty(N, dtype=np.float64)
    cdef double[:] temp = np.empty(N, dtype=np.float64)
    ### specify ploidy of the z direction
    cdef int is_diploid = ploidy3[0]
    cdef int is_auto = ploidy3[1]
    cdef int is_autohex = ploidy3[4]
    cdef int is_hex_a = ploidy3[7]
    # note: we don't support alloa, allob, hex_tetra, or hex_dip as being 
    # the third (middle) dimension of the phi array in a 5D model

    # we only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order as the last three pops, 
    # hence no hex_b or hex_c models here

    # compute step size and intermediate values
    compute_dx(&zz[0], N, &dz[0])
    compute_dfactor(&dz[0], N, &dfactor[0])
    compute_xInt(&zz[0], N, &zInt[0])
    # dynamic allocation of memory for tridiag
    tridiag_malloc(N)

    # branch on ploidy
    if is_diploid:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x, y, a, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        a_ = aa[ll]
                        b_ = bb[mm] 
                
                        Mfirst = Mfunc5D(zz[0], x,y,a_,b_, m31,m32,m34,m35, s3[0],s3[1])
                        Mlast = Mfunc5D(zz[N-1], x,y,a_,b_, m31,m32,m34,m35, s3[0],s3[1])  
                        for kk in range(0, N-1):
                            MInt[kk] = Mfunc5D(zInt[kk], x,y,a_,b_, m31,m32,m34,m35, s3[0],s3[1])
                        compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                        if x==1 and y==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                        for kk in range(0, N):
                            r[kk] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                        for kk in range(0, N):
                            phi[ii, jj, kk, ll, mm] = temp[kk]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc_tetra(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc_tetra(zInt[kk], nu3)
        # loop through x, y, a, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        a_ = aa[ll]
                        b_ = bb[mm] 
                
                        Mfirst = Mfunc5D_auto(zz[0], x,y,a_,b_, m31,m32,m34,m35, s3[0],s3[1],s3[2],s3[3])
                        Mlast = Mfunc5D_auto(zz[N-1], x,y,a_,b_, m31,m32,m34,m35, s3[0],s3[1],s3[2],s3[3])
                        for kk in range(0, N-1):
                            MInt[kk] = Mfunc5D_auto(zInt[kk], x,y,a_,b_, m31,m32,m34,m35, s3[0],s3[1],s3[2],s3[3])
                        compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.25/nu3 - Mfirst)*2/dz[0] 
                        if x==1 and y==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[N-1] += -(-0.25/nu3 - Mlast)*2/dz[N-2]

                        for kk in range(0, N):
                            r[kk] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                        for kk in range(0, N):
                            phi[ii, jj, kk, ll, mm] = temp[kk]

    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc_hex(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc_hex(zInt[kk], nu3)
        # loop through x, y, a, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        a_ = aa[ll]
                        b_ = bb[mm] 
                
                        Mfirst = Mfunc5D_autohex(zz[0], x,y,a_,b_, m31,m32,m34,m35, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5])
                        Mlast = Mfunc5D_autohex(zz[N-1], x,y,a_,b_, m31,m32,m34,m35, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5])
                        for kk in range(0, N-1):
                            MInt[kk] = Mfunc5D_autohex(zInt[kk], x,y,a_,b_, m31,m32,m34,m35, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5])
                        compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (1/(6*nu3) - Mfirst)*2/dz[0]
                        if x==1 and y==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[N-1] += -(-1/(6*nu3) - Mlast)*2/dz[N-2]

                        for kk in range(0, N):
                            r[kk] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                        for kk in range(0, N):
                            phi[ii, jj, kk, ll, mm] = temp[kk]

    elif is_hex_a:
        # compute everything we can outside of the spatial loop
        for kk in range(0, N):
            V[kk] = Vfunc(zz[kk], nu3)
        for kk in range(0, N-1):
            VInt[kk] = Vfunc(zInt[kk], nu3)
        # loop through x, y, a, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for ll in range(O):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        a_ = aa[ll]
                        b_ = bb[mm] 
                        # Note the order of migration params and grids being passed here is different
                        # This is to support a 2+2+2 hexaploid which must be passed as the last three pops/dimensions of phi
                        # and the Mfuncs are written so that the subgenome dimensions and migration/exchange params are first
                        Mfirst = Mfunc5D_hex_a(zz[0], a_,b_,x,y, m34,m35,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],
                                               s3[11],s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25])
                        Mlast = Mfunc5D_hex_a(zz[N-1], a_,b_,x,y, m34,m35,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],
                                               s3[11],s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25]) 
                        for kk in range(0, N-1):
                            MInt[kk] = Mfunc5D_hex_a(zInt[kk], a_,b_,x,y, m34,m35,m31,m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],
                                               s3[11],s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25])
                        compute_delj(&dz[0], &MInt[0], &VInt[0], N, &delj[0], use_delj_trick)
                        compute_abc_nobc(&dz[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, N, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and a_==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu3 - Mfirst)*2/dz[0] 
                        if x==1 and y==1 and a_==1 and b_==1 and Mlast >= 0:
                            b[N-1] += -(-0.5/nu3 - Mlast)*2/dz[N-2]

                        for kk in range(0, N):
                            r[kk] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], N)
                        for kk in range(0, N):
                            phi[ii, jj, kk, ll, mm] = temp[kk]

    tridiag_free()

    return np.asarray(phi)

def implicit_5Da(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu4, double m41, double m42, double m43, double m45, double[:] s4, 
                        double dt, int use_delj_trick, int[:] ploidy4):
    """
    Implicit 5D integration function for a direction of 5D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz, aa, bb: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu4: Population size for pop4
    m41: Migration rate to pop4 from pop1
    m42: Migration rate to pop4 from pop2
    m43: Migration rate to pop4 from pop3
    m45: Migration rate to pop4 from pop5
    s4: vector of selection parameters for pop4
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy4: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in a direction
    """
    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int N = zz.shape[0] # number of grid points in z direction
    cdef int O = aa.shape[0] # number of grid points in a direction
    cdef int P = bb.shape[0] # number of grid points in b direction
    cdef int ii, jj, kk, ll, mm # loop indices
    cdef double x, y, z, b_ # single values from x, y, z and b grids
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] da = np.empty(O-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(O, dtype=np.float64) 
    cdef double[:] aInt = np.empty(O-1, dtype=np.float64)
    cdef double[:] delj = np.empty(O-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(O-1, dtype=np.float64)
    cdef double[:] V = np.empty(O, dtype=np.float64)
    cdef double[:] VInt = np.empty(O-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(O, dtype=np.float64)
    cdef double[:] b = np.empty(O, dtype=np.float64)
    cdef double[:] c = np.empty(O, dtype=np.float64)
    cdef double[:] r = np.empty(O, dtype=np.float64)
    cdef double[:] temp = np.empty(O, dtype=np.float64)
    ### specify ploidy of the a direction
    cdef int is_diploid = ploidy4[0]
    cdef int is_auto = ploidy4[1]
    cdef int is_alloa = ploidy4[2]
    cdef int is_allob = ploidy4[3]
    cdef int is_autohex = ploidy4[4]
    cdef int is_hex_tetra = ploidy4[5]    
    cdef int is_hex_dip = ploidy4[6]
    cdef int is_hex_b = ploidy4[8]

    # we only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order as the last three pops, 
    # hence no hex_a or hex_c models here

    # compute step size and intermediate values
    compute_dx(&aa[0], O, &da[0])
    compute_dfactor(&da[0], O, &dfactor[0])
    compute_xInt(&aa[0], O, &aInt[0])
    # dynamic allocation of memory for tridiag
    tridiag_malloc(O)

    # branch on ploidy
    if is_diploid:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, z, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        b_ = bb[mm]
            
                        Mfirst = Mfunc5D(aa[0], x,y,z,b_, m41,m42,m43,m45, s4[0],s4[1])
                        Mlast = Mfunc5D(aa[O-1], x,y,z,b_, m41,m42,m43,m45, s4[0],s4[1])  
                        for ll in range(0, O-1):
                            MInt[ll] = Mfunc5D(aInt[ll], x,y,z,b_, m41,m42,m43,m45, s4[0],s4[1])
                        compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                        compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                        if x==1 and y==1 and z==1 and b_==1 and Mlast >= 0:
                            b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                        for ll in range(0, O):
                            r[ll] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                        for ll in range(0, O):
                            phi[ii, jj, kk, ll, mm] = temp[ll]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc_tetra(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc_tetra(aInt[ll], nu4)
        # loop through x, y, z, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        b_ = bb[mm]
            
                        Mfirst = Mfunc5D_auto(aa[0], x,y,z,b_, m41,m42,m43,m45, s4[0],s4[1],s4[2],s4[3])
                        Mlast = Mfunc5D_auto(aa[O-1], x,y,z,b_, m41,m42,m43,m45, s4[0],s4[1],s4[2],s4[3])
                        for ll in range(0, O-1):
                            MInt[ll] = Mfunc5D_auto(aInt[ll], x,y,z,b_, m41,m42,m43,m45, s4[0],s4[1],s4[2],s4[3])
                        compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                        compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.25/nu4 - Mfirst)*2/da[0] 
                        if x==1 and y==1 and z==1 and b_==1 and Mlast >= 0:
                            b[O-1] += -(-0.25/nu4 - Mlast)*2/da[O-2]

                        for ll in range(0, O):
                            r[ll] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                        for ll in range(0, O):
                            phi[ii, jj, kk, ll, mm] = temp[ll]
        
    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, z, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        b_ = bb[mm]
                        ### Note: the order of migration params and grids being passed here is different 
                        # This is for consistency with the allo cases where the first two dimensions passed
                        # to Mfunc need to be the allo subgenomes and the subgenomes are always passed 
                        # to the integrator as the a and b dimensions.
                        Mfirst = Mfunc5D_allo_a(aa[0], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                        Mlast = Mfunc5D_allo_a(aa[O-1], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                        for ll in range(0, O-1):
                            MInt[ll] = Mfunc5D_allo_a(aInt[ll], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                        compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                        compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                        if x==1 and y==1 and z==1 and b_==1 and Mlast >= 0:
                            b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                        for ll in range(0, O):
                            r[ll] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                        for ll in range(0, O):
                            phi[ii, jj, kk, ll, mm] = temp[ll]

    elif is_allob:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, z, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        b_ = bb[mm]
                        # see note above about the order of the params passed to Mfuncs here
                        Mfirst = Mfunc5D_allo_a(aa[0], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                        Mlast = Mfunc5D_allo_a(aa[O-1], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                        for ll in range(0, O-1):
                            MInt[ll] = Mfunc5D_allo_a(aInt[ll], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7])
                        compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                        compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                        if x==1 and y==1 and z==1 and b_==1 and Mlast >= 0:
                            b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                        for ll in range(0, O):
                            r[ll] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                        for ll in range(0, O):
                            phi[ii, jj, kk, ll, mm] = temp[ll]
        
    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc_hex(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc_hex(aInt[ll], nu4)
        # loop through x, y, z, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        b_ = bb[mm]
            
                        Mfirst = Mfunc5D_autohex(aa[0], x,y,z,b_, m41,m42,m43,m45, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5])
                        Mlast = Mfunc5D_autohex(aa[O-1], x,y,z,b_, m41,m42,m43,m45, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5])
                        for ll in range(0, O-1):
                            MInt[ll] = Mfunc5D_autohex(aInt[ll], x,y,z,b_, m41,m42,m43,m45, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5])
                        compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                        compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and b_==0 and Mfirst <= 0:
                            b[0] += (1/(6*nu4) - Mfirst)*2/da[0]
                        if x==1 and y==1 and z==1 and b_==1 and Mlast >= 0:
                            b[O-1] += -(-1/(6*nu4) - Mlast)*2/da[O-2]

                        for ll in range(0, O):
                            r[ll] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                        for ll in range(0, O):
                            phi[ii, jj, kk, ll, mm] = temp[ll]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc_tetra(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc_tetra(aInt[ll], nu4)
        # loop through x, y, z, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        b_ = bb[mm]
                        # see note above about the order of the params passed to Mfuncs here
                        Mfirst = Mfunc5D_hex_tetra(aa[0], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                        Mlast = Mfunc5D_hex_tetra(aa[O-1], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                        for ll in range(0, O-1):
                            MInt[ll] = Mfunc5D_hex_tetra(aInt[ll], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                        compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                        compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.25/nu4 - Mfirst)*2/da[0] 
                        if x==1 and y==1 and z==1 and b_==1 and Mlast >= 0:
                            b[O-1] += -(-0.25/nu4 - Mlast)*2/da[O-2]

                        for ll in range(0, O):
                            r[ll] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                        for ll in range(0, O):
                            phi[ii, jj, kk, ll, mm] = temp[ll]

    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, z, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        b_ = bb[mm]
                        # see note above about the order of the params passed to Mfuncs here
                        Mfirst = Mfunc5D_hex_dip(aa[0], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                        Mlast = Mfunc5D_hex_dip(aa[O-1], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                        for ll in range(0, O-1):
                            MInt[ll] = Mfunc5D_hex_dip(aInt[ll], b_,x,y,z, m45,m41,m42,m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13])
                        compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                        compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                        if x==1 and y==1 and z==1 and b_==1 and Mlast >= 0:
                            b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                        for ll in range(0, O):
                            r[ll] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                        for ll in range(0, O):
                            phi[ii, jj, kk, ll, mm] = temp[ll]

    elif is_hex_b:
        # compute everything we can outside of the spatial loop
        for ll in range(0, O):
            V[ll] = Vfunc(aa[ll], nu4)
        for ll in range(0, O-1):
            VInt[ll] = Vfunc(aInt[ll], nu4)
        # loop through x, y, z, and b dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for mm in range(P):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        b_ = bb[mm]
                        # Note the order of migration params and grids being passed here is different
                        # This is to support a 2+2+2 hexaploid which must be passed as the last three pops/dimensions of phi
                        # and the Mfuncs are written so that the subgenome dimensions and migration/exchange params are first
                        Mfirst = Mfunc5D_hex_b(aa[0], z,b_,x,y, m43,m45,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],
                                               s4[11],s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25])
                        Mlast = Mfunc5D_hex_b(aa[O-1], z,b_,x,y, m43,m45,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],
                                               s4[11],s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25]) 
                        for ll in range(0, O-1):
                            MInt[ll] = Mfunc5D_hex_b(aInt[ll], z,b_,x,y, m43,m45,m41,m42, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],
                                               s4[11],s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25])
                        compute_delj(&da[0], &MInt[0], &VInt[0], O, &delj[0], use_delj_trick)
                        compute_abc_nobc(&da[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, O, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and b_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu4 - Mfirst)*2/da[0] 
                        if x==1 and y==1 and z==1 and b_==1 and Mlast >= 0:
                            b[O-1] += -(-0.5/nu4 - Mlast)*2/da[O-2]

                        for ll in range(0, O):
                            r[ll] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], O)
                        for ll in range(0, O):
                            phi[ii, jj, kk, ll, mm] = temp[ll]

    tridiag_free()

    return np.asarray(phi)

def implicit_5Db(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu5, double m51, double m52, double m53, double m54, double[:] s5, 
                        double dt, int use_delj_trick, int[:] ploidy5):
    """
    Implicit 5D integration function for b direction of 5D diffusion equation.
    
    Parameters:
    -----------
    phi : numpy array (float64)
        Population frequency array (modified in-place)
    xx, yy, zz, aa, bb: numpy arrays (float64)
        discrete numerical grids for spatial dimensions
    nu5: Population size for pop5
    m51: Migration rate to pop5 from pop1
    m52: Migration rate to pop5 from pop2
    m53: Migration rate to pop5 from pop3
    m54: Migration rate to pop5 from pop4
    s5: vector of selection parameters for pop5
    dt: Time step
    use_delj_trick: Whether to use delj trick (0 or 1)
    ploidy5: Vector of ploidy Booleans (0 or 1)
        [dip, auto, alloa, allob, autohex, 
        hex_tetra, hex_dip, hexa, hexb, hexc]

    Returns:
    --------
    phi : modified phi after integration in a direction
    """
    # define memory for non-array variables
    # Note: all of the arrays are preallocated for efficiency
    cdef int L = xx.shape[0] # number of grid points in x direction
    cdef int M = yy.shape[0] # number of grid points in y direction
    cdef int N = zz.shape[0] # number of grid points in z direction
    cdef int O = aa.shape[0] # number of grid points in a direction
    cdef int P = bb.shape[0] # number of grid points in b direction
    cdef int ii, jj, kk, ll, mm # loop indices
    cdef double x, y, z, a_ # single values from x, y, z and a grids
    
    # Create memory views for everything we need to compute
    ### grid spacings and integration points
    cdef double[:] db = np.empty(P-1, dtype=np.float64)
    cdef double[:] dfactor = np.empty(P, dtype=np.float64) 
    cdef double[:] bInt = np.empty(P-1, dtype=np.float64)
    cdef double[:] delj = np.empty(P-1, dtype=np.float64)
    ### population genetic functions
    cdef double Mfirst, Mlast
    cdef double[:] MInt = np.empty(P-1, dtype=np.float64)
    cdef double[:] V = np.empty(P, dtype=np.float64)
    cdef double[:] VInt = np.empty(P-1, dtype=np.float64)
    ### for the tridiagonal matrix solver
    cdef double[:] a = np.empty(P, dtype=np.float64)
    cdef double[:] b = np.empty(P, dtype=np.float64)
    cdef double[:] c = np.empty(P, dtype=np.float64)
    cdef double[:] r = np.empty(P, dtype=np.float64)
    cdef double[:] temp = np.empty(P, dtype=np.float64)
    ### specify ploidy of the a direction
    cdef int is_diploid = ploidy5[0]
    cdef int is_auto = ploidy5[1]
    cdef int is_alloa = ploidy5[2]
    cdef int is_allob = ploidy5[3]
    cdef int is_autohex = ploidy5[4]
    cdef int is_hex_tetra = ploidy5[5]
    cdef int is_hex_dip = ploidy5[6]
    cdef int is_hex_c = ploidy5[9]

    # we only support the a, b, c subgenomes of a 2+2+2 hexaploid being specified in that exact order as the last three pops, 
    # hence no hex_a or hex_b models here

    # compute step size and intermediate values
    compute_dx(&bb[0], P, &db[0])
    compute_dfactor(&db[0], P, &dfactor[0])
    compute_xInt(&bb[0], P, &bInt[0])
    # dynamic allocation of memory for tridiag
    tridiag_malloc(P)

    # branch on ploidy
    if is_diploid:
        # compute everything we can outside of the spatial loop
        for mm in range(0, P):
            V[mm] = Vfunc(bb[mm], nu5)
        for mm in range(0, P-1):
            VInt[mm] = Vfunc(bInt[mm], nu5)
        # loop through x, y, z, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for ll in range(O):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
            
                        Mfirst = Mfunc5D(bb[0], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1])
                        Mlast = Mfunc5D(bb[P-1], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1])  
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D(bInt[mm], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1])
                        compute_delj(&db[0], &MInt[0], &VInt[0], P, &delj[0], use_delj_trick)
                        compute_abc_nobc(&db[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, P, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and a_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu5 - Mfirst)*2/db[0] 
                        if x==1 and y==1 and z==1 and a_==1 and Mlast >= 0:
                            b[P-1] += -(-0.5/nu5 - Mlast)*2/db[P-2]

                        for mm in range(0, P):
                            r[mm] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], P)
                        for mm in range(0, P):
                            phi[ii, jj, kk, ll, mm] = temp[mm]
    
    elif is_auto:
        # compute everything we can outside of the spatial loop
        for mm in range(0, P):
            V[mm] = Vfunc_tetra(bb[mm], nu5)
        for mm in range(0, P-1):
            VInt[mm] = Vfunc_tetra(bInt[mm], nu5)
        # loop through x, y, z, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for ll in range(O):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
            
                        Mfirst = Mfunc5D_auto(bb[0], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3])
                        Mlast = Mfunc5D_auto(bb[P-1], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3])
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D_auto(bInt[mm], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3])
                        compute_delj(&db[0], &MInt[0], &VInt[0], P, &delj[0], use_delj_trick)
                        compute_abc_nobc(&db[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, P, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and a_==0 and Mfirst <= 0:
                            b[0] += (0.25/nu5 - Mfirst)*2/db[0] 
                        if x==1 and y==1 and z==1 and a_==1 and Mlast >= 0:
                            b[P-1] += -(-0.25/nu5 - Mlast)*2/db[P-2]

                        for mm in range(0, P):
                            r[mm] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], P)
                        for mm in range(0, P):
                            phi[ii, jj, kk, ll, mm] = temp[mm]
        
    elif is_alloa:
        # compute everything we can outside of the spatial loop
        for mm in range(0, P):
            V[mm] = Vfunc(bb[mm], nu5)
        for mm in range(0, P-1):
            VInt[mm] = Vfunc(bInt[mm], nu5)
        # loop through x, y, z, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for ll in range(O):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        ### Note: the order of migration params and grids being passed here is different 
                        # This is for consistency with the allo cases where the first two dimensions passed
                        # to Mfunc need to be the allo subgenomes and the subgenomes are always passed 
                        # to the integrator as the a and b dimensions.
                        Mfirst = Mfunc5D_allo_a(bb[0], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        Mlast = Mfunc5D_allo_a(bb[P-1], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D_allo_a(bInt[mm], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        compute_delj(&db[0], &MInt[0], &VInt[0], P, &delj[0], use_delj_trick)
                        compute_abc_nobc(&db[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, P, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and a_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu5 - Mfirst)*2/db[0] 
                        if x==1 and y==1 and z==1 and a_==1 and Mlast >= 0:
                            b[P-1] += -(-0.5/nu5 - Mlast)*2/db[P-2]

                        for mm in range(0, P):
                            r[mm] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], P)
                        for mm in range(0, P):
                            phi[ii, jj, kk, ll, mm] = temp[mm]

    elif is_allob:
        # compute everything we can outside of the spatial loop
        for mm in range(0, P):
            V[mm] = Vfunc(bb[mm], nu5)
        for mm in range(0, P-1):
            VInt[mm] = Vfunc(bInt[mm], nu5)
        # loop through x, y, z, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for ll in range(O):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        # see note above about the order of the params passed to Mfuncs here
                        Mfirst = Mfunc5D_allo_b(bb[0], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        Mlast = Mfunc5D_allo_b(bb[P-1], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D_allo_b(bInt[mm], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        compute_delj(&db[0], &MInt[0], &VInt[0], P, &delj[0], use_delj_trick)
                        compute_abc_nobc(&db[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, P, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and a_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu5 - Mfirst)*2/db[0] 
                        if x==1 and y==1 and z==1 and a_==1 and Mlast >= 0:
                            b[P-1] += -(-0.5/nu5 - Mlast)*2/db[P-2]

                        for mm in range(0, P):
                            r[mm] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], P)
                        for mm in range(0, P):
                            phi[ii, jj, kk, ll, mm] = temp[mm]
        
    elif is_autohex:
        # compute everything we can outside of the spatial loop
        for mm in range(0, P):
            V[mm] = Vfunc_hex(bb[mm], nu5)
        for mm in range(0, P-1):
            VInt[mm] = Vfunc_hex(bInt[mm], nu5)
        # loop through x, y, z, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for ll in range(O):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
            
                        Mfirst = Mfunc5D_autohex(bb[0], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5])
                        Mlast = Mfunc5D_autohex(bb[P-1], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5])
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D_autohex(bInt[mm], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5])
                        compute_delj(&db[0], &MInt[0], &VInt[0], P, &delj[0], use_delj_trick)
                        compute_abc_nobc(&db[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, P, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and a_==0 and Mfirst <= 0:
                            b[0] += (1/(6*nu5) - Mfirst)*2/db[0]
                        if x==1 and y==1 and z==1 and a_==1 and Mlast >= 0:
                            b[P-1] += -(-1/(6*nu5) - Mlast)*2/db[P-2]

                        for mm in range(0, P):
                            r[mm] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], P)
                        for mm in range(0, P):
                            phi[ii, jj, kk, ll, mm] = temp[mm]

    elif is_hex_tetra:
        # compute everything we can outside of the spatial loop
        for mm in range(0, P):
            V[mm] = Vfunc_tetra(bb[mm], nu5)
        for mm in range(0, P-1):
            VInt[mm] = Vfunc_tetra(bInt[mm], nu5)
        # loop through x, y, z, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for ll in range(O):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        # see note above about the order of the params passed to Mfuncs here
                        Mfirst = Mfunc5D_hex_tetra(bb[0], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],s5[12],s5[13])
                        Mlast = Mfunc5D_hex_tetra(bb[P-1], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],s5[12],s5[13])
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D_hex_tetra(bInt[mm], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],s5[12],s5[13])
                        compute_delj(&db[0], &MInt[0], &VInt[0], P, &delj[0], use_delj_trick)
                        compute_abc_nobc(&db[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, P, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and a_==0 and Mfirst <= 0:
                            b[0] += (0.25/nu5 - Mfirst)*2/db[0] 
                        if x==1 and y==1 and z==1 and a_==1 and Mlast >= 0:
                            b[P-1] += -(-0.25/nu5 - Mlast)*2/db[P-2]

                        for mm in range(0, P):
                            r[mm] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], P)
                        for mm in range(0, P):
                            phi[ii, jj, kk, ll, mm] = temp[mm]
    
    elif is_hex_dip:
        # compute everything we can outside of the spatial loop
        for mm in range(0, P):
            V[mm] = Vfunc(bb[mm], nu5)
        for mm in range(0, P-1):
            VInt[mm] = Vfunc(bInt[mm], nu5)
        # loop through x, y, z, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for ll in range(O):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        # see note above about the order of the params passed to Mfuncs here
                        Mfirst = Mfunc5D_hex_dip(bb[0], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],s5[12],s5[13])
                        Mlast = Mfunc5D_hex_dip(bb[P-1], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],s5[12],s5[13])
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D_hex_dip(bInt[mm], a_,x,y,z, m54,m51,m52,m53, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],s5[12],s5[13])
                        compute_delj(&db[0], &MInt[0], &VInt[0], P, &delj[0], use_delj_trick)
                        compute_abc_nobc(&db[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, P, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and a_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu5 - Mfirst)*2/db[0] 
                        if x==1 and y==1 and z==1 and a_==1 and Mlast >= 0:
                            b[P-1] += -(-0.5/nu5 - Mlast)*2/db[P-2]

                        for mm in range(0, P):
                            r[mm] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], P)
                        for mm in range(0, P):
                            phi[ii, jj, kk, ll, mm] = temp[mm]

    elif is_hex_c:
        # compute everything we can outside of the spatial loop
        for mm in range(0, P):
            V[mm] = Vfunc(bb[mm], nu5)
        for mm in range(0, P-1):
            VInt[mm] = Vfunc(bInt[mm], nu5)
        # loop through x, y, z, and a dimensions
        for ii in range(L):
            for jj in range(M):
                for kk in range(N):
                    for ll in range(O):
                        x = xx[ii]
                        y = yy[jj]
                        z = zz[kk]
                        a_ = aa[ll]
                        # Note the order of migration params and grids being passed here is different
                        # This is to support a 2+2+2 hexaploid which must be passed as the last three pops/dimensions of phi
                        # and the Mfuncs are written so that the subgenome dimensions and migration/exchange params are first
                        Mfirst = Mfunc5D_hex_c(bb[0], z,a_,x,y, m53,m54,m51,m52, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],
                                               s5[11],s5[12],s5[13],s5[14],s5[15],s5[16],s5[17],s5[18],s5[19],s5[20],s5[21],s5[22],s5[23],s5[24],s5[25])
                        Mlast = Mfunc5D_hex_c(bb[P-1], z,a_,x,y, m53,m54,m51,m52, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],
                                               s5[11],s5[12],s5[13],s5[14],s5[15],s5[16],s5[17],s5[18],s5[19],s5[20],s5[21],s5[22],s5[23],s5[24],s5[25]) 
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D_hex_c(bInt[mm], z,a_,x,y, m53,m54,m51,m52, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],
                                               s5[11],s5[12],s5[13],s5[14],s5[15],s5[16],s5[17],s5[18],s5[19],s5[20],s5[21],s5[22],s5[23],s5[24],s5[25])
                        compute_delj(&db[0], &MInt[0], &VInt[0], P, &delj[0], use_delj_trick)
                        compute_abc_nobc(&db[0], &dfactor[0], &delj[0], &MInt[0], &V[0], dt, P, &a[0], &b[0], &c[0])
                        if x==0 and y==0 and z==0 and a_==0 and Mfirst <= 0:
                            b[0] += (0.5/nu5 - Mfirst)*2/db[0] 
                        if x==1 and y==1 and z==1 and a_==1 and Mlast >= 0:
                            b[P-1] += -(-0.5/nu5 - Mlast)*2/db[P-2]

                        for mm in range(0, P):
                            r[mm] = phi[ii, jj, kk, ll, mm]/dt
                        tridiag_premalloc(&a[0], &b[0], &c[0], &r[0], &temp[0], P)
                        for mm in range(0, P):
                            phi[ii, jj, kk, ll, mm] = temp[mm]

    tridiag_free()

    return np.asarray(phi)

