#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np

# =========================================================
# SHARED AND DIPLOID C FUNCTIONS
# =========================================================
cdef extern from "integration_shared.h":
    double Vfunc(double x, double nu)
    double Mfunc3D(double x, double y, double z, double mxy, double mxz, double gamma, double h)
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
    double Mfunc3D_auto(double x, double y, double z, double mxy, double mxz, double gam1, double gam2, double gam3, double gam4)
    double Mfunc3D_allo_a(double x, double y, double z, double mxy, double mxz, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22)
    double Mfunc3D_allo_b(double x, double y, double z, double mxy, double mxz, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22)

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
# CYTHON 3D INTEGRATION FUNCTIONS - VARIABLE PARAMS
# =========================================================
cdef void c_implicit_3Dx(double[:,:,:] phi, double[:] xx, double[:] yy, double[:] zz,
                        double nu1, double m12, double m13, double[:] s1, 
                        double dt, int use_delj_trick, int[:] ploidy1):
    
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
    cdef int is_alloa = ploidy1[2]
    cdef int is_allob = ploidy1[3]

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
        # loop through y and z dimensions
        for jj in range(M):
            for kk in range(N):
                y = yy[jj]
                z = zz[kk]

                Mfirst = Mfunc3D(xx[0], y, z, m12, m13, s1[0], s1[1])
                Mlast = Mfunc3D(xx[L-1], y, z, m12, m13, s1[0], s1[1])
                for ii in range(0, L-1):
                    MInt[ii] = Mfunc3D(xInt[ii], y, z, m12, m13, s1[0], s1[1]) 
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

                Mfirst = Mfunc3D_auto(xx[0], y, z, m12, m13, s1[0], s1[1], s1[2], s1[3])
                Mlast = Mfunc3D_auto(xx[L-1], y, z, m12, m13, s1[0], s1[1], s1[2], s1[3])
                for ii in range(0, L-1):
                    MInt[ii] = Mfunc3D_auto(xInt[ii], y, z, m12, m13, s1[0], s1[1], s1[2], s1[3]) 
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

    ### TODO: this should never be called, because we require alloa and allob to be the y and z dimensions (i.e. the last two populations specified)
    elif is_alloa:
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

                Mfirst = Mfunc3D_allo_a(xx[0], y, z, m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                Mlast = Mfunc3D_allo_a(xx[L-1], y, z, m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                for ii in range(0, L-1):
                    MInt[ii] = Mfunc3D_allo_a(xInt[ii], y, z, m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
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
    ### TODO: see above; this should never be called
    elif is_allob:
        # compute everything we can outside of the spatial loop
        for ii in range(0, L):
            V[ii] = Vfunc(xx[ii], nu1)
        for ii in range(0, L-1):
            VInt[ii] = Vfunc(xInt[ii], nu1)
        for jj in range(M):
            for kk in range(N):
                y = yy[jj]
                z = zz[kk]

                Mfirst = Mfunc3D_allo_b(xx[0], y, z, m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                Mlast = Mfunc3D_allo_b(xx[L-1], y, z, m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
                for ii in range(0, L-1):
                    MInt[ii] = Mfunc3D_allo_b(xInt[ii], y, z, m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7])
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
            
cdef void c_implicit_3Dy(double[:,:,:] phi, double[:] xx, double[:] yy, double[:] zz,
                        double nu2, double m21, double m23, double[:] s2, 
                        double dt, int use_delj_trick, int[:] ploidy2):
    
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
        # loop through x and z dimensions
        for ii in range(L):
            for kk in range(N):
                x = xx[ii]
                z = zz[kk]
                ### Note: the order of the params being passed here is different from 
                # Ryan's original code. This is for consistency with the allo cases where
                # the first two dimensions passed to Mfunc need to be the allo subgenomes 
                # and the subgenomes are always passed as y and z.
                Mfirst = Mfunc3D(yy[0], z, x, m23, m21, s2[0], s2[1])
                Mlast = Mfunc3D(yy[M-1], z, x, m23, m21, s2[0], s2[1])  
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D(yInt[jj], z, x, m23, m21, s2[0], s2[1])
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
                # see note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_auto(yy[0], z, x, m23, m21, s2[0],s2[1],s2[2],s2[3])
                Mlast = Mfunc3D_auto(yy[M-1], z, x, m23, m21, s2[0],s2[1],s2[2],s2[3])
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_auto(yInt[jj], z, x, m23, m21, s2[0],s2[1],s2[2],s2[3])
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
                # see note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_allo_a(yy[0], z, x, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                Mlast = Mfunc3D_allo_a(yy[M-1], z, x, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_allo_a(yInt[jj], z, x, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
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
                Mfirst = Mfunc3D_allo_b(yy[0], z, x, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                Mlast = Mfunc3D_allo_b(yy[M-1], z, x, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
                for jj in range(0, M-1):
                    MInt[jj] = Mfunc3D_allo_b(yInt[jj], z, x, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7])
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

cdef void c_implicit_3Dz(double[:,:,:] phi, double[:] xx, double[:] yy, double[:] zz,
                        double nu3, double m31, double m32, double[:] s3, 
                        double dt, int use_delj_trick, int[:] ploidy3):
    
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
    ### specify ploidy of the y direction
    cdef int is_diploid = ploidy3[0]
    cdef int is_auto = ploidy3[1]
    cdef int is_alloa = ploidy3[2]
    cdef int is_allob = ploidy3[3]

    # compute the y step size and intermediate y values
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
                ### Note: the order of the params being passed here is different from 
                # Ryan's original code. This is for consistency with the allo cases where
                # the first two dimensions passed to Mfunc need to be the allo subgenomes 
                # and the subgenomes are always passed as y and z.
                Mfirst = Mfunc3D(zz[0], y, x, m32, m31, s3[0], s3[1])
                Mlast = Mfunc3D(zz[N-1], y, x, m32, m31, s3[0], s3[1])  
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D(zInt[kk], y, x, m32, m31, s3[0], s3[1])
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
                # See note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_auto(zz[0], y, x, m32, m31, s3[0],s3[1],s3[2],s3[3])
                Mlast = Mfunc3D_auto(zz[N-1], y, x, m32, m31, s3[0],s3[1],s3[2],s3[3])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_auto(zInt[kk], y, x, m32, m31, s3[0],s3[1],s3[2],s3[3])
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
                # See note above about the order of the params passed to Mfuncs here
                Mfirst = Mfunc3D_allo_a(zz[0], y, x, m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                Mlast = Mfunc3D_allo_a(zz[N-1], y, x, m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_allo_a(zInt[kk], y, x, m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
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
                Mfirst = Mfunc3D_allo_b(zz[0], y, x, m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                Mlast = Mfunc3D_allo_b(zz[N-1], y, x, m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
                for kk in range(0, N-1):
                    MInt[kk] = Mfunc3D_allo_b(zInt[kk], y, x, m32, m31, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7])
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

### ==========================================================================
### CYTHON 3D INTEGRATION FUNCTIONS - CONSTANT PARAMS
### ==========================================================================

cdef void c_implicit_precalc_3Dx(double[:,:,:] phi, double[:,:,:] ax, double[:,:,:] bx,
                                 double[:,:,:] cx, double dt):
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

cdef void c_implicit_precalc_3Dy(double[:,:,:] phi, double[:,:,:] ay, double[:,:,:] by,
                                 double[:,:,:] cy, double dt):
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

cdef void c_implicit_precalc_3Dz(double[:,:,:] phi, double[:,:,:] az, double[:,:,:] bz,
                                 double[:,:,:] cz, double dt):
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

### ==========================================================================
### MAKE THE INTEGRATION FUNCTIONS CALLABLE FROM PYTHON
### ==========================================================================

def implicit_3Dx(np.ndarray[double, ndim=3] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 np.ndarray[double, ndim=1] zz, 
                 double nu1, 
                 double m12, 
                 double m13, 
                 np.ndarray[double, ndim=1] s1,
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy1):
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
        [dip, auto, alloa, allob]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    # Call the cdef function with memory views
    c_implicit_3Dx(phi, xx, yy, zz, nu1, m12, m13, s1, dt, use_delj_trick, ploidy1)
    return phi

def implicit_3Dy(np.ndarray[double, ndim=3] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 np.ndarray[double, ndim=1] zz, 
                 double nu2, 
                 double m21, 
                 double m23, 
                 np.ndarray[double, ndim=1] s2,
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy2):
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
        [dip, auto, alloa, allob]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    # Call the cdef function with memory views
    c_implicit_3Dy(phi, xx, yy, zz, nu2, m21, m23, s2, dt, use_delj_trick, ploidy2)
    return phi

def implicit_3Dz(np.ndarray[double, ndim=3] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 np.ndarray[double, ndim=1] zz, 
                 double nu3, 
                 double m31, 
                 double m32, 
                 np.ndarray[double, ndim=1] s3,
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy3):
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
        [dip, auto, alloa, allob]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    # Call the cdef function with memory views
    c_implicit_3Dz(phi, xx, yy, zz, nu3, m31, m32, s3, dt, use_delj_trick, ploidy3)
    return phi

def implicit_precalc_3Dx(np.ndarray[double, ndim=3] phi, 
                         np.ndarray[double, ndim=3] ax, 
                         np.ndarray[double, ndim=3] bx, 
                         np.ndarray[double, ndim=3] cx, 
                         double dt):
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
    # Call the cdef function with memory views
    c_implicit_precalc_3Dx(phi, ax, bx, cx, dt)
    return phi

def implicit_precalc_3Dy(np.ndarray[double, ndim=3] phi, 
                         np.ndarray[double, ndim=3] ay, 
                         np.ndarray[double, ndim=3] by, 
                         np.ndarray[double, ndim=3] cy, 
                         double dt):
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
    # Call the cdef function with memory views
    c_implicit_precalc_3Dy(phi, ay, by, cy, dt)
    return phi

def implicit_precalc_3Dz(np.ndarray[double, ndim=3] phi, 
                         np.ndarray[double, ndim=3] az, 
                         np.ndarray[double, ndim=3] bz, 
                         np.ndarray[double, ndim=3] cz, 
                         double dt):
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
    # Call the cdef function with memory views
    c_implicit_precalc_3Dz(phi, az, bz, cz, dt)
    return phi
