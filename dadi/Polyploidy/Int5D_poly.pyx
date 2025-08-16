#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np

# =========================================================
# SHARED AND DIPLOID C FUNCTIONS
# =========================================================
cdef extern from "integration_shared.h":
    double Vfunc(double x, double nu)
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
    double Mfunc5D_auto(double x, double y, double z, double a, double b,
                        double mxy, double mxz, double mxa, double mxb,
                        double gam1, double gam2, double gam3, double gam4)
    double Mfunc5D_allo_a(double x, double y, double z, double a, double b,
                          double exy, double mxz, double mxa, double mxb,
                          double g01, double g02, double g10, double g11, 
                          double g12, double g20, double g21, double g22)
    double Mfunc5D_allo_b(double x, double y, double z, double a, double b,
                          double exy, double mxz, double mxa, double mxb,
                          double g01, double g02, double g10, double g11, 
                          double g12, double g20, double g21, double g22)
    
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
# CYTHON 4D INTEGRATION FUNCTIONS - VARIABLE PARAMS
# =========================================================
cdef void c_implicit_5Dx(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu1, double m12, double m13, double m14, double m15, double[:] s1, 
                        double dt, int use_delj_trick, int[:] ploidy1):
    
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
    
    tridiag_free()
            
cdef void c_implicit_5Dy(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu2, double m21, double m23, double m24, double m25, double[:] s2, 
                        double dt, int use_delj_trick, int[:] ploidy2):
    
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
    tridiag_free()

cdef void c_implicit_5Dz(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu3, double m31, double m32, double m34, double m35, double[:] s3, 
                        double dt, int use_delj_trick, int[:] ploidy3):
    
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
    # note: we don't support alloa and allob as being the third (middle) dimension of the phi array in 5D

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

    tridiag_free()

cdef void c_implicit_5Da(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu4, double m41, double m42, double m43, double m45, double[:] s4, 
                        double dt, int use_delj_trick, int[:] ploidy4):
    
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
        
    tridiag_free()

cdef void c_implicit_5Db(double[:,:,:,:,:] phi, double[:] xx, double[:] yy, double[:] zz, double[:] aa, double[:] bb,
                        double nu5, double m51, double m52, double m53, double m54, double[:] s5, 
                        double dt, int use_delj_trick, int[:] ploidy5):
    
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
            
                        Mfirst = Mfunc5D_allo_a(bb[0], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        Mlast = Mfunc5D_allo_a(bb[P-1], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D_allo_a(bInt[mm], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
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
            
                        Mfirst = Mfunc5D_allo_b(bb[0], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        Mlast = Mfunc5D_allo_b(bb[P-1], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
                        for mm in range(0, P-1):
                            MInt[mm] = Mfunc5D_allo_b(bInt[mm], x,y,z,a_, m51,m52,m53,m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7])
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

### ==========================================================================
### MAKE THE INTEGRATION FUNCTIONS CALLABLE FROM PYTHON
### ==========================================================================

def implicit_5Dx(np.ndarray[double, ndim=5] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 np.ndarray[double, ndim=1] zz, 
                 np.ndarray[double, ndim=1] aa,
                 np.ndarray[double, ndim=1] bb,
                 double nu1, 
                 double m12, 
                 double m13, 
                 double m14,
                 double m15,
                 np.ndarray[double, ndim=1] s1,
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy1):
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
        [dip, auto, alloa, allob]

    Returns:
    --------
    phi : modified phi after integration in x direction
    """
    # Call the cdef function with memory views
    c_implicit_5Dx(phi, xx, yy, zz, aa, bb, nu1, m12, m13, m14, m15, s1, dt, use_delj_trick, ploidy1)
    return phi

def implicit_5Dy(np.ndarray[double, ndim=5] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 np.ndarray[double, ndim=1] zz, 
                 np.ndarray[double, ndim=1] aa,
                 np.ndarray[double, ndim=1] bb,
                 double nu2, 
                 double m21, 
                 double m23, 
                 double m24,
                 double m25,
                 np.ndarray[double, ndim=1] s2,
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy2):
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
        [dip, auto, alloa, allob]

    Returns:
    --------
    phi : modified phi after integration in y direction
    """
    # Call the cdef function with memory views
    c_implicit_5Dy(phi, xx, yy, zz, aa, bb, nu2, m21, m23, m24, m25, s2, dt, use_delj_trick, ploidy2)
    return phi

def implicit_5Dz(np.ndarray[double, ndim=5] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 np.ndarray[double, ndim=1] zz, 
                 np.ndarray[double, ndim=1] aa,
                 np.ndarray[double, ndim=1] bb,
                 double nu3, 
                 double m31, 
                 double m32, 
                 double m34,
                 double m35,
                 np.ndarray[double, ndim=1] s3,
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy3):
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
        [dip, auto, alloa, allob]

    Returns:
    --------
    phi : modified phi after integration in z direction
    """
    # Call the cdef function with memory views
    c_implicit_5Dz(phi, xx, yy, zz, aa, bb, nu3, m31, m32, m34, m35, s3, dt, use_delj_trick, ploidy3)
    return phi

def implicit_5Da(np.ndarray[double, ndim=5] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 np.ndarray[double, ndim=1] zz, 
                 np.ndarray[double, ndim=1] aa,
                 np.ndarray[double, ndim=1] bb,
                 double nu4, 
                 double m41, 
                 double m42, 
                 double m43,
                 double m45,
                 np.ndarray[double, ndim=1] s4,
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy4):
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
        [dip, auto, alloa, allob]

    Returns:
    --------
    phi : modified phi after integration in a direction
    """
    # Call the cdef function with memory views
    c_implicit_5Da(phi, xx, yy, zz, aa, bb, nu4, m41, m42, m43, m45, s4, dt, use_delj_trick, ploidy4)
    return phi

def implicit_5Db(np.ndarray[double, ndim=5] phi, 
                 np.ndarray[double, ndim=1] xx, 
                 np.ndarray[double, ndim=1] yy, 
                 np.ndarray[double, ndim=1] zz, 
                 np.ndarray[double, ndim=1] aa,
                 np.ndarray[double, ndim=1] bb,
                 double nu5, 
                 double m51, 
                 double m52, 
                 double m53,
                 double m54,
                 np.ndarray[double, ndim=1] s5,
                 double dt, 
                 int use_delj_trick,  
                 np.ndarray[int, ndim=1] ploidy5):
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
        [dip, auto, alloa, allob]

    Returns:
    --------
    phi : modified phi after integration in a direction
    """
    # Call the cdef function with memory views
    c_implicit_5Db(phi, xx, yy, zz, aa, bb, nu5, m51, m52, m53, m54, s5, dt, use_delj_trick, ploidy5)
    return phi
