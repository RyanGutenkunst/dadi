#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as cnp

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

# x direction here

# y direction here