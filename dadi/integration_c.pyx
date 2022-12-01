import numpy as np
cimport numpy as np

cdef extern from "integration_cython.h":
    # The c_func "func" syntax avoids name clashes between C and Python functions.
    # See https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html#resolving-naming-conflicts-c-name-specifications
    void c_implicit_1Dx "implicit_1Dx" (double *phi, double *xx,
        double nu, double gamma, double h, double beta, double dt, int L, 
        int use_delj_trick)
    void c_implicit_2Dx "implicit_2Dx" (double *phi, double *xx, double *yy,
        double nu1, double m12, double gamma1, double h1,
        double dt, int L, int M, int use_delj_trick,
        int Mstart, int Mend)
    void c_implicit_2Dy "implicit_2Dy" (double *phi, double *xx, double *yy,
        double nu2, double m21, double gamma2, double h2,
        double dt, int L, int M, int use_delj_trick, 
        int Lstart, int Lend)
    void c_implicit_precalc_2Dx "implicit_precalc_2Dx" (double *phi, double *ax, double *bx, double *cx,
        double dt, int L, int M, int Mstart, int Mend)
    void c_implicit_precalc_2Dy "implicit_precalc_2Dy" (double *phi, double *ay, double *by, double *cy,
        double dt, int L, int M, int Lstart, int Lend)
    void c_implicit_3Dx "implicit_3Dx" (double *phi, double *xx, double *yy, double *zz,
        double nu1, double m12, double m13, double gamma1, double h1,
        double dt, int L, int M, int N, int use_delj_trick,
        int Mstart, int Mend)
    void c_implicit_3Dy "implicit_3Dy" (double *phi, double *xx, double *yy, double *zz,
        double nu2, double m21, double m23, double gamma2, double h2,
        double dt, int L, int M, int N, int use_delj_trick,
        int Lstart, int Lend)
    void c_implicit_3Dz "implicit_3Dz" (double *phi, double *xx, double *yy, double *zz,
        double nu3, double m31, double m32, double gamma3, double h3,
        double dt, int L, int M, int N, int use_delj_trick,
        int Lstart, int Lend)
    void c_implicit_precalc_3Dx "implicit_precalc_3Dx" (double *phi, double *ax, double *bx, double *cx,
        double dt, int L, int M, int N, int Mstart, int Mend)
    void c_implicit_precalc_3Dy "implicit_precalc_3Dy" (double *phi, double *ay, double *by, double *cy,
        double dt, int L, int M, int N, int Lstart, int Lend)
    void c_implicit_precalc_3Dz "implicit_precalc_3Dz" (double *phi, double *az, double *bz, double *cz,
        double dt, int L, int M, int N, int Lstart, int Lend)
    void c_implicit_4Dx "implicit_4Dx" (double *phi, double *xx, double *yy, double *zz, double *aa,
        double nu1, double m12, double m13, double m14, double gamma1, double h1,
        double dt, int L, int M, int N, int O, int use_delj_trick)
    void c_implicit_4Dy "implicit_4Dy" (double *phi, double *xx, double *yy, double *zz, double *aa,
        double nu2, double m21, double m23, double m24, double gamma2, double h2,
        double dt, int L, int M, int N, int O, int use_delj_trick)
    void c_implicit_4Dz "implicit_4Dz" (double *phi, double *xx, double *yy, double *zz, double *aa,
        double nu3, double m31, double m32, double m34, double gamma3, double h3,
        double dt, int L, int M, int N, int O, int use_delj_trick)
    void c_implicit_4Da "implicit_4Da" (double *phi, double *xx, double *yy, double *zz, double *aa,
        double nu4, double m41, double m42, double m43, double gamma4, double h4,
        double dt, int L, int M, int N, int O, int use_delj_trick)
    void c_implicit_5Dx "implicit_5Dx" (double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
        double nu1, double m12, double m13, double m14, double m15, double gamma1, double h1,
        double dt, int L, int M, int N, int O, int P, int use_delj_trick)
    void c_implicit_5Dy "implicit_5Dy" (double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
        double nu2, double m21, double m23, double m24, double m25, double gamma2, double h2,
        double dt, int L, int M, int N, int O, int P, int use_delj_trick)
    void c_implicit_5Dz "implicit_5Dz" (double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
        double nu3, double m31, double m32, double m34, double m35, double gamma3, double h3,
        double dt, int L, int M, int N, int O, int P, int use_delj_trick)
    void c_implicit_5Da "implicit_5Da" (double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
        double nu4, double m41, double m42, double m43, double m45, double gamma4, double h4,
        double dt, int L, int M, int N, int O, int P, int use_delj_trick)
    void c_implicit_5Db "implicit_5Db" (double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
        double nu5, double m51, double m52, double m53, double m54, double gamma5, double h5,
        double dt, int L, int M, int N, int O, int P, int use_delj_trick)

def implicit_1Dx(np.ndarray phi, np.ndarray xx,
        nu, gamma, h, beta, dt, use_delj_trick):
    c_implicit_1Dx(<double*> phi.data, <double*> xx.data,
        nu, gamma, h, beta, dt, phi.shape[0], use_delj_trick)
    return phi

def implicit_2Dx(np.ndarray phi, np.ndarray xx, np.ndarray yy,
        nu1, m12, gamma1, h1, dt, use_delj_trick):
    c_implicit_2Dx(<double*> phi.data, <double*> xx.data, <double*> yy.data, 
            nu1, m12, gamma1, h1, dt, phi.shape[0], phi.shape[1], use_delj_trick, 
            0, phi.shape[0])
    return phi
def implicit_2Dy(np.ndarray phi, np.ndarray xx, np.ndarray yy,
        nu2, m21, gamma2, h2, dt, use_delj_trick):
    c_implicit_2Dy(<double*> phi.data, <double*> xx.data, <double*> yy.data,
            nu2, m21, gamma2, h2, dt, phi.shape[0], phi.shape[1], use_delj_trick, 
            0, phi.shape[1])
    return phi
def implicit_precalc_2Dx(np.ndarray phi, np.ndarray ax, np.ndarray bx, 
        np.ndarray cx, dt):
    c_implicit_precalc_2Dx(<double*> phi.data, <double*> ax.data, <double*> bx.data, 
            <double*> cx.data, dt, phi.shape[0], phi.shape[1], 0, phi.shape[0])
    return phi
def implicit_precalc_2Dy(np.ndarray phi, np.ndarray ay, np.ndarray by, 
        np.ndarray cy, dt):
    c_implicit_precalc_2Dy(<double*> phi.data, <double*> ay.data, <double*> by.data, 
            <double*> cy.data, dt, phi.shape[0], phi.shape[1], 0, phi.shape[1])
    return phi

def implicit_3Dx(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz,
        nu1,  m12,  m13,  gamma1,  h1, dt, use_delj_trick):
    c_implicit_3Dx(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data,
        nu1,  m12,  m13,  gamma1,  h1, dt, 
        phi.shape[0], phi.shape[1], phi.shape[2], use_delj_trick,
        0, phi.shape[1])
    return phi
def implicit_3Dy(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz,
        nu2, m21, m23, gamma2, h2, dt, int use_delj_trick):
    c_implicit_3Dy(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data,
        nu2, m21, m23, gamma2, h2, dt, 
        phi.shape[0], phi.shape[1], phi.shape[2], use_delj_trick,
        0, phi.shape[1])
    return phi
def implicit_3Dz(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz,
        nu3, m31, m32, gamma3, h3, dt, use_delj_trick):
    c_implicit_3Dz(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data,
        nu3, m31, m32, gamma3, h3, dt, 
        phi.shape[0], phi.shape[1], phi.shape[2], use_delj_trick,
        0, phi.shape[2])
    return phi
def implicit_precalc_3Dx(np.ndarray phi, np.ndarray ax, np.ndarray bx, np.ndarray cx,
        dt):
    c_implicit_precalc_3Dx(<double*> phi.data, <double*> ax.data, <double*> bx.data, <double*> cx.data,
        dt, phi.shape[0], phi.shape[1], phi.shape[2], 0, phi.shape[0])
    return phi
def implicit_precalc_3Dy(np.ndarray phi, np.ndarray ay, np.ndarray by, np.ndarray cy,
        dt):
    c_implicit_precalc_3Dy(<double*> phi.data, <double*> ay.data, <double*> by.data, <double*> cy.data,
        dt, phi.shape[0], phi.shape[1], phi.shape[2], 0, phi.shape[1])
    return phi
def implicit_precalc_3Dz(np.ndarray phi, np.ndarray az, np.ndarray bz, np.ndarray cz,
        dt):
    c_implicit_precalc_3Dz(<double*> phi.data, <double*> az.data, <double*> bz.data, <double*> cz.data,
        dt, phi.shape[0], phi.shape[1], phi.shape[2], 0, phi.shape[2])
    return phi

def implicit_4Dx(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz, np.ndarray aa,
        nu1, m12, m13, m14, gamma1, h1,
        dt, use_delj_trick):
    c_implicit_4Dx(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data, <double*> aa.data,
        nu1, m12, m13, m14, gamma1, h1, dt, 
        phi.shape[0], phi.shape[1], phi.shape[2], phi.shape[3], use_delj_trick)
    return phi
def implicit_4Dy(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz, np.ndarray aa,
        nu2, m21, m23, m24, gamma2, h2,
        dt, use_delj_trick):
    c_implicit_4Dy(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data, <double*> aa.data,
        nu2, m21, m23, m24, gamma2, h2, dt, 
        phi.shape[0], phi.shape[1], phi.shape[2], phi.shape[3], use_delj_trick)
    return phi
def implicit_4Dz(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz, np.ndarray aa,
        nu3, m31, m32, m34, gamma3, h3,
        dt, use_delj_trick):
    c_implicit_4Dz(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data, <double*> aa.data,
        nu3, m31, m32, m34, gamma3, h3, dt,
        phi.shape[0], phi.shape[1], phi.shape[2], phi.shape[3], use_delj_trick)
    return phi
def implicit_4Da(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz, np.ndarray aa,
        nu4, m41, m42, m43, gamma4, h4,
        dt, use_delj_trick):
    c_implicit_4Da(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data, <double*> aa.data,
        nu4, m41, m42, m43, gamma4, h4, dt,
        phi.shape[0], phi.shape[1], phi.shape[2], phi.shape[3], use_delj_trick)
    return phi

def implicit_5Dx(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz, np.ndarray aa, np.ndarray bb,
        nu1, m12, m13, m14, m15, gamma1, h1,
        dt, use_delj_trick):
    c_implicit_5Dx(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data, <double*> aa.data, <double*> bb.data,
        nu1, m12, m13, m14, m15, gamma1, h1, dt,
        phi.shape[0], phi.shape[1], phi.shape[2], phi.shape[3], phi.shape[4], use_delj_trick)
    return phi
def implicit_5Dy(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz, np.ndarray aa, np.ndarray bb,
        nu2, m21, m23, m24, m25, gamma2, h2,
        dt, use_delj_trick):
    c_implicit_5Dy(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data, <double*> aa.data, <double*> bb.data,
        nu2, m21, m23, m24, m25, gamma2, h2, dt,
        phi.shape[0], phi.shape[1], phi.shape[2], phi.shape[3], phi.shape[4], use_delj_trick)
    return phi
def implicit_5Dz(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz, np.ndarray aa, np.ndarray bb,
        nu3, m31, m32, m34, m35, gamma3, h3,
        dt, use_delj_trick):
    c_implicit_5Dz(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data, <double*> aa.data, <double*> bb.data,
        nu3, m31, m32, m34, m35, gamma3, h3, dt,
        phi.shape[0], phi.shape[1], phi.shape[2], phi.shape[3], phi.shape[4], use_delj_trick)
    return phi
def implicit_5Da(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz, np.ndarray aa, np.ndarray bb,
        nu4, m41, m42, m43, m45, gamma4, h4,
        dt, int use_delj_trick):
    c_implicit_5Da(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data, <double*> aa.data, <double*> bb.data,
        nu4, m41, m42, m43, m45, gamma4, h4, dt,
        phi.shape[0], phi.shape[1], phi.shape[2], phi.shape[3], phi.shape[4], use_delj_trick)
    return phi
def implicit_5Db(np.ndarray phi, np.ndarray xx, np.ndarray yy, np.ndarray zz, np.ndarray aa, np.ndarray bb,
        nu5, m51, m52, m53, m54, gamma5, h5,
        dt, int use_delj_trick):
    c_implicit_5Db(<double*> phi.data, <double*> xx.data, <double*> yy.data, <double*> zz.data, <double*> aa.data, <double*> bb.data,
        nu5, m51, m52, m53, m54, gamma5, h5, dt,
        phi.shape[0], phi.shape[1], phi.shape[2], phi.shape[3], phi.shape[4], use_delj_trick)
    return phi