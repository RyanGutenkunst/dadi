void implicit_1Dx(double *phi, double *xx,
    double nu, double gamma, double h, double beta, double dt, int L, 
    int use_delj_trick);
void implicit_2Dx(double *phi, double *xx, double *yy,
    double nu1, double m12, double gamma1, double h1,
    double dt, int L, int M, int use_delj_trick,
    int Mstart, int Mend);
void implicit_2Dy(double *phi, double *xx, double *yy,
    double nu2, double m21, double gamma2, double h2,
    double dt, int L, int M, int use_delj_trick, 
    int Lstart, int Lend);
void implicit_precalc_2Dx(double *phi, double *ax, double *bx, double *cx,
    double dt, int L, int M, int Mstart, int Mend);
void implicit_precalc_2Dy(double *phi, double *ay, double *by, double *cy,
    double dt, int L, int M, int Lstart, int Lend);
void implicit_3Dx(double *phi, double *xx, double *yy, double *zz,
    double nu1, double m12, double m13, double gamma1, double h1,
    double dt, int L, int M, int N, int use_delj_trick,
    int Mstart, int Mend);
void implicit_3Dy(double *phi, double *xx, double *yy, double *zz,
    double nu2, double m21, double m23, double gamma2, double h2,
    double dt, int L, int M, int N, int use_delj_trick,
    int Lstart, int Lend);
void implicit_3Dz(double *phi, double *xx, double *yy, double *zz,
    double nu3, double m31, double m32, double gamma3, double h3,
    double dt, int L, int M, int N, int use_delj_trick,
    int Lstart, int Lend);
void implicit_precalc_3Dx(double *phi, double *ax, double *bx, double *cx,
    double dt, int L, int M, int N, int Mstart, int Mend);
void implicit_precalc_3Dy(double *phi, double *ay, double *by, double *cy,
    double dt, int L, int M, int N, int Lstart, int Lend);
void implicit_precalc_3Dz(double *phi, double *az, double *bz, double *cz,
    double dt, int L, int M, int N, int Lstart, int Lend);
void implicit_4Dx(double *phi, double *xx, double *yy, double *zz, double *aa,
    double nu1, double m12, double m13, double m14, double gamma1, double h1,
    double dt, int L, int M, int N, int O, int use_delj_trick);
void implicit_4Dy(double *phi, double *xx, double *yy, double *zz, double *aa,
    double nu2, double m21, double m23, double m24, double gamma2, double h2,
    double dt, int L, int M, int N, int O, int use_delj_trick);
void implicit_4Dz(double *phi, double *xx, double *yy, double *zz, double *aa,
    double nu3, double m31, double m32, double m34, double gamma3, double h3,
    double dt, int L, int M, int N, int O, int use_delj_trick);
void implicit_4Da(double *phi, double *xx, double *yy, double *zz, double *aa,
    double nu4, double m41, double m42, double m43, double gamma4, double h4,
    double dt, int L, int M, int N, int O, int use_delj_trick);
void implicit_5Dx(double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
    double nu1, double m12, double m13, double m14, double m15, double gamma1, double h1,
    double dt, int L, int M, int N, int O, int P, int use_delj_trick);
void implicit_5Dy(double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
    double nu2, double m21, double m23, double m24, double m25, double gamma2, double h2,
    double dt, int L, int M, int N, int O, int P, int use_delj_trick);
void implicit_5Dz(double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
    double nu3, double m31, double m32, double m34, double m35, double gamma3, double h3,
    double dt, int L, int M, int N, int O, int P, int use_delj_trick);
void implicit_5Da(double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
    double nu4, double m41, double m42, double m43, double m45, double gamma4, double h4,
    double dt, int L, int M, int N, int O, int P, int use_delj_trick);
void implicit_5Db(double *phi, double *xx, double *yy, double *zz, double *aa, double *bb,
    double nu5, double m51, double m52, double m53, double m54, double gamma5, double h5,
    double dt, int L, int M, int N, int O, int P, int use_delj_trick);