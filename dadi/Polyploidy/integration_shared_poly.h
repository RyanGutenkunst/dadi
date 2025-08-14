/* Population genetic functions modified for tetraploids
*/

// shared new delta_j function
void compute_delj(double *dx, double *MInt, double *VInt, double *VIntprime,
        int N, double *delj, int use_delj_trick);
double Vfunc_prime(double x, double nu);
double Vfunc_auto_prime(double x, double nu);

// First, for the autos
double Vfunc_auto(double x, double nu);
double Mfunc1D_auto(double x, double gam1, double gam2, double gam3, double gam4);
double Mfunc2D_auto(double x, double y, double mxy, double gam1, double gam2, double gam3, double gam4);
double Mfunc3D_auto(double x, double y, double z, double mxy, double mxz, double gam1, double gam2, double gam3, double gam4);
double Mfunc4D_auto(double x, double y, double z, double a, double mxy, double mxz, double mxa, double gam1, double gam2, double gam3, double gam4);

// Then for allotetraploids
double Mfunc2D_allo_a(double x, double y, double mxy, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22);
double Mfunc2D_allo_b(double x, double y, double mxy, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22);
double Mfunc3D_allo_a(double x, double y, double z, double mxy, double mxz, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22);
double Mfunc3D_allo_b(double x, double y, double z, double mxy, double mxz, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22);
double Mfunc4D_allo_a(double x, double y, double z, double a, double mxy, double mxz, double mxa, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22);
double Mfunc4D_allo_b(double x, double y, double z, double a, double mxy, double mxz, double mxa, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22);