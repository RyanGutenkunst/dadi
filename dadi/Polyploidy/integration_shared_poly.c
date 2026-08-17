#include <stdio.h>

/*
* AUTOTETRAPLOIDS POP GEN FUNCTIONS
*/
double Vfunc_tetra(double x, double nu){
    return 1./(2.*nu) * x*(1.-x);
}
// Use Horner's method here to evaluate the polynomial is an easy optimization
double Mfunc1D_auto(double x, double gam1, double gam2, double gam3, double gam4){
    double poly = (((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                    (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                    (-6.*gam1 + 3.*gam2)) * x +
                    gam1;
    return x * (1. - x) * 2. * poly;
}
double Mfunc2D_auto(double x, double y, double mxy, 
                    double gam1, double gam2, double gam3, double gam4){
    double poly = (((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                    (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                    (-6.*gam1 + 3.*gam2)) * x +
                    gam1;
    return mxy * (y-x) + x * (1. - x) * 2. * poly;
}
double Mfunc3D_auto(double x, double y, double z, double mxy, double mxz, 
                    double gam1, double gam2, double gam3, double gam4){
    double poly = (((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                    (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                    (-6.*gam1 + 3.*gam2)) * x +
                    gam1;
    return mxy * (y-x) + mxz * (z-x) + x * (1. - x) * 2. * poly;
}
double Mfunc4D_auto(double x, double y, double z, double a,
                    double mxy, double mxz, double mxa,
                    double gam1, double gam2, double gam3, double gam4){
    double poly = (((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                    (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                    (-6.*gam1 + 3.*gam2)) * x +
                    gam1;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
}
double Mfunc5D_auto(double x, double y, double z, double a, double b,
                    double mxy, double mxz, double mxa, double mxb,
                    double gam1, double gam2, double gam3, double gam4){
    double poly = (((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                    (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                    (-6.*gam1 + 3.*gam2)) * x +
                    gam1;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
}

/*
* ALLOTETRAPLOIDS POP GEN FUNCTIONS 
* Note that the variance is the same as for diploids.
* 
* Also, note that it is non-trivial to write a single function for
* both subgenomes, so I wrote one function for each subgenome.
* I think you could swap x with y and swap g01 with g10, g20 with g02, etc., 
* but that's a mess to deal with in the Cython integration code.
*
* In each case, x is treated as the *current* subgenome frequency
* and y is the *corresponding/opposite* subgenome frequency.
* Also note that gij refers to gamma_ij (not a gamete frequency!)
*/
double Mfunc2D_allo_a(double x, double y, double exy, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22){
    /*
    * x is x_a, y is x_b
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g10 + (-2.*g10 + g20)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g01 - g02 + g10 -2.*g11 + g12)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g01 + 4.*g10 -4.*g11 -2.*g20 +2.*g21)*xy;
    return exy * (y-x) + x * (1. - x) * 2. * poly;
}
double Mfunc2D_allo_b(double x, double y, double exy, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22){
    /*
    * x is x_b, y is x_a
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g01 + (-2.*g01 + g02)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g10 - g20 + g01 -2.*g11 + g21)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g10 + 4.*g01 -4.*g11 -2.*g02 +2.*g12)*xy;
    return exy * (y-x) + x * (1. - x) * 2. * poly;
}
double Mfunc3D_allo_a(double x, double y, double z, double exy, double mxz, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22){
    /*
    * x is x_a, y is x_b, z is a separate population
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g10 + (-2.*g10 + g20)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g01 - g02 + g10 -2.*g11 + g12)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g01 + 4.*g10 -4.*g11 -2.*g20 +2.*g21)*xy;
    return exy * (y-x) + mxz * (z-x) + x * (1. - x) * 2. * poly;
}
double Mfunc3D_allo_b(double x, double y, double z, double exy, double mxz, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22){
    /*
    * x is x_b, y is x_a, z is a separate population
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g01 + (-2.*g01 + g02)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g10 - g20 + g01 -2.*g11 + g21)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g10 + 4.*g01 -4.*g11 -2.*g02 +2.*g12)*xy;
    return exy * (y-x) + mxz * (z-x) + x * (1. - x) * 2. * poly;
}
double Mfunc4D_allo_a(double x, double y, double z, double a,
                      double exy, double mxz, double mxa,
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22){
    /*
    * x is x_a, y is x_b, z and a are separate populations
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g10 + (-2.*g10 + g20)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g01 - g02 + g10 -2.*g11 + g12)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g01 + 4.*g10 -4.*g11 -2.*g20 +2.*g21)*xy;
    return exy * (y-x) + mxz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
}
double Mfunc4D_allo_b(double x, double y, double z, double a,
                      double exy, double mxz, double mxa,
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22){
    /*
    * x is x_b, y is x_a, z and a are separate populations
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g01 + (-2.*g01 + g02)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g10 - g20 + g01 -2.*g11 + g21)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g10 + 4.*g01 -4.*g11 -2.*g02 +2.*g12)*xy;
    return exy * (y-x) + mxz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
}
double Mfunc5D_allo_a(double x, double y, double z, double a, double b,
                      double exy, double mxz, double mxa, double mxb,
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22){
    /*
    * x is x_a; y is x_b; z, a, and b are separate populations
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g10 + (-2.*g10 + g20)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g01 - g02 + g10 -2.*g11 + g12)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g01 + 4.*g10 -4.*g11 -2.*g20 +2.*g21)*xy;
    return exy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
}
double Mfunc5D_allo_b(double x, double y, double z, double a, double b,
                      double exy, double mxz, double mxa, double mxb,
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22){
    /*
    * x is x_b; y is x_a; z, a, and b are separate populations
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g01 + (-2.*g01 + g02)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g10 - g20 + g01 -2.*g11 + g21)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g10 + 4.*g01 -4.*g11 -2.*g02 +2.*g12)*xy;
    return exy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
}
/*
* AUTOHEXAPLOIDS POP GEN FUNCTIONS 
* 
* Note that gi refers to gamma_i (not a gamete frequency!)
*/
double Vfunc_hex(double x, double nu){
    return 1./(3.*nu) * x*(1.-x);
}
double Mfunc1D_autohex(double x, double g1, double g2, double g3, double g4, double g5, double g6){
    double poly = (((((-6.*g1 + 15.*g2 - 20.*g3 + 15.*g4 - 6.*g5 + g6) * x +
                      (25.*g1 - 50.*g2 + 50.*g3 - 25.*g4 + 5.*g5)) * x +
                      (-40.*g1 + 60.*g2 - 40.*g3 + 10.*g4)) * x +
                      (30.*g1 - 30.*g2 + 10.*g3)) * x +
                      (-10.*g1 + 5.*g2)) * x +
                      g1;
    return x * (1. - x) * 2. * poly;
}
double Mfunc2D_autohex(double x, double y, double mxy, double g1, double g2, double g3, double g4, double g5, double g6){
    double poly = (((((-6.*g1 + 15.*g2 - 20.*g3 + 15.*g4 - 6.*g5 + g6) * x +
                      (25.*g1 - 50.*g2 + 50.*g3 - 25.*g4 + 5.*g5)) * x +
                      (-40.*g1 + 60.*g2 - 40.*g3 + 10.*g4)) * x +
                      (30.*g1 - 30.*g2 + 10.*g3)) * x +
                      (-10.*g1 + 5.*g2)) * x +
                      g1;
    return mxy * (y-x) + x * (1. - x) * 2. * poly;
}
double Mfunc3D_autohex(double x, double y, double z, double mxy, double mxz, 
                   double g1, double g2, double g3, double g4, double g5, double g6){
    double poly = (((((-6.*g1 + 15.*g2 - 20.*g3 + 15.*g4 - 6.*g5 + g6) * x +
                      (25.*g1 - 50.*g2 + 50.*g3 - 25.*g4 + 5.*g5)) * x +
                      (-40.*g1 + 60.*g2 - 40.*g3 + 10.*g4)) * x +
                      (30.*g1 - 30.*g2 + 10.*g3)) * x +
                      (-10.*g1 + 5.*g2)) * x +
                      g1;
    return mxy * (y-x) + mxz * (z-x) + x * (1. - x) * 2. * poly;
}
double Mfunc4D_autohex(double x, double y, double z, double a, double mxy, double mxz, double mxa, 
                   double g1, double g2, double g3, double g4, double g5, double g6){
    double poly = (((((-6.*g1 + 15.*g2 - 20.*g3 + 15.*g4 - 6.*g5 + g6) * x +
                      (25.*g1 - 50.*g2 + 50.*g3 - 25.*g4 + 5.*g5)) * x +
                      (-40.*g1 + 60.*g2 - 40.*g3 + 10.*g4)) * x +
                      (30.*g1 - 30.*g2 + 10.*g3)) * x +
                      (-10.*g1 + 5.*g2)) * x +
                      g1;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
}
double Mfunc5D_autohex(double x, double y, double z, double a, double b,
                   double mxy, double mxz, double mxa, double mxb,
                   double g1, double g2, double g3, double g4, double g5, double g6){
    double poly = (((((-6.*g1 + 15.*g2 - 20.*g3 + 15.*g4 - 6.*g5 + g6) * x +
                      (25.*g1 - 50.*g2 + 50.*g3 - 25.*g4 + 5.*g5)) * x +
                      (-40.*g1 + 60.*g2 - 40.*g3 + 10.*g4)) * x +
                      (30.*g1 - 30.*g2 + 10.*g3)) * x +
                      (-10.*g1 + 5.*g2)) * x +
                      g1;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
}

/*
* 4+2 HEXAPLOIDS POP GEN FUNCTIONS 
* 
* Note that gij refers to gamma_ij (not a gamete frequency!)
* 
* Also note that x always refers to the *current* subgenome frequency and 
* y refers to the other subgenome. Thus, z, a, and b are always separate populations.
*/
double Mfunc2D_hex_tetra(double x, double y, double exy, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_4, y is x_2 where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    */
    double xy = x*y; // qa*qb
    double xx = x*x; // qa^2
    double xxx = xx*x; // qa^3
    double yy = y*y; // qb^2
    double xyy = xy*y; // qa*qb^2
    double xxy = xx*y; // qa^2*qb
    double xxxy = xxx*y; // qa^3*qb
    double xxyy = xxy*y; // qa^2*qb^2
    double xxxyy = xxxy*y; // qa^3*qb^2
    double poly = g10 + (-6.*g10 + 3.*g20) * x +
                  (-2.*g01 - 2.*g10 + 2.*g11) * y +
                  (9.*g10 -9.*g20 + 3.*g30) * xx + 
                  (-4.*g10 + 6.*g20 - 4.*g30 + g40) * xxx + 
                  (2.*g01 - g02 + g10 - 2.*g11 + g12) * yy + 
                  (-6.*g01 + 3.*g02 - 6.*g10 + 12.*g11 - 6.*g12 + 3.*g20 - 6.*g21 + 3.*g22) * xyy +
                  (-6.*g01 - 18.*g10 + 18.*g11 + 18.*g20 -18.*g21 -6.*g30 +6.*g31) * xxy +
                  (2.*g01 + 8.*g10 - 8.*g11 - 12.*g20 + 12.*g21 + 8.*g30 - 8.*g31 - 2.*g40 + 2.*g41) * xxxy +
                  (6.*g01 - 3.*g02 + 9.*g10 - 18.*g11 + 9.*g12 - 9.*g20 + 18.*g21 - 9.*g22 + 3.*g30 - 6.*g31 + 3.*g32) * xxyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xxxyy +
                  (6.*g01 + 12.*g10 - 12.*g11 - 6.*g20 + 6.*g21) * xy;
    // note the 1/2 term in the exchange term here to correct for differences in ploidy between subgenomes
    return exy * (y-x) / 2. + x * (1. - x) * 2. * poly;
}   

double Mfunc2D_hex_dip(double x, double y, double exy, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_2, y is x_4 
    * where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    */
    double xy = x*y; // qa*qb
    double yy = y*y; // qa^2
    double yyy = yy*y; // qa^3
    double yyyy = yyy*y; // qa^4
    double xyy = xy*y; // qa^2*qb
    double xyyy = xyy*y; // qa^3*qb
    double xyyyy = xyyy*y; // qa^4*qb
    double poly = g01 + (-4.*g01 - 4.*g10 + 4.*g11) * y + 
                  (-2.*g01 + g02) * x + 
                  (6.*g01 + 12.*g10 - 12.*g11 -6.*g20 + 6.*g21) * yy +
                  (-4.*g01 -12.*g10 + 12.*g11 + 12.*g20 - 12.*g21 - 4.*g30 + 4.*g31) * yyy + 
                  (g01 + 4.*g10 - 4.*g11 - 6.*g20 + 6.*g21 + 4.*g30 - 4.*g31 - g40 + g41) * yyyy +
                  (-12.*g01 + 6.*g02 - 12.*g10 + 24.*g11 - 12.*g12 + 6.*g20 - 12.*g21 + 6.*g22) * xyy + 
                  (8.*g01 - 4.*g02 + 12.*g10 - 24.*g11 + 12.*g12 - 12.*g20 + 24.*g21 - 12.*g22 + 4.*g30 - 8.*g31 + 4.*g32) * xyyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xyyyy + 
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12) * xy;
    return exy * (y-x) + x * (1. - x) * 2. * poly;
}         

double Mfunc3D_hex_tetra(double x, double y, double z, double exy, double mxz, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_4, y is x_2 where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    * z is a separate population
    */
    double xy = x*y;
    double xx = x*x;
    double xxx = xx*x;
    double yy = y*y;
    double xyy = xy*y;
    double xxy = xx*y;
    double xxxy = xxx*y;
    double xxyy = xxy*y;
    double xxxyy = xxxy*y;
    double poly = g10 + (-6.*g10 + 3.*g20) * x +
                  (-2.*g01 - 2.*g10 + 2.*g11) * y +
                  (9.*g10 -9.*g20 + 3.*g30) * xx + 
                  (-4.*g10 + 6.*g20 - 4.*g30 + g40) * xxx + 
                  (2.*g01 - g02 + g10 - 2.*g11 + g12) * yy + 
                  (-6.*g01 + 3.*g02 - 6.*g10 + 12.*g11 - 6.*g12 + 3.*g20 - 6.*g21 + 3.*g22) * xyy +
                  (-6.*g01 - 18.*g10 + 18.*g11 + 18.*g20 -18.*g21 -6.*g30 +6.*g31) * xxy +
                  (2.*g01 + 8.*g10 - 8.*g11 - 12.*g20 + 12.*g21 + 8.*g30 - 8.*g31 - 2.*g40 + 2.*g41) * xxxy +
                  (6.*g01 - 3.*g02 + 9.*g10 - 18.*g11 + 9.*g12 - 9.*g20 + 18.*g21 - 9.*g22 + 3.*g30 - 6.*g31 + 3.*g32) * xxyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xxxyy +
                  (6.*g01 + 12.*g10 - 12.*g11 - 6.*g20 + 6.*g21) * xy;
    // note the 1/2 term in the exchange term here to correct for differences in ploidy between subgenomes
    return exy * (y-x) / 2. + mxz * (z-x) + x * (1. - x) * 2. * poly;
}   

double Mfunc3D_hex_dip(double x, double y, double z, double exy, double mxz, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_2, y is x_4 
    * where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    */
    double xy = x*y; // qa*qb
    double yy = y*y; // qa^2
    double yyy = yy*y; // qa^3
    double yyyy = yyy*y; // qa^4
    double xyy = xy*y; // qa^2*qb
    double xyyy = xyy*y; // qa^3*qb
    double xyyyy = xyyy*y; // qa^4*qb
    double poly = g01 + (-4.*g01 - 4.*g10 + 4.*g11) * y + 
                  (-2.*g01 + g02) * x + 
                  (6.*g01 + 12.*g10 - 12.*g11 -6.*g20 + 6.*g21) * yy +
                  (-4.*g01 -12.*g10 + 12.*g11 + 12.*g20 - 12.*g21 - 4.*g30 + 4.*g31) * yyy + 
                  (g01 + 4.*g10 - 4.*g11 - 6.*g20 + 6.*g21 + 4.*g30 - 4.*g31 - g40 + g41) * yyyy +
                  (-12.*g01 + 6.*g02 - 12.*g10 + 24.*g11 - 12.*g12 + 6.*g20 - 12.*g21 + 6.*g22) * xyy + 
                  (8.*g01 - 4.*g02 + 12.*g10 - 24.*g11 + 12.*g12 - 12.*g20 + 24.*g21 - 12.*g22 + 4.*g30 - 8.*g31 + 4.*g32) * xyyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xyyyy + 
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12) * xy;
    return exy * (y-x) + mxz * (z-x) + x * (1. - x) * 2. * poly;
}         

double Mfunc4D_hex_tetra(double x, double y, double z, double a, double exy, double mxz, double mxa, 
                         double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_4, y is x_2 where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    * z and a are separate populations
    */
    double xy = x*y;
    double xx = x*x;
    double xxx = xx*x;
    double yy = y*y;
    double xyy = xy*y;
    double xxy = xx*y;
    double xxxy = xxx*y;
    double xxyy = xxy*y;
    double xxxyy = xxxy*y;
    double poly = g10 + (-6.*g10 + 3.*g20) * x +
                  (-2.*g01 - 2.*g10 + 2.*g11) * y +
                  (9.*g10 -9.*g20 + 3.*g30) * xx + 
                  (-4.*g10 + 6.*g20 - 4.*g30 + g40) * xxx + 
                  (2.*g01 - g02 + g10 - 2.*g11 + g12) * yy + 
                  (-6.*g01 + 3.*g02 - 6.*g10 + 12.*g11 - 6.*g12 + 3.*g20 - 6.*g21 + 3.*g22) * xyy +
                  (-6.*g01 - 18.*g10 + 18.*g11 + 18.*g20 -18.*g21 -6.*g30 +6.*g31) * xxy +
                  (2.*g01 + 8.*g10 - 8.*g11 - 12.*g20 + 12.*g21 + 8.*g30 - 8.*g31 - 2.*g40 + 2.*g41) * xxxy +
                  (6.*g01 - 3.*g02 + 9.*g10 - 18.*g11 + 9.*g12 - 9.*g20 + 18.*g21 - 9.*g22 + 3.*g30 - 6.*g31 + 3.*g32) * xxyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xxxyy +
                  (6.*g01 + 12.*g10 - 12.*g11 - 6.*g20 + 6.*g21) * xy;
    // note the 1/2 term in the exchange term here to correct for differences in ploidy between subgenomes
    return exy * (y-x) / 2. + mxz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
}

double Mfunc4D_hex_dip(double x, double y, double z, double a, double exy, double mxz, double mxa,
                       double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                       double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_2, y is x_4 
    * where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    */
    double xy = x*y; // qa*qb
    double yy = y*y; // qa^2
    double yyy = yy*y; // qa^3
    double yyyy = yyy*y; // qa^4
    double xyy = xy*y; // qa^2*qb
    double xyyy = xyy*y; // qa^3*qb
    double xyyyy = xyyy*y; // qa^4*qb
    double poly = g01 + (-4.*g01 - 4.*g10 + 4.*g11) * y + 
                  (-2.*g01 + g02) * x + 
                  (6.*g01 + 12.*g10 - 12.*g11 -6.*g20 + 6.*g21) * yy +
                  (-4.*g01 -12.*g10 + 12.*g11 + 12.*g20 - 12.*g21 - 4.*g30 + 4.*g31) * yyy + 
                  (g01 + 4.*g10 - 4.*g11 - 6.*g20 + 6.*g21 + 4.*g30 - 4.*g31 - g40 + g41) * yyyy +
                  (-12.*g01 + 6.*g02 - 12.*g10 + 24.*g11 - 12.*g12 + 6.*g20 - 12.*g21 + 6.*g22) * xyy + 
                  (8.*g01 - 4.*g02 + 12.*g10 - 24.*g11 + 12.*g12 - 12.*g20 + 24.*g21 - 12.*g22 + 4.*g30 - 8.*g31 + 4.*g32) * xyyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xyyyy + 
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12) * xy;
    return exy * (y-x) + mxz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
}         

double Mfunc5D_hex_tetra(double x, double y, double z, double a, double b, double exy, double mxz, double mxa, double mxb,
                         double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_4, y is x_2 where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    * z, a, and b are separate populations
    */
    double xy = x*y;
    double xx = x*x;
    double xxx = xx*x;
    double yy = y*y;
    double xyy = xy*y;
    double xxy = xx*y;
    double xxxy = xxx*y;
    double xxyy = xxy*y;
    double xxxyy = xxxy*y;
    double poly = g10 + (-6.*g10 + 3.*g20) * x +
                  (-2.*g01 - 2.*g10 + 2.*g11) * y +
                  (9.*g10 -9.*g20 + 3.*g30) * xx + 
                  (-4.*g10 + 6.*g20 - 4.*g30 + g40) * xxx + 
                  (2.*g01 - g02 + g10 - 2.*g11 + g12) * yy + 
                  (-6.*g01 + 3.*g02 - 6.*g10 + 12.*g11 - 6.*g12 + 3.*g20 - 6.*g21 + 3.*g22) * xyy +
                  (-6.*g01 - 18.*g10 + 18.*g11 + 18.*g20 -18.*g21 -6.*g30 +6.*g31) * xxy +
                  (2.*g01 + 8.*g10 - 8.*g11 - 12.*g20 + 12.*g21 + 8.*g30 - 8.*g31 - 2.*g40 + 2.*g41) * xxxy +
                  (6.*g01 - 3.*g02 + 9.*g10 - 18.*g11 + 9.*g12 - 9.*g20 + 18.*g21 - 9.*g22 + 3.*g30 - 6.*g31 + 3.*g32) * xxyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xxxyy +
                  (6.*g01 + 12.*g10 - 12.*g11 - 6.*g20 + 6.*g21) * xy;
    // note the 1/2 term in the exchange term here to correct for differences in ploidy between subgenomes
    return exy * (y-x) / 2. + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
}

double Mfunc5D_hex_dip(double x, double y, double z, double a, double b, double exy, double mxz, double mxa, double mxb,
                       double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                       double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_2, y is x_4 
    * where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    */
    double xy = x*y; // qa*qb
    double yy = y*y; // qa^2
    double yyy = yy*y; // qa^3
    double yyyy = yyy*y; // qa^4
    double xyy = xy*y; // qa^2*qb
    double xyyy = xyy*y; // qa^3*qb
    double xyyyy = xyyy*y; // qa^4*qb
    double poly = g01 + (-4.*g01 - 4.*g10 + 4.*g11) * y + 
                  (-2.*g01 + g02) * x + 
                  (6.*g01 + 12.*g10 - 12.*g11 -6.*g20 + 6.*g21) * yy +
                  (-4.*g01 -12.*g10 + 12.*g11 + 12.*g20 - 12.*g21 - 4.*g30 + 4.*g31) * yyy + 
                  (g01 + 4.*g10 - 4.*g11 - 6.*g20 + 6.*g21 + 4.*g30 - 4.*g31 - g40 + g41) * yyyy +
                  (-12.*g01 + 6.*g02 - 12.*g10 + 24.*g11 - 12.*g12 + 6.*g20 - 12.*g21 + 6.*g22) * xyy + 
                  (8.*g01 - 4.*g02 + 12.*g10 - 24.*g11 + 12.*g12 - 12.*g20 + 24.*g21 - 12.*g22 + 4.*g30 - 8.*g31 + 4.*g32) * xyyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xyyyy + 
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12) * xy;
    return exy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
}  

/*
* 2+2+2 HEXAPLOIDS POP GEN FUNCTIONS 
* 
* Note that gijk refers to gamma_ijk (not a gamete frequency!)
* 
* Also note that x always refers to the *current* subgenome frequency,
* y refers to the *first alphabetical* frequency of the remaining two,
* and z refers to the *third* frequency.
* Thus, a, and b are always separate populations.
*
* Comments inside each function specify this, but as an example, 
* for qb, we have 
*   x = qb (the current subgenome frequency)
*   y = qa (the first alphabetical subgenome frequency of the remaining two)
*   z = qc (the third subgenome frequency)
*/
double Mfunc3D_hex_a(double x, double y, double z, double exy, double exz, 
                    double g001, double g002, double g010, double g011, double g012, 
                    double g020, double g021, double g022, double g100, double g101, double g102,
                    double g110, double g111, double g112, double g120, double g121, double g122,
                    double g200, double g201, double g202, double g210, double g211, double g212,
                    double g220, double g221, double g222){
    /*
    * x is x_a, y is x_b, z is x_c
    */
    double yy = y*y; // qb^2
    double zz = z*z; // qc^2
    double xyy = x*yy; // qa*qb^2
    double xzz = x*zz; // qa*qc^2
    double yzz = y*zz; // qb*qc^2
    double yyz = yy*z; // qb^2*qc
    double yyzz = yy*zz; // qb^2*qc^2
    double xy = x*y; // qa*qb
    double xz = x*z; // qa*qc
    double yz = y*z; // qb*qc
    double xyz = xy*z; // qa*qb*qc
    double xyzz = xyz*z; // qa*qb*qc^2
    double xyyz = xyy*z; // qa*qb^2*qc
    double xyyzz = xyyz*z; // qa*qb^2*qc^2
    double poly = g100 + (- 2.*g100 + g200) * x + 
                  (-2.*g010 - 2.*g100 + 2.*g110) * y + 
                  (-2.*g001 - 2.*g100 + 2.*g101) * z +
                  (2.*g010 - g020 + g100 - 2.*g110 + g120) * yy +
                  (2.*g001 - g002 + g100 - 2.*g101 + g102) * zz + 
                  (-2.*g010 + g020 - 2.*g100 + 4.*g110 - 2.*g120 + g200 - 2.*g210 + g220) * xyy + 
                  (-2.*g001 + g002 - 2.*g100 + 4.*g101 - 2.*g102 + g200 - 2.*g201 + g202) * xzz +
                  (-4.*g001 + 2.*g002 - 2.*g010 + 4.*g011 - 2.*g012 - 2.*g100 + 4.*g101 - 2.*g102 + 2.*g110 - 4.*g111 + 2.*g112) * yzz +
                  (-2.*g001 - 4.*g010 + 4.*g011 + 2.*g020 - 2.*g021 - 2.*g100 + 2.*g101 + 4.*g110 - 4.*g111 - 2.*g120 + 2.*g121) * yyz +
                  (2.*g001 - g002 + 2.*g010 - 4.*g011 + 2.*g012  - g020 + 2.*g021 - g022 + g100 - 2.*g101  + g102 - 2.*g110 + 4.*g111 - 2.*g112 + g120 - 2.*g121 + g122) * yyzz +
                  (2.*g010 + 4.*g100 - 4.*g110 - 2.*g200 + 2.*g210) * xy +
                  (2.*g001 + 4.*g100 - 4.*g101 - 2.*g200 + 2.*g201) * xz + 
                  (4.*g001 + 4.*g010 - 4.*g011 + 4.*g100 - 4.*g101 - 4.*g110 + 4.*g111) * yz +
                  (-4.*g001 - 4.*g010 + 4.*g011 - 8.*g100 + 8.*g101 + 8.*g110 - 8.*g111 + 4.*g200 - 4.*g201 - 4.*g210 + 4.*g211) * xyz +
                  (4.*g001 - 2.*g002 + 2.*g010 - 4.*g011 + 2.*g012 + 4.*g100 - 8.*g101 + 4.*g102 - 4.*g110 + 8.*g111 - 4.*g112 - 2.*g200 + 4.*g201 - 2.*g202 + 2.*g210 - 4.*g211 + 2.*g212) * xyzz +
                  (2.*g001 + 4.*g010 - 4.*g011 - 2.*g020 + 2.*g021 + 4.*g100 - 4.*g101 - 8.*g110 + 8.*g111 + 4.*g120 - 4.*g121 - 2.*g200 + 2.*g201 + 4.*g210 - 4.*g211 - 2.*g220 + 2.*g221) * xyyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022 - 2.*g100 + 4.*g101 - 2.*g102 + 4.*g110 - 8.*g111 + 4.*g112 - 2.*g120 + 4.*g121 - 2.*g122 + g200 - 2.*g201 + g202 - 2.*g210 + 4.*g211 - 2.*g212 + g220 - 2.*g221 + g222) * xyyzz;
    return exy * (y-x) + exz * (z-x) + x * (1. - x) * 2. * poly;
} 

double Mfunc3D_hex_b(double x, double y, double z, double exy, double exz, 
                    double g001, double g002, double g010, double g011, double g012, 
                    double g020, double g021, double g022, double g100, double g101, double g102,
                    double g110, double g111, double g112, double g120, double g121, double g122,
                    double g200, double g201, double g202, double g210, double g211, double g212,
                    double g220, double g221, double g222){
    /*
    * x is x_b, y is x_a, z is x_c
    */
    double yy = y*y; // qa^2
    double zz = z*z; // qc^2
    double xyy = x*yy; // qa^2*qb
    double yzz = y*zz; // qa*qc^2
    double yyz = yy*z; // qa^2*qc
    double xzz = x*zz; // qb*qc^2
    double yyzz = yy*zz; // qa^2*qc^2
    double xy = x*y; // qa*qb
    double yz = y*z; // qa*qc
    double xz = x*z; // qb*qc
    double xyz = xy*z; // qa*qb*qc
    double xyzz = xyz*z; // qa*qb*qc^2
    double xyyz = xyy*z; // qa^2*qb*qc
    double xyyzz = xyyz*z; // qa^2*qb*qc^2
    double poly = g010 + (-2.*g010 - 2.*g100 + 2.*g110 ) * y + 
                  (-2.*g010 + g020) * x + 
                  (-2.*g001 - 2.*g010 + 2.*g011) * z + 
                  (g010 + 2.*g100 - 2.*g110 - g200 + g210) * yy + 
                  (2.*g001 - g002 + g010 - 2.*g011 + g012) * zz + 
                  (-2.*g010 + g020 - 2.*g100 + 4.*g110 - 2.*g120 + g200 - 2.*g210 + g220) * xyy + 
                  (-4.*g001 + 2.*g002 - 2.*g010 + 4.*g011 - 2.*g012 - 2.*g100 + 4.*g101- 2.*g102 + 2.*g110 - 4.*g111 + 2.*g112) * yzz + 
                  (-2.*g001 - 2.*g010 + 2.*g011 - 4.*g100 + 4.*g101 + 4.*g110 - 4.*g111 + 2.*g200 - 2.*g201 - 2.*g210 + 2.*g211) * yyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022) * xzz + 
                  (2.*g001 - g002 + g010 - 2.*g011 + g012 + 2.*g100 - 4.*g101 + 2.*g102 - 2.*g110 + 4.*g111 - 2.*g112 - g200 + 2.*g201 - g202 + g210 - 2.*g211 + g212) * yyzz + 
                  (4.*g010 - 2.*g020 + 2.*g100 - 4.*g110 + 2.*g120) * xy + 
                  (4.*g001 + 4.*g010 - 4.*g011 + 4.*g100 - 4.*g101 - 4.*g110 + 4.*g111) * yz + 
                  (2.*g001 + 4.*g010 - 4.*g011 - 2.*g020 + 2.*g021) * xz + 
                  (-4.*g001 - 8.*g010 + 8.*g011 + 4.*g020 - 4.*g021 - 4.*g100 + 4.*g101 + 8.*g110 - 8.*g111 - 4.*g120 + 4.*g121) * xyz + 
                  (4.*g001 - 2.*g002 + 4.*g010 - 8.*g011 + 4.*g012 - 2.*g020 + 4.*g021 - 2.*g022 + 2.*g100 - 4.*g101 + 2.*g102 - 4.*g110+ 8.*g111 - 4.*g112 + 2.*g120 - 4.*g121 + 2.*g122) * xyzz + 
                  (2.*g001 + 4.*g010 - 4.*g011 - 2.*g020 + 2.*g021 + 4.*g100 - 4.*g101 - 8.*g110 + 8.*g111 + 4.*g120  - 4.*g121 - 2.*g200 + 2.*g201 + 4.*g210 - 4.*g211 - 2.*g220 + 2.*g221) * xyyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022 - 2.*g100 + 4.*g101 - 2.*g102 + 4.*g110 - 8.*g111 + 4.*g112 - 2.*g120 + 4.*g121 - 2.*g122 + g200 - 2.*g201 + g202 - 2.*g210 + 4.*g211 - 2.*g212 + g220 - 2.*g221 + g222) * xyyzz;
    return exy * (y-x) + exz * (z-x) + x * (1. - x) * 2. * poly;
}        
        
double Mfunc3D_hex_c(double x, double y, double z, double exy, double exz, 
                    double g001, double g002, double g010, double g011, double g012, 
                    double g020, double g021, double g022, double g100, double g101, double g102,
                    double g110, double g111, double g112, double g120, double g121, double g122,
                    double g200, double g201, double g202, double g210, double g211, double g212,
                    double g220, double g221, double g222){
    /*
    * x is x_c, y is x_a, z is x_b
    */
    double yy = y*y; // qa^2
    double zz = z*z; // qb^2
    double yzz = y*zz; // qa*qb^2
    double yyz = yy*z; // qa^2*qb
    double xyy = x*yy; // qa^2*qc
    double xzz = x*zz; // qb^2*qc
    double yyzz = yy*zz; // qa^2*qb^2
    double yz = y*z; // qa*qb
    double xy = x*y; // qa*qc
    double xz = x*z; // qb*qc
    double xyz = xy*z; // qa*qb*qc
    double xyzz = xyz*z; // qa*qb^2*qc
    double xyyz = xyy*z; // qa^2*qb*qc
    double xyyzz = xyyz*z; // qa^2*qb^2*qc
    double poly = g001 + (-2.*g001 - 2.*g100 + 2.*g101) * y + 
                  (-2.*g001 - 2.*g010 + 2.*g011) * z + 
                  (-2.*g001 + g002) * x + 
                  (g001 + 2.*g100 - 2.*g101 - g200 + g201) * yy + 
                  (g001 + 2.*g010 - 2.*g011 - g020 + g021) * zz + 
                  (-2.*g001 - 4.*g010 + 4.*g011 + 2.*g020 - 2.*g021 - 2.*g100 + 2.*g101 + 4.*g110 - 4.*g111 - 2.*g120 + 2.*g121) * yzz + 
                  (-2.*g001 - 2.*g010 + 2.*g011 - 4.*g100 + 4.*g101 + 4.*g110 - 4.*g111 + 2.*g200 - 2.*g201 - 2.*g210 + 2.*g211) * yyz + 
                  (-2.*g001 + g002 - 2.*g100 + 4.*g101 - 2.*g102 + g200 - 2.*g201 + g202) * xyy + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022) * xzz + 
                  (g001 + 2.*g010 - 2.*g011 - g020 + g021 + 2.*g100 - 2.*g101 - 4.*g110 + 4.*g111 + 2.*g120 - 2.*g121 - g200 + g201 + 2.*g210 - 2.*g211 - g220 + g221) * yyzz + 
                  (4.*g001 + 4.*g010 - 4.*g011 + 4.*g100 - 4.*g101 - 4.*g110 + 4.*g111) * yz + 
                  (4.*g001 - 2.*g002 + 2.*g100 - 4.*g101 + 2.*g102) * xy + 
                  (4.*g001 - 2.*g002 + 2.*g010 - 4.*g011 + 2.*g012 ) * xz + 
                  (-8.*g001 + 4.*g002 - 4.*g010 + 8.*g011 - 4.*g012 - 4.*g100 + 8.*g101 - 4.*g102 + 4.*g110 - 8.*g111 + 4.*g112) * xyz + 
                  (4.*g001 - 2.*g002 + 4.*g010 - 8.*g011 + 4.*g012 - 2.*g020 + 4.*g021 - 2.*g022 + 2.*g100 - 4.*g101 + 2.*g102 - 4.*g110 + 8.*g111 - 4.*g112 + 2.*g120 - 4.*g121 + 2.*g122) * xyzz + 
                  (4.*g001 - 2.*g002 + 2.*g010 - 4.*g011 + 2.*g012 + 4.*g100 - 8.*g101 + 4.*g102 - 4.*g110 + 8.*g111 - 4.*g112 - 2.*g200 + 4.*g201 - 2.*g202 + 2.*g210 - 4.*g211 + 2.*g212) * xyyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022 - 2.*g100 + 4.*g101 - 2.*g102 + 4.*g110 - 8.*g111 + 4.*g112 - 2.*g120 + 4.*g121 - 2.*g122 + g200 - 2.*g201 + g202 - 2.*g210 + 4.*g211 - 2.*g212 + g220 - 2.*g221 + g222) * xyyzz;
    return exy * (y-x) + exz * (z-x) + x * (1. - x) * 2. * poly;
}                
     
double Mfunc4D_hex_a(double x, double y, double z, double a, double exy, double exz, double mxa,
                    double g001, double g002, double g010, double g011, double g012, 
                    double g020, double g021, double g022, double g100, double g101, double g102,
                    double g110, double g111, double g112, double g120, double g121, double g122,
                    double g200, double g201, double g202, double g210, double g211, double g212,
                    double g220, double g221, double g222){
    /*
    * x is x_a, y is x_b, z is x_c
    * a is a separate population
    */
    double yy = y*y; // qb^2
    double zz = z*z; // qc^2
    double xyy = x*yy; // qa*qb^2
    double xzz = x*zz; // qa*qc^2
    double yzz = y*zz; // qb*qc^2
    double yyz = yy*z; // qb^2*qc
    double yyzz = yy*zz; // qb^2*qc^2
    double xy = x*y; // qa*qb
    double xz = x*z; // qa*qc
    double yz = y*z; // qb*qc
    double xyz = xy*z; // qa*qb*qc
    double xyzz = xyz*z; // qa*qb*qc^2
    double xyyz = xyy*z; // qa*qb^2*qc
    double xyyzz = xyyz*z; // qa*qb^2*qc^2
    double poly = g100 + (- 2.*g100 + g200) * x + 
                  (-2.*g010 - 2.*g100 + 2.*g110) * y + 
                  (-2.*g001 - 2.*g100 + 2.*g101) * z +
                  (2.*g010 - g020 + g100 - 2.*g110 + g120) * yy +
                  (2.*g001 - g002 + g100 - 2.*g101 + g102) * zz + 
                  (-2.*g010 + g020 - 2.*g100 + 4.*g110 - 2.*g120 + g200 - 2.*g210 + g220) * xyy + 
                  (-2.*g001 + g002 - 2.*g100 + 4.*g101 - 2.*g102 + g200 - 2.*g201 + g202) * xzz +
                  (-4.*g001 + 2.*g002 - 2.*g010 + 4.*g011 - 2.*g012 - 2.*g100 + 4.*g101 - 2.*g102 + 2.*g110 - 4.*g111 + 2.*g112) * yzz +
                  (-2.*g001 - 4.*g010 + 4.*g011 + 2.*g020 - 2.*g021 - 2.*g100 + 2.*g101 + 4.*g110 - 4.*g111 - 2.*g120 + 2.*g121) * yyz +
                  (2.*g001 - g002 + 2.*g010 - 4.*g011 + 2.*g012  - g020 + 2.*g021 - g022 + g100 - 2.*g101  + g102 - 2.*g110 + 4.*g111 - 2.*g112 + g120 - 2.*g121 + g122) * yyzz +
                  (2.*g010 + 4.*g100 - 4.*g110 - 2.*g200 + 2.*g210) * xy +
                  (2.*g001 + 4.*g100 - 4.*g101 - 2.*g200 + 2.*g201) * xz + 
                  (4.*g001 + 4.*g010 - 4.*g011 + 4.*g100 - 4.*g101 - 4.*g110 + 4.*g111) * yz +
                  (-4.*g001 - 4.*g010 + 4.*g011 - 8.*g100 + 8.*g101 + 8.*g110 - 8.*g111 + 4.*g200 - 4.*g201 - 4.*g210 + 4.*g211) * xyz +
                  (4.*g001 - 2.*g002 + 2.*g010 - 4.*g011 + 2.*g012 + 4.*g100 - 8.*g101 + 4.*g102 - 4.*g110 + 8.*g111 - 4.*g112 - 2.*g200 + 4.*g201 - 2.*g202 + 2.*g210 - 4.*g211 + 2.*g212) * xyzz +
                  (2.*g001 + 4.*g010 - 4.*g011 - 2.*g020 + 2.*g021 + 4.*g100 - 4.*g101 - 8.*g110 + 8.*g111 + 4.*g120 - 4.*g121 - 2.*g200 + 2.*g201 + 4.*g210 - 4.*g211 - 2.*g220 + 2.*g221) * xyyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022 - 2.*g100 + 4.*g101 - 2.*g102 + 4.*g110 - 8.*g111 + 4.*g112 - 2.*g120 + 4.*g121 - 2.*g122 + g200 - 2.*g201 + g202 - 2.*g210 + 4.*g211 - 2.*g212 + g220 - 2.*g221 + g222) * xyyzz;
    return exy * (y-x) + exz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
} 

double Mfunc4D_hex_b(double x, double y, double z, double a, double exy, double exz, double mxa,
                    double g001, double g002, double g010, double g011, double g012, 
                    double g020, double g021, double g022, double g100, double g101, double g102,
                    double g110, double g111, double g112, double g120, double g121, double g122,
                    double g200, double g201, double g202, double g210, double g211, double g212,
                    double g220, double g221, double g222){
    /*
    * x is x_b, y is x_a, z is x_c
    * a is a separate population
    */
    double yy = y*y; // qa^2
    double zz = z*z; // qc^2
    double xyy = x*yy; // qa^2*qb
    double yzz = y*zz; // qa*qc^2
    double yyz = yy*z; // qa^2*qc
    double xzz = x*zz; // qb*qc^2
    double yyzz = yy*zz; // qa^2*qc^2
    double xy = x*y; // qa*qb
    double yz = y*z; // qa*qc
    double xz = x*z; // qb*qc
    double xyz = xy*z; // qa*qb*qc
    double xyzz = xyz*z; // qa*qb*qc^2
    double xyyz = xyy*z; // qa^2*qb*qc
    double xyyzz = xyyz*z; // qa^2*qb*qc^2
    double poly = g010 + (-2.*g010 - 2.*g100 + 2.*g110 ) * y + 
                  (-2.*g010 + g020) * x + 
                  (-2.*g001 - 2.*g010 + 2.*g011) * z + 
                  (g010 + 2.*g100 - 2.*g110 - g200 + g210) * yy + 
                  (2.*g001 - g002 + g010 - 2.*g011 + g012) * zz + 
                  (-2.*g010 + g020 - 2.*g100 + 4.*g110 - 2.*g120 + g200 - 2.*g210 + g220) * xyy + 
                  (-4.*g001 + 2.*g002 - 2.*g010 + 4.*g011 - 2.*g012 - 2.*g100 + 4.*g101- 2.*g102 + 2.*g110 - 4.*g111 + 2.*g112) * yzz + 
                  (-2.*g001 - 2.*g010 + 2.*g011 - 4.*g100 + 4.*g101 + 4.*g110 - 4.*g111 + 2.*g200 - 2.*g201 - 2.*g210 + 2.*g211) * yyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022) * xzz + 
                  (2.*g001 - g002 + g010 - 2.*g011 + g012 + 2.*g100 - 4.*g101 + 2.*g102 - 2.*g110 + 4.*g111 - 2.*g112 - g200 + 2.*g201 - g202 + g210 - 2.*g211 + g212) * yyzz + 
                  (4.*g010 - 2.*g020 + 2.*g100 - 4.*g110 + 2.*g120) * xy + 
                  (4.*g001 + 4.*g010 - 4.*g011 + 4.*g100 - 4.*g101 - 4.*g110 + 4.*g111) * yz + 
                  (2.*g001 + 4.*g010 - 4.*g011 - 2.*g020 + 2.*g021) * xz + 
                  (-4.*g001 - 8.*g010 + 8.*g011 + 4.*g020 - 4.*g021 - 4.*g100 + 4.*g101 + 8.*g110 - 8.*g111 - 4.*g120 + 4.*g121) * xyz + 
                  (4.*g001 - 2.*g002 + 4.*g010 - 8.*g011 + 4.*g012 - 2.*g020 + 4.*g021 - 2.*g022 + 2.*g100 - 4.*g101 + 2.*g102 - 4.*g110+ 8.*g111 - 4.*g112 + 2.*g120 - 4.*g121 + 2.*g122) * xyzz + 
                  (2.*g001 + 4.*g010 - 4.*g011 - 2.*g020 + 2.*g021 + 4.*g100 - 4.*g101 - 8.*g110 + 8.*g111 + 4.*g120  - 4.*g121 - 2.*g200 + 2.*g201 + 4.*g210 - 4.*g211 - 2.*g220 + 2.*g221) * xyyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022 - 2.*g100 + 4.*g101 - 2.*g102 + 4.*g110 - 8.*g111 + 4.*g112 - 2.*g120 + 4.*g121 - 2.*g122 + g200 - 2.*g201 + g202 - 2.*g210 + 4.*g211 - 2.*g212 + g220 - 2.*g221 + g222) * xyyzz;
    return exy * (y-x) + exz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
}        
        
double Mfunc4D_hex_c(double x, double y, double z, double a, double exy, double exz, double mxa,
                    double g001, double g002, double g010, double g011, double g012, 
                    double g020, double g021, double g022, double g100, double g101, double g102,
                    double g110, double g111, double g112, double g120, double g121, double g122,
                    double g200, double g201, double g202, double g210, double g211, double g212,
                    double g220, double g221, double g222){
    /*
    * x is x_c, y is x_a, z is x_b
    * a is a separate population
    */
    double yy = y*y; // qa^2
    double zz = z*z; // qb^2
    double yzz = y*zz; // qa*qb^2
    double yyz = yy*z; // qa^2*qb
    double xyy = x*yy; // qa^2*qc
    double xzz = x*zz; // qb^2*qc
    double yyzz = yy*zz; // qa^2*qb^2
    double yz = y*z; // qa*qb
    double xy = x*y; // qa*qc
    double xz = x*z; // qb*qc
    double xyz = xy*z; // qa*qb*qc
    double xyzz = xyz*z; // qa*qb^2*qc
    double xyyz = xyy*z; // qa^2*qb*qc
    double xyyzz = xyyz*z; // qa^2*qb^2*qc
    double poly = g001 + (-2.*g001 - 2.*g100 + 2.*g101) * y + 
                  (-2.*g001 - 2.*g010 + 2.*g011) * z + 
                  (-2.*g001 + g002) * x + 
                  (g001 + 2.*g100 - 2.*g101 - g200 + g201) * yy + 
                  (g001 + 2.*g010 - 2.*g011 - g020 + g021) * zz + 
                  (-2.*g001 - 4.*g010 + 4.*g011 + 2.*g020 - 2.*g021 - 2.*g100 + 2.*g101 + 4.*g110 - 4.*g111 - 2.*g120 + 2.*g121) * yzz + 
                  (-2.*g001 - 2.*g010 + 2.*g011 - 4.*g100 + 4.*g101 + 4.*g110 - 4.*g111 + 2.*g200 - 2.*g201 - 2.*g210 + 2.*g211) * yyz + 
                  (-2.*g001 + g002 - 2.*g100 + 4.*g101 - 2.*g102 + g200 - 2.*g201 + g202) * xyy + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022) * xzz + 
                  (g001 + 2.*g010 - 2.*g011 - g020 + g021 + 2.*g100 - 2.*g101 - 4.*g110 + 4.*g111 + 2.*g120 - 2.*g121 - g200 + g201 + 2.*g210 - 2.*g211 - g220 + g221) * yyzz + 
                  (4.*g001 + 4.*g010 - 4.*g011 + 4.*g100 - 4.*g101 - 4.*g110 + 4.*g111) * yz + 
                  (4.*g001 - 2.*g002 + 2.*g100 - 4.*g101 + 2.*g102) * xy + 
                  (4.*g001 - 2.*g002 + 2.*g010 - 4.*g011 + 2.*g012 ) * xz + 
                  (-8.*g001 + 4.*g002 - 4.*g010 + 8.*g011 - 4.*g012 - 4.*g100 + 8.*g101 - 4.*g102 + 4.*g110 - 8.*g111 + 4.*g112) * xyz + 
                  (4.*g001 - 2.*g002 + 4.*g010 - 8.*g011 + 4.*g012 - 2.*g020 + 4.*g021 - 2.*g022 + 2.*g100 - 4.*g101 + 2.*g102 - 4.*g110 + 8.*g111 - 4.*g112 + 2.*g120 - 4.*g121 + 2.*g122) * xyzz + 
                  (4.*g001 - 2.*g002 + 2.*g010 - 4.*g011 + 2.*g012 + 4.*g100 - 8.*g101 + 4.*g102 - 4.*g110 + 8.*g111 - 4.*g112 - 2.*g200 + 4.*g201 - 2.*g202 + 2.*g210 - 4.*g211 + 2.*g212) * xyyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022 - 2.*g100 + 4.*g101 - 2.*g102 + 4.*g110 - 8.*g111 + 4.*g112 - 2.*g120 + 4.*g121 - 2.*g122 + g200 - 2.*g201 + g202 - 2.*g210 + 4.*g211 - 2.*g212 + g220 - 2.*g221 + g222) * xyyzz;
    return exy * (y-x) + exz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
}      
  
double Mfunc5D_hex_a(double x, double y, double z, double a, double b, double exy, double exz, double mxa, double mxb,
                    double g001, double g002, double g010, double g011, double g012, 
                    double g020, double g021, double g022, double g100, double g101, double g102,
                    double g110, double g111, double g112, double g120, double g121, double g122,
                    double g200, double g201, double g202, double g210, double g211, double g212,
                    double g220, double g221, double g222){
    /*
    * x is x_a, y is x_b, z is x_c
    * a and b are a separate populations
    */
    double yy = y*y; // qb^2
    double zz = z*z; // qc^2
    double xyy = x*yy; // qa*qb^2
    double xzz = x*zz; // qa*qc^2
    double yzz = y*zz; // qb*qc^2
    double yyz = yy*z; // qb^2*qc
    double yyzz = yy*zz; // qb^2*qc^2
    double xy = x*y; // qa*qb
    double xz = x*z; // qa*qc
    double yz = y*z; // qb*qc
    double xyz = xy*z; // qa*qb*qc
    double xyzz = xyz*z; // qa*qb*qc^2
    double xyyz = xyy*z; // qa*qb^2*qc
    double xyyzz = xyyz*z; // qa*qb^2*qc^2
    double poly = g100 + (- 2.*g100 + g200) * x + 
                  (-2.*g010 - 2.*g100 + 2.*g110) * y + 
                  (-2.*g001 - 2.*g100 + 2.*g101) * z +
                  (2.*g010 - g020 + g100 - 2.*g110 + g120) * yy +
                  (2.*g001 - g002 + g100 - 2.*g101 + g102) * zz + 
                  (-2.*g010 + g020 - 2.*g100 + 4.*g110 - 2.*g120 + g200 - 2.*g210 + g220) * xyy + 
                  (-2.*g001 + g002 - 2.*g100 + 4.*g101 - 2.*g102 + g200 - 2.*g201 + g202) * xzz +
                  (-4.*g001 + 2.*g002 - 2.*g010 + 4.*g011 - 2.*g012 - 2.*g100 + 4.*g101 - 2.*g102 + 2.*g110 - 4.*g111 + 2.*g112) * yzz +
                  (-2.*g001 - 4.*g010 + 4.*g011 + 2.*g020 - 2.*g021 - 2.*g100 + 2.*g101 + 4.*g110 - 4.*g111 - 2.*g120 + 2.*g121) * yyz +
                  (2.*g001 - g002 + 2.*g010 - 4.*g011 + 2.*g012  - g020 + 2.*g021 - g022 + g100 - 2.*g101  + g102 - 2.*g110 + 4.*g111 - 2.*g112 + g120 - 2.*g121 + g122) * yyzz +
                  (2.*g010 + 4.*g100 - 4.*g110 - 2.*g200 + 2.*g210) * xy +
                  (2.*g001 + 4.*g100 - 4.*g101 - 2.*g200 + 2.*g201) * xz + 
                  (4.*g001 + 4.*g010 - 4.*g011 + 4.*g100 - 4.*g101 - 4.*g110 + 4.*g111) * yz +
                  (-4.*g001 - 4.*g010 + 4.*g011 - 8.*g100 + 8.*g101 + 8.*g110 - 8.*g111 + 4.*g200 - 4.*g201 - 4.*g210 + 4.*g211) * xyz +
                  (4.*g001 - 2.*g002 + 2.*g010 - 4.*g011 + 2.*g012 + 4.*g100 - 8.*g101 + 4.*g102 - 4.*g110 + 8.*g111 - 4.*g112 - 2.*g200 + 4.*g201 - 2.*g202 + 2.*g210 - 4.*g211 + 2.*g212) * xyzz +
                  (2.*g001 + 4.*g010 - 4.*g011 - 2.*g020 + 2.*g021 + 4.*g100 - 4.*g101 - 8.*g110 + 8.*g111 + 4.*g120 - 4.*g121 - 2.*g200 + 2.*g201 + 4.*g210 - 4.*g211 - 2.*g220 + 2.*g221) * xyyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022 - 2.*g100 + 4.*g101 - 2.*g102 + 4.*g110 - 8.*g111 + 4.*g112 - 2.*g120 + 4.*g121 - 2.*g122 + g200 - 2.*g201 + g202 - 2.*g210 + 4.*g211 - 2.*g212 + g220 - 2.*g221 + g222) * xyyzz;
    return exy * (y-x) + exz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
} 

double Mfunc5D_hex_b(double x, double y, double z, double a, double b, double exy, double exz, double mxa, double mxb,
                    double g001, double g002, double g010, double g011, double g012, 
                    double g020, double g021, double g022, double g100, double g101, double g102,
                    double g110, double g111, double g112, double g120, double g121, double g122,
                    double g200, double g201, double g202, double g210, double g211, double g212,
                    double g220, double g221, double g222){
    /*
    * x is x_b, y is x_a, z is x_c
    * a and b are separate populations
    */
    double yy = y*y; // qa^2
    double zz = z*z; // qc^2
    double xyy = x*yy; // qa^2*qb
    double yzz = y*zz; // qa*qc^2
    double yyz = yy*z; // qa^2*qc
    double xzz = x*zz; // qb*qc^2
    double yyzz = yy*zz; // qa^2*qc^2
    double xy = x*y; // qa*qb
    double yz = y*z; // qa*qc
    double xz = x*z; // qb*qc
    double xyz = xy*z; // qa*qb*qc
    double xyzz = xyz*z; // qa*qb*qc^2
    double xyyz = xyy*z; // qa^2*qb*qc
    double xyyzz = xyyz*z; // qa^2*qb*qc^2
    double poly = g010 + (-2.*g010 - 2.*g100 + 2.*g110 ) * y + 
                  (-2.*g010 + g020) * x + 
                  (-2.*g001 - 2.*g010 + 2.*g011) * z + 
                  (g010 + 2.*g100 - 2.*g110 - g200 + g210) * yy + 
                  (2.*g001 - g002 + g010 - 2.*g011 + g012) * zz + 
                  (-2.*g010 + g020 - 2.*g100 + 4.*g110 - 2.*g120 + g200 - 2.*g210 + g220) * xyy + 
                  (-4.*g001 + 2.*g002 - 2.*g010 + 4.*g011 - 2.*g012 - 2.*g100 + 4.*g101- 2.*g102 + 2.*g110 - 4.*g111 + 2.*g112) * yzz + 
                  (-2.*g001 - 2.*g010 + 2.*g011 - 4.*g100 + 4.*g101 + 4.*g110 - 4.*g111 + 2.*g200 - 2.*g201 - 2.*g210 + 2.*g211) * yyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022) * xzz + 
                  (2.*g001 - g002 + g010 - 2.*g011 + g012 + 2.*g100 - 4.*g101 + 2.*g102 - 2.*g110 + 4.*g111 - 2.*g112 - g200 + 2.*g201 - g202 + g210 - 2.*g211 + g212) * yyzz + 
                  (4.*g010 - 2.*g020 + 2.*g100 - 4.*g110 + 2.*g120) * xy + 
                  (4.*g001 + 4.*g010 - 4.*g011 + 4.*g100 - 4.*g101 - 4.*g110 + 4.*g111) * yz + 
                  (2.*g001 + 4.*g010 - 4.*g011 - 2.*g020 + 2.*g021) * xz + 
                  (-4.*g001 - 8.*g010 + 8.*g011 + 4.*g020 - 4.*g021 - 4.*g100 + 4.*g101 + 8.*g110 - 8.*g111 - 4.*g120 + 4.*g121) * xyz + 
                  (4.*g001 - 2.*g002 + 4.*g010 - 8.*g011 + 4.*g012 - 2.*g020 + 4.*g021 - 2.*g022 + 2.*g100 - 4.*g101 + 2.*g102 - 4.*g110+ 8.*g111 - 4.*g112 + 2.*g120 - 4.*g121 + 2.*g122) * xyzz + 
                  (2.*g001 + 4.*g010 - 4.*g011 - 2.*g020 + 2.*g021 + 4.*g100 - 4.*g101 - 8.*g110 + 8.*g111 + 4.*g120  - 4.*g121 - 2.*g200 + 2.*g201 + 4.*g210 - 4.*g211 - 2.*g220 + 2.*g221) * xyyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022 - 2.*g100 + 4.*g101 - 2.*g102 + 4.*g110 - 8.*g111 + 4.*g112 - 2.*g120 + 4.*g121 - 2.*g122 + g200 - 2.*g201 + g202 - 2.*g210 + 4.*g211 - 2.*g212 + g220 - 2.*g221 + g222) * xyyzz;
    return exy * (y-x) + exz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
}        
        
double Mfunc5D_hex_c(double x, double y, double z, double a, double b, double exy, double exz, double mxa, double mxb,
                    double g001, double g002, double g010, double g011, double g012, 
                    double g020, double g021, double g022, double g100, double g101, double g102,
                    double g110, double g111, double g112, double g120, double g121, double g122,
                    double g200, double g201, double g202, double g210, double g211, double g212,
                    double g220, double g221, double g222){
    /*
    * x is x_c, y is x_a, z is x_b
    * a and b are separate populations
    */
    double yy = y*y; // qa^2
    double zz = z*z; // qb^2
    double yzz = y*zz; // qa*qb^2
    double yyz = yy*z; // qa^2*qb
    double xyy = x*yy; // qa^2*qc
    double xzz = x*zz; // qb^2*qc
    double yyzz = yy*zz; // qa^2*qb^2
    double yz = y*z; // qa*qb
    double xy = x*y; // qa*qc
    double xz = x*z; // qb*qc
    double xyz = xy*z; // qa*qb*qc
    double xyzz = xyz*z; // qa*qb^2*qc
    double xyyz = xyy*z; // qa^2*qb*qc
    double xyyzz = xyyz*z; // qa^2*qb^2*qc
    double poly = g001 + (-2.*g001 - 2.*g100 + 2.*g101) * y + 
                  (-2.*g001 - 2.*g010 + 2.*g011) * z + 
                  (-2.*g001 + g002) * x + 
                  (g001 + 2.*g100 - 2.*g101 - g200 + g201) * yy + 
                  (g001 + 2.*g010 - 2.*g011 - g020 + g021) * zz + 
                  (-2.*g001 - 4.*g010 + 4.*g011 + 2.*g020 - 2.*g021 - 2.*g100 + 2.*g101 + 4.*g110 - 4.*g111 - 2.*g120 + 2.*g121) * yzz + 
                  (-2.*g001 - 2.*g010 + 2.*g011 - 4.*g100 + 4.*g101 + 4.*g110 - 4.*g111 + 2.*g200 - 2.*g201 - 2.*g210 + 2.*g211) * yyz + 
                  (-2.*g001 + g002 - 2.*g100 + 4.*g101 - 2.*g102 + g200 - 2.*g201 + g202) * xyy + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022) * xzz + 
                  (g001 + 2.*g010 - 2.*g011 - g020 + g021 + 2.*g100 - 2.*g101 - 4.*g110 + 4.*g111 + 2.*g120 - 2.*g121 - g200 + g201 + 2.*g210 - 2.*g211 - g220 + g221) * yyzz + 
                  (4.*g001 + 4.*g010 - 4.*g011 + 4.*g100 - 4.*g101 - 4.*g110 + 4.*g111) * yz + 
                  (4.*g001 - 2.*g002 + 2.*g100 - 4.*g101 + 2.*g102) * xy + 
                  (4.*g001 - 2.*g002 + 2.*g010 - 4.*g011 + 2.*g012 ) * xz + 
                  (-8.*g001 + 4.*g002 - 4.*g010 + 8.*g011 - 4.*g012 - 4.*g100 + 8.*g101 - 4.*g102 + 4.*g110 - 8.*g111 + 4.*g112) * xyz + 
                  (4.*g001 - 2.*g002 + 4.*g010 - 8.*g011 + 4.*g012 - 2.*g020 + 4.*g021 - 2.*g022 + 2.*g100 - 4.*g101 + 2.*g102 - 4.*g110 + 8.*g111 - 4.*g112 + 2.*g120 - 4.*g121 + 2.*g122) * xyzz + 
                  (4.*g001 - 2.*g002 + 2.*g010 - 4.*g011 + 2.*g012 + 4.*g100 - 8.*g101 + 4.*g102 - 4.*g110 + 8.*g111 - 4.*g112 - 2.*g200 + 4.*g201 - 2.*g202 + 2.*g210 - 4.*g211 + 2.*g212) * xyyz + 
                  (-2.*g001 + g002 - 2.*g010 + 4.*g011 - 2.*g012 + g020 - 2.*g021 + g022 - 2.*g100 + 4.*g101 - 2.*g102 + 4.*g110 - 8.*g111 + 4.*g112 - 2.*g120 + 4.*g121 - 2.*g122 + g200 - 2.*g201 + g202 - 2.*g210 + 4.*g211 - 2.*g212 + g220 - 2.*g221 + g222) * xyyzz;
    return exy * (y-x) + exz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
} 
 