#include <stdio.h>
#include <math.h>

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
    return exy * (x-y) / 2. + x * (1. - x) * 2. * poly;
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
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12);
    return exy * (x-y) + x * (1. - x) * 2. * poly;
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
    return exy * (x-y) / 2. + mxz * (z-x) + x * (1. - x) * 2. * poly;
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
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12);
    return exy * (x-y) + mxz * (z-x) + x * (1. - x) * 2. * poly;
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
    return exy * (x-y) / 2. + mxz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
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
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12);
    return exy * (x-y) + mxz * (z-x) + mxa * (a-x) + x * (1. - x) * 2. * poly;
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
    return exy * (x-y) / 2. + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
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
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12);
    return exy * (x-y) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + x * (1. - x) * 2. * poly;
}  