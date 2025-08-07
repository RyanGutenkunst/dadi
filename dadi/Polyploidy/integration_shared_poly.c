#include <stdio.h>
#include <math.h>

/*
* AUTOTETRAPLOIDS POP GEN FUNCTIONS
*/
double Vfunc_auto(double x, double nu){
    return 1./(2.*nu) * x*(1.-x);
}
// Use Horner's method here to evaluate the polynomial is an easy optimization
double Mfunc1D_auto(double x, double gam1, double gam2, double gam3, double gam4){
    double poly = ((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                  (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                  (-6.*gam1 + 3.*gam2) * x +
                  gam1;
    return x * (1. - x) * 2. * poly;
}
double Mfunc2D_auto(double x, double y, double mxy, double gam1, double gam2, double gam3, double gam4){
    double poly = ((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                  (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                  (-6.*gam1 + 3.*gam2) * x + 
                  gam1;
    return mxy * (y-x) + x * (1. - x) * 2. * poly;
}

/*
* ALLOTETRAPLOIDS POP GEN FUNCTIONS 
* Note that the variance is the same as for diploids.
* 
* Also, note that it is non-trivial to write a single function for
* both subgenomes, so I wrote one function for each subgenome.
* I think you could swap x with y and swap g01 with g10, g20 with g02, etc., 
* but that's a mess to deal with in the Cython integration code.
* In each case, x is treated as the *current* subgenome frequency.
*/
double Mfunc2D_allo_a(double x, double y, double mxy, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22){
    /*
    * x is x_a, y is x_b
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g10 + (-2*g10 + g20)*x + 
                  (-2*g01 - 2*g10 + 2*g11)*y +
                  (2*g01 - g02 + g10 -2*g11 + g12)*yy +
                  (-2*g01 + g02 - 2*g10 + 4*g11 -2*g12 + g20 -2*g21 + g22)*xyy + 
                  (2*g01 + 4*g10 -4*g11 -2*g20 +2*g21)*xy;
    return mxy * (y-x) + x * (1. - x) * 2. * poly;
}
double Mfunc2D_allo_b(double x, double y, double mxy, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22){
    /*
    * x is x_b, y is x_a
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g01 + (-2*g01 + g02)*x + 
                  (-2*g01 - 2*g10 + 2*g11)*y +
                  (2*g10 - g20 + g01 -2*g11 + g21)*yy +
                  (-2*g01 + g02 - 2*g10 + 4*g11 -2*g12 + g20 -2*g21 + g22)*xyy + 
                  (2*g10 + 4*g01 -4*g11 -2*g02 +2*g12)*xy;
    return mxy * (y-x) + x * (1. - x) * 2. * poly;
}
