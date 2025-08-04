#include <stdio.h>
#include <math.h>

double Vfunc_auto(double x, double nu){
    return 1./(2.*nu) * x*(1.-x);
}
/* Use Horner's method here to evaluate the polynomia as an easy optimization
*/
double Mfunc1D_auto(double x, double gam1, double gam2, double gam3, double gam4){
    double poly = ((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                  (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                  (-6.*gam1 + 3.*gam2) * x +
                  gam1;
    return x * (1. - x) * 2. * poly;
}
