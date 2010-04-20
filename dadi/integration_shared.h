/* First, functions that define the dynamics we're integrating. These are not
 * used in the 'precalc' integration functions. For those, we do all this work
 * in Python.
 */
double Vfunc(double x, double nu);
double Mfunc1D(double x, double gamma, double h);
double Mfunc2D(double x, double y, double m, double gamma, double h);
double Mfunc3D(double x, double y, double z, double mxy, double mxz,
        double gamma, double h);

/* Differences between x values.
 */
void compute_dx(double *xx, int N, double *dx);
/* Values intermediate between x values.
 */
void compute_xInt(double *xx, int N, double *xInt);
/* dfactor which normalizes the fluxes to be consistent with probability
 * conservation given the trapezoid rule.
 */
void compute_dfactor(double *dx, int N, double *dfactor);
/* Chang and Cooper's delj factor, if use_delj_trick is True. Else just returns
 * an array of 0.5.
 */
void compute_delj(double *dx, double *MInt, double *VInt,
        int N, double *delj, int use_delj_trick);
/* a,b,c arrays for use with tridiag, corresponding to a fully implicit
 * integration. Doing them simultaneously allows an easy but minor optimization.
 */
void compute_abc_nobc(double *dx, double *dfactor, 
        double *delj, double *MInt, double *V, double dt, int N,
        double *a, double *b, double *c);
