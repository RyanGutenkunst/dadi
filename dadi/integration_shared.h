/* First, functions that define the dynamics we're integrating. These are not
 * used in the 'precalc' integration functions. For those, we do all this work
 * in Python.
 */
double Vfunc(double x, double nu);
double Mfunc1D(double x, double gamma, double h);
double Mfunc2D(double x, double y, double m, double gamma, double h);
double Mfunc3D(double x, double y, double z, double mxy, double mxz,
        double gamma, double h);

// Differences between x values.
void compute_dx(int N; double xx[N], int N, double dx[N-1]);
// Values intermediate between x values.
void compute_xInt(int N; double xx[N], int N, double xInt[N-1]);
/* dfactor which normalizes the fluxes to be consistent with probability
 * conservation given the trapezoid rule.
 */
void compute_dfactor(int N; double dx[N-1], int N, double dfactor[N]);
/* Chang and Cooper's delj factor, if use_delj_trick is True. Else just returns
 * an array of 0.5.
 */
void compute_delj(int N; double dx[N-1], double MInt[N-1], double VInt[N-1],
        int N, double delj[N-1], int use_delj_trick);
/* a,b,c arrays for use with tridiag, corresponding to a fully implicit
 * integration. Doing them simultaneously allows an easy but minor optimization.
 */
void compute_abc_nobc(int N; double dx[N-1], double dfactor[N], 
        double delj[N-1], double MInt[N-1], double V[N], double dt, int N,
        double a[N], double b[N], double c[N]);
