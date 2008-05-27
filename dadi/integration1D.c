#include "integration_shared.h"
#include "tridiag.h"

#include <stdio.h>

void implicit_1Dx(int L;
        double phi[L], double xx[L],
        double nu, double gamma, double dt, int L, int use_delj_trick){
    int ii;
    
    double dx[L-1], dfactor[L], xInt[L-1];
    compute_dx(xx, L, dx);
    compute_dfactor(dx, L, dfactor);
    compute_xInt(xx, L, xInt);

    double Mfirst, Mlast;
    double MInt[L-1], V[L], VInt[L-1];
    Mfirst = Mfunc1D(xx[0], gamma);
    Mlast = Mfunc1D(xx[L-1], gamma);
    for(ii=0; ii < L; ii++)
        V[ii] = Vfunc(xx[ii], nu);
    for(ii=0; ii < L-1; ii++){
        MInt[ii] = Mfunc1D(xInt[ii], gamma);
        VInt[ii] = Vfunc(xInt[ii], nu);
    }

    double delj[L-1];
    compute_delj(dx, MInt, VInt, L, delj, use_delj_trick);

    double a[L], b[L], c[L], r[L];
    compute_abc_nobc(dx, dfactor, delj, MInt, V, dt, L, a, b, c);
    for(ii = 0; ii < L; ii++)
        r[ii] = phi[ii]/dt;

    // Boundary conditions
    if(Mfirst <= 0)
        b[0] += (0.5/nu - Mfirst)*2./dx[0];
    if(Mlast >= 0)
        b[L-1] += -(-0.5/nu - Mlast)*2./dx[L-2];
    
    tridiag(a, b, c, r, phi, L);
}
