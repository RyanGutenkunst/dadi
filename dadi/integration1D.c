#include "integration_shared.h"
#include "tridiag.h"

#include <stdio.h>
#include <stdlib.h>

void implicit_1Dx(double *phi, double *xx,
        double nu, double gamma, double h, double dt, int L, 
        int use_delj_trick){
    int ii;
    
    double *dx = malloc((L-1) * sizeof(*dx));
    double *dfactor = malloc(L * sizeof(*dfactor));
    double *xInt = malloc((L-1) * sizeof(*xInt));

    double Mfirst, Mlast;
    double *MInt = malloc((L-1) * sizeof(*MInt));
    double *V = malloc(L * sizeof(*V));
    double *VInt = malloc((L-1) * sizeof(*VInt));

    double *delj = malloc((L-1) * sizeof(*delj));

    double *a = malloc(L * sizeof(*a));
    double *b = malloc(L * sizeof(*b));
    double *c = malloc(L * sizeof(*c));
    double *r = malloc(L * sizeof(*r));

    compute_dx(xx, L, dx);
    compute_dfactor(dx, L, dfactor);
    compute_xInt(xx, L, xInt);

    Mfirst = Mfunc1D(xx[0], gamma, h);
    Mlast = Mfunc1D(xx[L-1], gamma, h);
    for(ii=0; ii < L; ii++)
        V[ii] = Vfunc(xx[ii], nu);
    for(ii=0; ii < L-1; ii++){
        MInt[ii] = Mfunc1D(xInt[ii], gamma, h);
        VInt[ii] = Vfunc(xInt[ii], nu);
    }

    compute_delj(dx, MInt, VInt, L, delj, use_delj_trick);

    compute_abc_nobc(dx, dfactor, delj, MInt, V, dt, L, a, b, c);
    for(ii = 0; ii < L; ii++)
        r[ii] = phi[ii]/dt;

    /* Boundary conditions */
    if(Mfirst <= 0)
        b[0] += (0.5/nu - Mfirst)*2./dx[0];
    if(Mlast >= 0)
        b[L-1] += -(-0.5/nu - Mlast)*2./dx[L-2];
    
    tridiag(a, b, c, r, phi, L);

    free(dx);
    free(dfactor);
    free(xInt);
    free(MInt);
    free(V);
    free(VInt);
    free(delj);
    free(a);
    free(b);
    free(c);
    free(r);
}
