#include "integration_shared.h"
#include "tridiag.h"

#include <stdio.h>
#include <stdlib.h>

/* Visual C++ does not support the C99 standard, so we can't use
 * variable-length arrays such as arr[L][M].
 *
 * Given that, f2py passes 2D arrays in as flat 1D arrays. As a result,
 * indexing is more complicated.  If we have an array arr[L][M],
 * then arr[ii][jj] = arr[ii*M + jj]. More subtly, if we want to pull
 * out row ii to pass to another function, the notation is &arr[ii*M].
 *
 * To see versions of these functions that use variable-length arrays (and
 * thus are easier to understand, look prior to SVN revision 351.
 */

void implicit_2Dx(double *phi, double *xx, double *yy,
        double nu1, double m12, double gamma1, double h1,
        double dt, int L, int M, int use_delj_trick){
    int ii, jj;

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
    double *temp = malloc(L * sizeof(*temp));

    double y;

    compute_dx(xx, L, dx);
    compute_dfactor(dx, L, dfactor);
    compute_xInt(xx, L, xInt);
    
    for(ii=0; ii < L; ii++)
        V[ii] = Vfunc(xx[ii], nu1);
    for(ii=0; ii < L-1; ii++)
        VInt[ii] = Vfunc(xInt[ii], nu1);

    tridiag_malloc(L);
    for(jj=0; jj < M; jj++){
        y = yy[jj];

        Mfirst = Mfunc2D(xx[0], y, m12, gamma1, h1);
        Mlast = Mfunc2D(xx[L-1], y, m12, gamma1, h1);
        for(ii=0; ii < L-1; ii++)
            MInt[ii] = Mfunc2D(xInt[ii], y, m12, gamma1, h1);

        compute_delj(dx, MInt, VInt, L, delj, use_delj_trick);
        compute_abc_nobc(dx, dfactor, delj, MInt, V, dt, L, a, b, c);
        for(ii = 0; ii < L; ii++)
            r[ii] = phi[ii*M + jj]/dt;

        if((jj==0) && (Mfirst <= 0))
            b[0] += (0.5/nu1 - Mfirst)*2./dx[0];
        if((jj==M-1) && (Mlast >= 0))
            b[L-1] += -(-0.5/nu1 - Mlast)*2./dx[L-2];

        tridiag_premalloc(a, b, c, r, temp, L);
        for(ii = 0; ii < L; ii++)
            phi[ii*M + jj] = temp[ii];
    }
    tridiag_free();

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
    free(temp);
}

void implicit_2Dy(double *phi, double *xx, double *yy,
        double nu2, double m21, double gamma2, double h2,
        double dt, int L, int M, int use_delj_trick){
    int ii, jj;

    double *dy = malloc((M-1) * sizeof(*dy));
    double *dfactor = malloc(M * sizeof(*dfactor));
    double *yInt = malloc((M-1) * sizeof(*yInt));

    double Mfirst, Mlast;

    double *MInt = malloc((M-1) * sizeof(*MInt));
    double *V = malloc(M * sizeof(*V));
    double *VInt = malloc((M-1) * sizeof(*VInt));

    double *delj = malloc((M-1) * sizeof(*delj));

    double *a = malloc(M * sizeof(*a));
    double *b = malloc(M * sizeof(*b));
    double *c = malloc(M * sizeof(*c));
    double *r = malloc(M * sizeof(*r));

    double x;

    compute_dx(yy, M, dy);
    compute_dfactor(dy, M, dfactor);
    compute_xInt(yy, M, yInt);
    
    for(jj=0; jj < M; jj++)
        V[jj] = Vfunc(yy[jj], nu2);
    for(jj=0; jj < M-1; jj++)
        VInt[jj] = Vfunc(yInt[jj], nu2);

    tridiag_malloc(M);
    for(ii=0; ii < L; ii++){
        x = xx[ii];

        Mfirst = Mfunc2D(yy[0], x, m21, gamma2, h2);
        Mlast = Mfunc2D(yy[M-1], x, m21, gamma2, h2);
        for(jj=0; jj < M-1; jj++)
            MInt[jj] = Mfunc2D(yInt[jj], x, m21, gamma2, h2);

        compute_delj(dy, MInt, VInt, M, delj, use_delj_trick);
        compute_abc_nobc(dy, dfactor, delj, MInt, V, dt, M, a, b, c);
        for(jj = 0; jj < M; jj++)
            r[jj] = phi[ii*M + jj]/dt;

        if((ii==0) && (Mfirst <= 0))
            b[0] += (0.5/nu2 - Mfirst)*2./dy[0];
        if((ii==L-1) && (Mlast >= 0))
            b[M-1] += -(-0.5/nu2 - Mlast)*2./dy[M-2];

        tridiag_premalloc(a, b, c, r, &phi[ii*M], M);
    }
    tridiag_free();

    free(dy);
    free(dfactor);
    free(yInt);
    free(MInt);
    free(V);
    free(VInt);
    free(delj);
    free(a);
    free(b);
    free(c);
    free(r);
}

void implicit_precalc_2Dx(double *phi, double *ax, double *bx, double *cx,
        double dt, int L, int M){
    /* Warning: The bx passed in here should *not* include the 1/dt
     * contribution.
     */
    int ii, jj;

    double *a = malloc(L * sizeof(*a));
    double *b = malloc(L * sizeof(*b));
    double *c = malloc(L * sizeof(*c));
    double *r = malloc(L * sizeof(*r));
    double *temp = malloc(L * sizeof(*temp));

    tridiag_malloc(L);
    for(jj=0; jj < M; jj++){
        for(ii = 0; ii < L; ii++){
            a[ii] = ax[ii*M + jj];
            b[ii] = bx[ii*M + jj] + 1/dt;
            c[ii] = cx[ii*M + jj];
            r[ii] = 1/dt * phi[ii*M + jj];
        }

        tridiag_premalloc(a, b, c, r, temp, L);
        for(ii = 0; ii < L; ii++)
            phi[ii*M + jj] = temp[ii];
    }
    tridiag_free();

    free(a);
    free(b);
    free(c);
    free(r);
    free(temp);
}

void implicit_precalc_2Dy(double *phi, double *ay, double *by, double *cy,
        double dt, int L, int M){
    int ii, jj;

    double *b = malloc(M * sizeof(*b));
    double *r = malloc(M * sizeof(*r));

    tridiag_malloc(M);
    for(ii = 0; ii < L; ii++){
        for(jj = 0; jj < M; jj++){
            b[jj] = by[ii*M + jj] + 1/dt;
            r[jj] = 1/dt * phi[ii*M + jj];
        }

        tridiag_premalloc(&ay[ii*M], b, &cy[ii*M], r, &phi[ii*M], M);
    }
    tridiag_free();

    free(b);
    free(r);
}
