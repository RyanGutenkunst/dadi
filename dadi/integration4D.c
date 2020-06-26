#include "integration_shared.h"
#include "tridiag.h"
#include <stdio.h>
#include <stdlib.h>

/* Visual C++ does not support the C99 standard, so we can't use
 * variable-length arrays such as arr[L][M][N].
 *
 * See integration2D.c for detail on how this complicates indexing, and what
 * tricks are necessary.
 */

void implicit_4Dx(double *phi, double *xx, double *yy, double *zz, double *aa,
        double nu1, double m12, double m13, double m14, double gamma1, double h1,
        double dt, int L, int M, int N, int O, int use_delj_trick){
    int ii,jj,kk,ll;

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

    double y, z, a_;

    compute_dx(xx, L, dx);
    compute_dfactor(dx, L, dfactor);
    compute_xInt(xx, L, xInt);
    
    for(ii=0; ii < L; ii++)
        V[ii] = Vfunc(xx[ii], nu1);
    for(ii=0; ii < L-1; ii++)
        VInt[ii] = Vfunc(xInt[ii], nu1);

    tridiag_malloc(L);
    for(jj = 0; jj < M; jj++){
        for(kk = 0; kk < N; kk++){
            for(ll = 0; ll < O; ll++){
                y = yy[jj];
                z = zz[kk];
                a_ = aa[ll];

                Mfirst = Mfunc4D(xx[0], y, z, a_, m12, m13, m14, gamma1, h1);
                Mlast = Mfunc4D(xx[L-1], y, z, a_, m12, m13, m14, gamma1, h1);
                for(ii=0; ii < L-1; ii++)
                    MInt[ii] = Mfunc4D(xInt[ii], y, z, a_, m12, m13, m14, gamma1, h1);

                compute_delj(dx, MInt, VInt, L, delj, use_delj_trick);
                compute_abc_nobc(dx, dfactor, delj, MInt, V, dt, L, a, b, c);
                for(ii = 0; ii < L; ii++)
                    r[ii] = phi[ii*M*N*O + jj*N*O + kk*O + ll]/dt;

                if((yy[jj]==0) && (zz[kk]==0) && (aa[ll]==0) && (Mfirst <= 0))
                    b[0] += (0.5/nu1 - Mfirst)*2./dx[0];
                if((yy[jj]==1) && (zz[kk]==1) && (aa[ll]==1) && (Mlast >= 0))
                    b[L-1] += -(-0.5/nu1 - Mlast)*2./dx[L-2];

                tridiag_premalloc(a, b, c, r, temp, L);
                for(ii = 0; ii < L; ii++)
                    phi[ii*M*N*O + jj*N*O + kk*O + ll] = temp[ii];
            }
        }
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

void implicit_4Dy(double *phi, double *xx, double *yy, double *zz, double *aa,
        double nu2, double m21, double m23, double m24, double gamma2, double h2,
        double dt, int L, int M, int N, int O, int use_delj_trick){
    int ii,jj,kk,ll;

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
    double *temp = malloc(M * sizeof(*temp));

    double x, z, a_;

    compute_dx(yy, M, dy);
    compute_dfactor(dy, M, dfactor);
    compute_xInt(yy, M, yInt);
    
    for(jj=0; jj < M; jj++)
        V[jj] = Vfunc(yy[jj], nu2);
    for(jj=0; jj < M-1; jj++)
        VInt[jj] = Vfunc(yInt[jj], nu2);

    tridiag_malloc(M);
    for(ii = 0; ii < L; ii++){
        for(kk = 0; kk < N; kk++){
            for(ll = 0; ll < O; ll++){
                x = xx[ii];
                z = zz[kk];
                a_ = aa[ll];

                Mfirst = Mfunc4D(yy[0], x, z, a_, m21, m23, m24, gamma2, h2);
                Mlast = Mfunc4D(yy[M-1], x, z, a_, m21, m23, m24, gamma2, h2);
                for(jj=0; jj < M-1; jj++)
                    MInt[jj] = Mfunc4D(yInt[jj], x, z, a_, m21, m23, m24, gamma2, h2);

                compute_delj(dy, MInt, VInt, M, delj, use_delj_trick);
                compute_abc_nobc(dy, dfactor, delj, MInt, V, dt, M, a, b, c);
                for(jj = 0; jj < M; jj++)
                    r[jj] = phi[ii*M*N*O + jj*N*O + kk*O + ll]/dt;

                if((xx[ii]==0) && (zz[kk]==0) && (aa[ll]==0) && (Mfirst <= 0))
                    b[0] += (0.5/nu2 - Mfirst)*2./dy[0];
                if((xx[ii]==1) && (zz[kk]==1) && (aa[ll]==1) && (Mlast >= 0))
                    b[M-1] += -(-0.5/nu2 - Mlast)*2./dy[M-2];

                tridiag_premalloc(a, b, c, r, temp, M);
                for(jj = 0; jj < M; jj++)
                    phi[ii*M*N*O + jj*N*O + kk*O + ll] = temp[jj];
            }
        }
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
    free(temp);
}

void implicit_4Dz(double *phi, double *xx, double *yy, double *zz, double *aa,
        double nu3, double m31, double m32, double m34, double gamma3, double h3,
        double dt, int L, int M, int N, int O, int use_delj_trick){
    int ii,jj,kk,ll;

    double *dz = malloc((N-1) * sizeof(*dz));
    double *dfactor = malloc(N * sizeof(*dfactor));
    double *zInt = malloc((N-1) * sizeof(*zInt));

    double Mfirst, Mlast;

    double *MInt = malloc((N-1) * sizeof(*MInt));
    double *V = malloc(N * sizeof(*V));
    double *VInt = malloc((N-1) * sizeof(*VInt));

    double *delj = malloc((N-1) * sizeof(*delj));

    double *a = malloc(N * sizeof(*a));
    double *b = malloc(N * sizeof(*b));
    double *c = malloc(N * sizeof(*c));
    double *r = malloc(N * sizeof(*r));
    double *temp = malloc(N * sizeof(*temp));

    double x, y, a_;

    compute_dx(zz, N, dz);
    compute_dfactor(dz, N, dfactor);
    compute_xInt(zz, N, zInt);
    
    for(kk=0; kk < N; kk++)
        V[kk] = Vfunc(zz[kk], nu3);
    for(kk=0; kk < N-1; kk++)
        VInt[kk] = Vfunc(zInt[kk], nu3);

    tridiag_malloc(N);
    for(ii = 0; ii < L; ii++){
        for(jj = 0; jj < M; jj++){
            for(ll = 0; ll < O; ll++){
                x = xx[ii];
                y = yy[jj];
                a_ = aa[ll];

                Mfirst = Mfunc4D(zz[0], x, y, a_, m31, m32, m34, gamma3, h3);
                Mlast = Mfunc4D(zz[N-1], x, y, a_, m31, m32, m34, gamma3, h3);
                for(kk=0; kk < N-1; kk++)
                    MInt[kk] = Mfunc4D(zInt[kk], x, y, a_, m31, m32, m34, gamma3, h3);

                compute_delj(dz, MInt, VInt, N, delj, use_delj_trick);
                compute_abc_nobc(dz, dfactor, delj, MInt, V, dt, N, a, b, c);
                for(kk = 0; kk < N; kk++)
                    r[kk] = phi[ii*M*N*O + jj*N*O + kk*O + ll]/dt;

                if((xx[ii]==0) && (yy[jj]==0) && (aa[ll] == 0) && (Mfirst <= 0))
                    b[0] += (0.5/nu3 - Mfirst)*2./dz[0];
                if((xx[ii]==1) && (yy[jj]==1) && (aa[ll] == 1) && (Mlast >= 0))
                    b[N-1] += -(-0.5/nu3 - Mlast)*2./dz[N-2];

                tridiag_premalloc(a, b, c, r, temp, N);
                for(kk = 0; kk < N; kk++)
                    phi[ii*M*N*O + jj*N*O + kk*O + ll] = temp[kk];
            }
        }
    }
    tridiag_free();

    free(dz);
    free(dfactor);
    free(zInt);
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

void implicit_4Da(double *phi, double *xx, double *yy, double *zz, double *aa,
        double nu4, double m41, double m42, double m43, double gamma4, double h4,
        double dt, int L, int M, int N, int O, int use_delj_trick){
    int ii,jj,kk,ll;

    double *da = malloc((O-1) * sizeof(*da));
    double *dfactor = malloc(O * sizeof(*dfactor));
    double *aInt = malloc((O-1) * sizeof(*aInt));

    double Mfirst, Mlast;

    double *MInt = malloc((O-1) * sizeof(*MInt));
    double *V = malloc(O * sizeof(*V));
    double *VInt = malloc((O-1) * sizeof(*VInt));

    double *delj = malloc((O-1) * sizeof(*delj));

    double *a = malloc(O * sizeof(*a));
    double *b = malloc(O * sizeof(*b));
    double *c = malloc(O * sizeof(*c));
    double *r = malloc(O * sizeof(*r));

    double x, y, z;

    compute_dx(aa, O, da);
    compute_dfactor(da, O, dfactor);
    compute_xInt(aa, O, aInt);
    
    for(ll=0; ll < O; ll++)
        V[ll] = Vfunc(aa[ll], nu4);
    for(ll=0; ll < O-1; ll++)
        VInt[ll] = Vfunc(aInt[ll], nu4);

    tridiag_malloc(O);
    for(ii = 0; ii < L; ii++){
        for(jj = 0; jj < M; jj++){
            for(kk = 0; kk < N; kk++){
                x = xx[ii];
                y = yy[jj];
                z = zz[kk];

                Mfirst = Mfunc4D(aa[0], x, y, z, m41, m42, m43, gamma4, h4);
                Mlast = Mfunc4D(aa[O-1], x, y, z, m41, m42, m43, gamma4, h4);
                for(ll=0; ll < O-1; ll++)
                    MInt[ll] = Mfunc4D(aInt[ll], x, y, z, m41, m42, m43, gamma4, h4);

                compute_delj(da, MInt, VInt, O, delj, use_delj_trick);
                compute_abc_nobc(da, dfactor, delj, MInt, V, dt, O, a, b, c);
                for(ll = 0; ll < O; ll++)
                    r[ll] = phi[ii*M*N*O + jj*N*O + kk*O + ll]/dt;

                if((xx[ii]==0) && (yy[jj]==0) && (zz[kk] == 0) && (Mfirst <= 0))
                    b[0] += (0.5/nu4 - Mfirst)*2./da[0];
                if((xx[ii]==1) && (yy[jj]==1) && (zz[kk] == 1) && (Mlast >= 0))
                    b[O-1] += -(-0.5/nu4 - Mlast)*2./da[O-2];

                tridiag_premalloc(a, b, c, r, &phi[ii*M*N*O + jj*N*O + kk*O], O);
            }
        }
    }
    tridiag_free();

    free(da);
    free(dfactor);
    free(aInt);
    free(MInt);
    free(V);
    free(VInt);
    free(delj);
    free(a);
    free(b);
    free(c);
    free(r);
}