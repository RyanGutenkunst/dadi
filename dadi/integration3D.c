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

void implicit_3Dx(double *phi, double *xx, double *yy, double *zz,
        double nu1, double m12, double m13, double gamma1, double h1,
        double dt, int L, int M, int N, int use_delj_trick){
    int ii,jj,kk;

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

    double y, z;

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
            y = yy[jj];
            z = zz[kk];

            Mfirst = Mfunc3D(xx[0], y, z, m12, m13, gamma1, h1);
            Mlast = Mfunc3D(xx[L-1], y, z, m12, m13, gamma1, h1);
            for(ii=0; ii < L-1; ii++)
                MInt[ii] = Mfunc3D(xInt[ii], y, z, m12, m13, gamma1, h1);

            compute_delj(dx, MInt, VInt, L, delj, use_delj_trick);
            compute_abc_nobc(dx, dfactor, delj, MInt, V, dt, L, a, b, c);
            for(ii = 0; ii < L; ii++)
                r[ii] = phi[ii*M*N + jj*N + kk]/dt;

            if((jj==0) && (kk==0) && (Mfirst <= 0))
                b[0] += (0.5/nu1 - Mfirst)*2./dx[0];
            if((jj==M-1) && (kk==N-1) && (Mlast >= 0))
                b[L-1] += -(-0.5/nu1 - Mlast)*2./dx[L-2];

            tridiag_premalloc(a, b, c, r, temp, L);
            for(ii = 0; ii < L; ii++)
                phi[ii*M*N + jj*N + kk] = temp[ii];
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

void implicit_3Dy(double *phi, double *xx, double *yy, double *zz,
        double nu2, double m21, double m23, double gamma2, double h2,
        double dt, int L, int M, int N, int use_delj_trick){
    int ii,jj,kk;

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

    double x, z;

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
            x = xx[ii];
            z = zz[kk];

            Mfirst = Mfunc3D(yy[0], x, z, m21, m23, gamma2, h2);
            Mlast = Mfunc3D(yy[M-1], x, z, m21, m23, gamma2, h2);
            for(jj=0; jj < M-1; jj++)
                MInt[jj] = Mfunc3D(yInt[jj], x, z, m21, m23, gamma2, h2);

            compute_delj(dy, MInt, VInt, M, delj, use_delj_trick);
            compute_abc_nobc(dy, dfactor, delj, MInt, V, dt, M, a, b, c);
            for(jj = 0; jj < M; jj++)
                r[jj] = phi[ii*M*N + jj*N + kk]/dt;

            if((ii==0) && (kk==0) && (Mfirst <= 0))
                b[0] += (0.5/nu2 - Mfirst)*2./dy[0];
            if((ii==L-1) && (kk==N-1) && (Mlast >= 0))
                b[M-1] += -(-0.5/nu2 - Mlast)*2./dy[M-2];

            tridiag_premalloc(a, b, c, r, temp, M);
            for(jj = 0; jj < M; jj++)
                phi[ii*M*N + jj*N + kk] = temp[jj];
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

void implicit_3Dz(double *phi, double *xx, double *yy, double *zz,
        double nu3, double m31, double m32, double gamma3, double h3,
        double dt, int L, int M, int N, int use_delj_trick){
    int ii,jj,kk;

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

    double x, y;

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
            x = xx[ii];
            y = yy[jj];

            Mfirst = Mfunc3D(zz[0], x, y, m31, m32, gamma3, h3);
            Mlast = Mfunc3D(zz[N-1], x, y, m31, m32, gamma3, h3);
            for(kk=0; kk < N-1; kk++)
                MInt[kk] = Mfunc3D(zInt[kk], x, y, m31, m32, gamma3, h3);

            compute_delj(dz, MInt, VInt, N, delj, use_delj_trick);
            compute_abc_nobc(dz, dfactor, delj, MInt, V, dt, N, a, b, c);
            for(kk = 0; kk < N; kk++)
                r[kk] = phi[ii*M*N + jj*N + kk]/dt;

            if((ii==0) && (jj==0) && (Mfirst <= 0))
                b[0] += (0.5/nu3 - Mfirst)*2./dz[0];
            if((ii==L-1) && (jj==M-1) && (Mlast >= 0))
                b[N-1] += -(-0.5/nu3 - Mlast)*2./dz[N-2];

            tridiag_premalloc(a, b, c, r, &phi[ii*M*N + jj*N], N);
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

void implicit_precalc_3Dx(double *phi, double *ax, double *bx, double *cx,
        double dt, int L, int M, int N){
    int ii,jj,kk;
    int index;

    double *a = malloc(L * sizeof(*a));
    double *b = malloc(L * sizeof(*b));
    double *c = malloc(L * sizeof(*c));
    double *r = malloc(L * sizeof(*r));
    double *new_row = malloc(L * sizeof(*new_row));

    tridiag_malloc(L);
    for(jj = 0; jj < M; jj++){
        for(kk = 0; kk < N; kk++){
            for(ii = 0; ii < L; ii++){
                index = ii*M*N + jj*N + kk;
                a[ii] = ax[index];
                b[ii] = bx[index] + 1/dt;
                c[ii] = cx[index];
                r[ii] = 1/dt * phi[index];
            }

            tridiag_premalloc(a, b, c, r, new_row, L);
            for(ii = 0; ii < L; ii++)
                phi[ii*M*N + jj*N + kk] = new_row[ii];
        }
    }
    tridiag_free();

    free(a);
    free(b);
    free(c);
    free(r);
    free(new_row);
}

void implicit_precalc_3Dy(double *phi, double *ay, double *by, double *cy,
        double dt, int L, int M, int N){
    int ii,jj,kk;
    int index;

    double *a = malloc(M * sizeof(*a));
    double *b = malloc(M * sizeof(*b));
    double *c = malloc(M * sizeof(*c));
    double *r = malloc(M * sizeof(*r));
    double *new_row = malloc(M * sizeof(*new_row));

    tridiag_malloc(M);
    for(ii = 0; ii < L; ii++){
        for(kk = 0; kk < N; kk++){
            for(jj = 0; jj < M; jj++){
                index = ii*M*N + jj*N + kk;
                a[jj] = ay[index];
                b[jj] = by[index] + 1/dt;
                c[jj] = cy[index];
                r[jj] = 1/dt * phi[index];
            }

            tridiag_premalloc(a, b, c, r, new_row, M);
            for(jj = 0; jj < M; jj++)
                phi[ii*M*N + jj*N + kk] = new_row[jj];
        }
    }
    tridiag_free();

    free(a);
    free(b);
    free(c);
    free(r);
    free(new_row);
}

void implicit_precalc_3Dz(double *phi, double *az, double *bz, double *cz,
        double dt, int L, int M, int N){
    int ii,jj,kk;
    int index;

    double *a = malloc(N * sizeof(*a));
    double *b = malloc(N * sizeof(*b));
    double *c = malloc(N * sizeof(*c));
    double *r = malloc(N * sizeof(*r));
    double *new_row = malloc(N * sizeof(*new_row));

    tridiag_malloc(N);
    for(ii = 0; ii < L; ii++){
        for(jj = 0; jj < M; jj++){
            for(kk = 0; kk < N; kk++){
                index = ii*M*N + jj*N + kk;
                a[kk] = az[index];
                b[kk] = bz[index] + 1/dt;
                c[kk] = cz[index];
                r[kk] = 1/dt * phi[index];
            }

            tridiag_premalloc(a, b, c, r, &phi[ii*M*N + jj*N], N);
        }
    }
    tridiag_free();

    free(a);
    free(b);
    free(c);
    free(r);
    free(new_row);
}
