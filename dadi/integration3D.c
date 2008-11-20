#include "integration_shared.h"
#include "tridiag.h"

void implicit_3Dx(int L, int M, int N;
        double phi[L][M][N], 
        double xx[L], double yy[M], double zz[N],
        double nu1, double m12, double m13, double gamma1, double h1,
        double dt, int L, int M, int N, int use_delj_trick){
    int ii,jj,kk;

    double dx[L-1], dfactor[L], xInt[L-1];
    compute_dx(xx, L, dx);
    compute_dfactor(dx, L, dfactor);
    compute_xInt(xx, L, xInt);
    
    double Mfirst, Mlast;
    double MInt[L-1], V[L], VInt[L-1];
    double delj[L-1];
    for(ii=0; ii < L; ii++)
        V[ii] = Vfunc(xx[ii], nu1);
    for(ii=0; ii < L-1; ii++)
        VInt[ii] = Vfunc(xInt[ii], nu1);

    double a[L], b[L], c[L], r[L], temp[L];
    double y, z;
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
                r[ii] = phi[ii][jj][kk]/dt;

            if((jj==0) && (kk==0) && (Mfirst <= 0))
                b[0] += (0.5/nu1 - Mfirst)*2./dx[0];
            if((jj==M-1) && (kk==N-1) && (Mlast >= 0))
                b[L-1] += -(-0.5/nu1 - Mlast)*2./dx[L-2];

            tridiag(a, b, c, r, temp, L);
            for(ii = 0; ii < L; ii++)
                phi[ii][jj][kk] = temp[ii];
        }
    }
}

void implicit_3Dy(int L, int M, int N;
        double phi[L][M][N], 
        double xx[L], double yy[M], double zz[N],
        double nu2, double m21, double m23, double gamma2, double h2,
        double dt, int L, int M, int N, int use_delj_trick){
    int ii,jj,kk;

    double dy[M-1], dfactor[M], yInt[M-1];
    compute_dx(yy, M, dy);
    compute_dfactor(dy, M, dfactor);
    compute_xInt(yy, M, yInt);
    
    double Mfirst, Mlast;
    double MInt[M-1], V[M], VInt[M-1];
    double delj[M-1];
    for(jj=0; jj < M; jj++)
        V[jj] = Vfunc(yy[jj], nu2);
    for(jj=0; jj < M-1; jj++)
        VInt[jj] = Vfunc(yInt[jj], nu2);

    double a[M], b[M], c[M], r[M], temp[M];
    double x, z;
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
                r[jj] = phi[ii][jj][kk]/dt;

            if((ii==0) && (kk==0) && (Mfirst <= 0))
                b[0] += (0.5/nu2 - Mfirst)*2./dy[0];
            if((ii==L-1) && (kk==N-1) && (Mlast >= 0))
                b[M-1] += -(-0.5/nu2 - Mlast)*2./dy[M-2];

            tridiag(a, b, c, r, temp, M);
            for(jj = 0; jj < M; jj++)
                phi[ii][jj][kk] = temp[jj];
        }
    }
}

void implicit_3Dz(int L, int M, int N;
        double phi[L][M][N], 
        double xx[L], double yy[M], double zz[N],
        double nu3, double m31, double m32, double gamma3, double h3,
        double dt, int L, int M, int N, int use_delj_trick){
    int ii,jj,kk;

    double dz[N-1], dfactor[N], zInt[N-1];
    compute_dx(zz, N, dz);
    compute_dfactor(dz, N, dfactor);
    compute_xInt(zz, N, zInt);
    
    double Mfirst, Mlast;
    double MInt[N-1], V[N], VInt[N-1];
    double delj[N-1];
    for(kk=0; kk < N; kk++)
        V[kk] = Vfunc(zz[kk], nu3);
    for(kk=0; kk < N-1; kk++)
        VInt[kk] = Vfunc(zInt[kk], nu3);

    double a[N], b[N], c[N], r[N];
    double x, y;
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
                r[kk] = phi[ii][jj][kk]/dt;

            if((ii==0) && (jj==0) && (Mfirst <= 0))
                b[0] += (0.5/nu3 - Mfirst)*2./dz[0];
            if((ii==L-1) && (jj==M-1) && (Mlast >= 0))
                b[N-1] += -(-0.5/nu3 - Mlast)*2./dz[N-2];

            tridiag(a, b, c, r, phi[ii][jj], N);
        }
    }
}

void implicit_precalc_3Dx(int L, int M, int N;
        double phi[L][M][N], 
        double ax[L][M][N], double bx[L][M][N], double cx[L][M][N],
        double dt, int L, int M, int N){
    int ii,jj,kk;

    double a[L], b[L], c[L], r[L];
    double new_row[L];

    for(jj = 0; jj < M; jj++){
        for(kk = 0; kk < N; kk++){
            for(ii = 0; ii < L; ii++){
                a[ii] = ax[ii][jj][kk];
                b[ii] = bx[ii][jj][kk] + 1/dt;
                c[ii] = cx[ii][jj][kk];
                r[ii] = 1/dt * phi[ii][jj][kk];
            }

            tridiag(a, b, c, r, new_row, L);
            for(ii = 0; ii < L; ii++)
                phi[ii][jj][kk] = new_row[ii];
        }
    }
}

void implicit_precalc_3Dy(int L, int M, int N;
        double phi[L][M][N], 
        double ay[L][M][N], double by[L][M][N], double cy[L][M][N],
        double dt, int L, int M, int N){
    int ii,jj,kk;

    double a[M], b[M], c[M], r[M];
    double new_row[M];

    for(ii = 0; ii < L; ii++){
        for(kk = 0; kk < N; kk++){
            for(jj = 0; jj < M; jj++){
                a[jj] = ay[ii][jj][kk];
                b[jj] = by[ii][jj][kk] + 1/dt;
                c[jj] = cy[ii][jj][kk];
                r[jj] = 1/dt * phi[ii][jj][kk];
            }

            tridiag(a, b, c, r, new_row, M);
            for(jj = 0; jj < M; jj++)
                phi[ii][jj][kk] = new_row[jj];
        }
    }
}

void implicit_precalc_3Dz(int L, int M, int N;
        double phi[L][M][N], 
        double az[L][M][N], double bz[L][M][N], double cz[L][M][N],
        double dt, int L, int M, int N){
    int ii,jj,kk;

    double a[N], b[N], c[N], r[N];

    for(ii = 0; ii < L; ii++){
        for(jj = 0; jj < M; jj++){
            for(kk = 0; kk < N; kk++){
                a[kk] = az[ii][jj][kk];
                b[kk] = bz[ii][jj][kk] + 1/dt;
                c[kk] = cz[ii][jj][kk];
                r[kk] = 1/dt * phi[ii][jj][kk];
            }

            tridiag(a, b, c, r, phi[ii][jj], N);
        }
    }
}
