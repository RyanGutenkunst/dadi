#include <stdio.h>
#include <math.h>

double Vfunc(double x, double nu){
    return 1./nu * x*(1.-x);
}

double Mfunc1D(double x, double gamma){
    return gamma * x*(1.-x);
}
double Mfunc2D(double x, double y, double m, double gamma){
    return m * (y-x) + gamma * x*(1.-x);
}
double Mfunc3D(double x, double y, double z, double mxy, double mxz,
        double gamma){
    return mxy * (y-x) + mxz * (z-x) + gamma * x*(1.-x);
}


void compute_dx(int N; double xx[N], int N, double dx[N-1]){
    int ii;
    for(ii = 0; ii < N-1; ii++)
        dx[ii] = xx[ii+1]-xx[ii];
}

void compute_dfactor(int N; double dx[N-1], int N, double dfactor[N]){
    int ii;
    for(ii=1; ii < N-1; ii++)
        dfactor[ii] = 2./(dx[ii] + dx[ii-1]);    
    dfactor[0] = 2./dx[0];
    dfactor[N-1] = 2./dx[N-2];
}

void compute_xInt(int N; double xx[N], int N, double xInt[N-1]){
    int ii;
    for(ii = 0; ii < N-1; ii++)
        xInt[ii] = 0.5*(xx[ii+1]+xx[ii]);
}

void compute_delj(int N; double dx[N-1], double MInt[N-1], double VInt[N-1],
        int N, double delj[N-1], int use_delj_trick){
    int ii;
    double wj, epsj;
    if(!use_delj_trick){
        for(ii=0; ii < N-1; ii++)
            delj[ii] = 0.5;
        return;
    }

    for(ii=0; ii < N-1; ii++){
        wj = 2 * MInt[ii] * dx[ii];
        epsj = exp(wj/VInt[ii]);
        if((epsj != 1.0) && (wj != 0))
            delj[ii] = (-epsj*wj + epsj*VInt[ii] - VInt[ii])/(wj - epsj*wj);
        else
            delj[ii] = 0.5;
    }
}

void compute_abc_nobc(int N; double dx[N-1], double dfactor[N], 
        double delj[N-1], double MInt[N-1], double V[N], double dt, int N,
        double a[N], double b[N], double c[N]){
    int ii;
    a[0] = 0;
    c[N-1] = 0;
    for(ii = 0; ii < N; ii++)
        b[ii] = 1./dt;

    // Using atemp and ctemp yields an ~10% speed-up.
    double atemp, ctemp;
    for(ii = 0; ii < N-1; ii++){
        atemp = MInt[ii] * delj[ii] + V[ii]/(2*dx[ii]);
        a[ii+1] = -dfactor[ii+1]*atemp;
        b[ii] += dfactor[ii]*atemp;

        ctemp = -MInt[ii] * (1 - delj[ii]) + V[ii+1]/(2*dx[ii]);
        b[ii+1] += dfactor[ii+1]*ctemp;
        c[ii] = -dfactor[ii]*ctemp;
    }
}
