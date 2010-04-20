#include <stdio.h>
#include <math.h>

double Vfunc(double x, double nu){
    return 1./nu * x*(1.-x);
}

double Mfunc1D(double x, double gamma, double h){
    return gamma * 2*(h + (1.-2*h)*x) * x*(1.-x);
}
double Mfunc2D(double x, double y, double m, double gamma, double h){
    return m * (y-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x);
}
double Mfunc3D(double x, double y, double z, double mxy, double mxz,
        double gamma, double h){
    return mxy * (y-x) + mxz * (z-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x);
}


void compute_dx(double *xx, int N, double *dx){
    int ii;
    for(ii = 0; ii < N-1; ii++)
        dx[ii] = xx[ii+1]-xx[ii];
}

void compute_dfactor(double *dx, int N, double *dfactor){
    int ii;
    for(ii=1; ii < N-1; ii++)
        dfactor[ii] = 2./(dx[ii] + dx[ii-1]);    
    dfactor[0] = 2./dx[0];
    dfactor[N-1] = 2./dx[N-2];
}

void compute_xInt(double *xx, int N, double *xInt){
    int ii;
    for(ii = 0; ii < N-1; ii++)
        xInt[ii] = 0.5*(xx[ii+1]+xx[ii]);
}

void compute_delj(double *dx, double *MInt, double *VInt,
        int N, double *delj, int use_delj_trick){
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

void compute_abc_nobc(double *dx, double *dfactor, 
        double *delj, double *MInt, double *V, double dt, int N,
        double *a, double *b, double *c){
    int ii;
    double atemp, ctemp;

    a[0] = 0;
    c[N-1] = 0;
    for(ii = 0; ii < N; ii++)
        b[ii] = 1./dt;

    /* Using atemp and ctemp yields an ~10% speed-up. */
    for(ii = 0; ii < N-1; ii++){
        atemp = MInt[ii] * delj[ii] + V[ii]/(2*dx[ii]);
        a[ii+1] = -dfactor[ii+1]*atemp;
        b[ii] += dfactor[ii]*atemp;

        ctemp = -MInt[ii] * (1 - delj[ii]) + V[ii+1]/(2*dx[ii]);
        b[ii+1] += dfactor[ii+1]*ctemp;
        c[ii] = -dfactor[ii]*ctemp;
    }
}
