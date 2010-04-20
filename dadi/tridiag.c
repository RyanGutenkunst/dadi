#include <stdlib.h>
#include "tridiag.h"

double *gam;

void tridiag_malloc(int n){
    gam = malloc(n * sizeof(*gam));
}

void tridiag_free(){
    free(gam);
}

void tridiag_premalloc(double a[], double b[], double c[], double r[], double u[], int n){
    /*
    Based on Numerical Recipes in C tridiag function.

    This version can re-use dynamically allocated memory in the global gam
    variable.
    */
    double bet = b[0];
    int j;

    u[0] = r[0]/bet;
    for(j=1; j <= n-1; j++){
        gam[j] = c[j-1]/bet;
        bet = b[j] - a[j]*gam[j];
        u[j] = (r[j]-a[j]*u[j-1])/bet;
    }
    
    for(j=(n-2); j >= 0; j--){
        u[j] -= gam[j+1]*u[j+1];
    }
}

void tridiag(double a[], double b[], double c[], double r[], double u[], int n){
    tridiag_malloc(n);
    tridiag_premalloc(a,b,c,r,u,n);
    tridiag_free();
}

void tridiag_fl(float a[], float b[], float c[], float r[], float u[], int n){
    /*
    Based on Numerical Recipes in C tridiag function.
    */
    float *gam = malloc(n * sizeof(*gam));
    float bet = b[0];
    int j;

    u[0] = r[0]/bet;
    for(j=1; j <= n-1; j++){
        gam[j] = c[j-1]/bet;
        bet = b[j] - a[j]*gam[j];
        u[j] = (r[j]-a[j]*u[j-1])/bet;
    }
    
    for(j=(n-2); j >= 0; j--){
        u[j] -= gam[j+1]*u[j+1];
    }

    free(gam);
}
