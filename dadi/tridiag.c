#include "tridiag.h"

void tridiag(int n;
        double a[n], double b[n], double c[n], double r[n], double u[n], int n){
    /*
    Based on Numerical Recipes in C tridiag function.
    */
    double gam[n];

    double bet = b[0];
    u[0] = r[0]/bet;

    int j;
    for(j=1; j <= n-1; j++){
        gam[j] = c[j-1]/bet;
        bet = b[j] - a[j]*gam[j];
        u[j] = (r[j]-a[j]*u[j-1])/bet;
    }
    
    for(j=(n-2); j >= 0; j--){
        u[j] -= gam[j+1]*u[j+1];
    }
}

void tridiag_fl(float a[], float b[], float c[], float r[], float u[], int n){
    /*
    Based on Numerical Recipes in C tridiag function.
    */
    float gam[n];

    float bet = b[0];
    u[0] = r[0]/bet;

    int j;
    for(j=1; j <= n-1; j++){
        gam[j] = c[j-1]/bet;
        bet = b[j] - a[j]*gam[j];
        u[j] = (r[j]-a[j]*u[j-1])/bet;
    }
    
    for(j=(n-2); j >= 0; j--){
        u[j] -= gam[j+1]*u[j+1];
    }
}
