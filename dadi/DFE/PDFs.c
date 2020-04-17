#include <stdlib.h>
#include <math.h>
/* Define M_PI here, since it seems undefined in some Microsoft compilers */
#define M_PI 3.14159265358979323846264338327950288

void biv_lognormal(double *xx, double *yy, double *params, int n, int m,
                   int Nparams, double *output);
double gamma_func(double z);
void biv_ind_gamma(double *xx, double *yy, double *params, int n, int m,
                   int Nparams, double *output);

void biv_lognormal(double *xx, double *yy, double *params, int n, int m,
                   int Nparams, double *output){
    double mu1,mu2,sigma1,sigma2,rho;
    double *delx, *dely;
    int ii, jj;
    double norm, q, pre;

    mu1=0;mu2=0;sigma1=0;sigma2=0;rho=0;
    if (Nparams == 3){
        mu1 = params[0];
        mu2 = params[0];
        sigma1 = params[1];
        sigma2 = params[1];
        rho = params[2];
    } else if (Nparams == 5){
        mu1 = params[0];
        mu2 = params[1];
        sigma1 = params[2];
        sigma2 = params[3];
        rho = params[4];
    }
    delx = malloc(n * sizeof(*xx));
    dely = malloc(m * sizeof(*yy));

    for(ii=0; ii<n; ii++){
        delx[ii] = (log(xx[ii]) - mu1)/sigma1;
    }
    for(jj=0; jj<m; jj++){
        dely[jj] = (log(yy[jj]) - mu2)/sigma2;
    }

    pre = 2*M_PI * sigma1*sigma2 * sqrt(1.-rho*rho);
    for(ii=0; ii<n; ii++){
        for (jj=0; jj<m; jj++){
            norm = pre * xx[ii]*yy[jj];
            q = (delx[ii]*delx[ii] - 2.*rho*delx[ii]*dely[jj] + dely[jj]*dely[jj])/(1.-rho*rho);
            output[ii*m+jj] = exp(-q/2.)/norm;
        }
    }

    free(delx);
    free(dely);
}

double gamma_func(double z){
    /* Based on code at https://en.wikipedia.org/wiki/Lanczos_approximation */
    int ii;
    double t,x,y;
    double p[8] = {676.5203681218851,
	    -1259.1392167224028,
	    771.32342877765313,
	    -176.61502916214059,
	    12.507343278686905,
	    -0.13857109526572012,
	    9.9843695780195716e-6,
	    1.5056327351493116e-7
    };

    if (z < 0.5){
        y = M_PI / (sin(M_PI*z) * gamma_func(1.-z));
    } else {
        z -= 1;
        x = 0.99999999999980993;
        for (ii=0; ii<8; ii++){
            x += p[ii] / (z+ii+1);
	}
        t = z + 8 - 0.5;
        y = sqrt(2*M_PI) * pow(t, z+0.5) * exp(-t) * x;
    }
    return y;
}

void biv_ind_gamma(double *xx, double *yy, double *params, int n, int m,
                   int Nparams, double *output){
    double alpha1, alpha2, beta1, beta2;
    double *margx, *margy;
    int ii, jj;
    double cx, cy;

    alpha1=0; alpha2=0; beta1=0; beta2=0;
    if (Nparams == 2 || Nparams == 3){
        alpha1 = params[0];
        alpha2 = params[0];
        beta1 = params[1];
        beta2 = params[1];
    } else if (Nparams == 4 || Nparams == 5){
        alpha1 = params[0];
        alpha2 = params[1];
        beta1 = params[2];
        beta2 = params[3];
    }

    margx = malloc(n * sizeof(*xx));
    margy = malloc(m * sizeof(*yy));

    cx = pow(beta1, alpha1) * gamma_func(alpha1);
    for(ii=0; ii<n; ii++){
        margx[ii] = pow(xx[ii], alpha1-1.) * exp(-xx[ii]/beta1) / cx;
    }
    cy = pow(beta2, alpha2) * gamma_func(alpha2);
    for(jj=0; jj<m; jj++){
        margy[jj] = pow(yy[jj], alpha2-1.) * exp(-yy[jj]/beta2) / cy;
    }

    for(ii=0; ii<n; ii++){
        for (jj=0; jj<m; jj++){
            output[ii*m+jj] = margx[ii]*margy[jj];
        }
    }

    free(margx);
    free(margy);
}
