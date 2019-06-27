#include <stdlib.h>
#include <math.h>

void biv_lognormal(double *xx, double *yy, double *params, int n, int m,
                   int Nparams, double *output){
    double mu1=0,mu2=0,sigma1=0,sigma2=0,rho=0;
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

    double *delx = malloc(n * sizeof(*xx));
    double *dely = malloc(m * sizeof(*yy));
    int ii,jj;
    for(ii=0; ii<n; ii++){
        delx[ii] = (log(xx[ii]) - mu1)/sigma1;
    }
    for(jj=0; jj<m; jj++){
        dely[jj] = (log(yy[jj]) - mu2)/sigma2;
    }

    double norm, q;
    double pre = 2*M_PI * sigma1*sigma2 * sqrt(1.-rho*rho);
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

void biv_ind_gamma(double *xx, double *yy, double *params, int n, int m,
                   int Nparams, double *output){
    double alpha1=0, alpha2=0, beta1=0, beta2=0;                   
    if (Nparams == 2){
        alpha1 = params[0];
        alpha2 = params[0];
        beta1 = params[1];
        beta2 = params[1];
    } else if (Nparams == 4){
        alpha1 = params[0];
        alpha2 = params[1];
        beta1 = params[2];
        beta2 = params[3];
    }

    double *margx = malloc(n * sizeof(*xx));
    double *margy = malloc(m * sizeof(*yy));
    int ii,jj;
    double cx = pow(beta1, alpha1) * tgamma(alpha1);
    for(ii=0; ii<n; ii++){
        margx[ii] = pow(xx[ii], alpha1-1.) * exp(-xx[ii]/beta1) / cx;
    }
    double cy = pow(beta2, alpha2) * tgamma(alpha2);
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
