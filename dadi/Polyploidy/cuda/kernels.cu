/*
* functions to inject mutations into the phi matrix
*/ 
__global__ void inject_mutations_2D(double *phi, int L, double val01, double val10){
    phi[1] += val01;
    phi[L] += val10;
}

__global__ void inject_mutations_3D(double *phi, int L, double val001, double val010, double val100){
    phi[1] += val001;
    phi[L] += val010;
    phi[L*L] += val100;
}

__global__ void inject_mutations_4D(double *phi, int L, double val0001, double val0010, double val0100, double val1000){
    phi[1] += val0001;
    phi[L] += val0010;
    phi[L*L] += val0100;
    phi[L*L*L] += val1000;
}

__global__ void inject_mutations_5D(double *phi, int L, double val00001, double val00010, double val00100, double val01000, double val10000){
    phi[1] += val00001;
    phi[L] += val00010;
    phi[L*L] += val00100;
    phi[L*L*L] += val01000;
    phi[L*L*L*L] += val10000;
}

/* 
* a few kernels from Ryan's code for memory management and computing the a, b, and c matrices for the tridiagonal solver
*/

// We need an additional simple kernel to zero out the necessary
// values of the c array, because the Interleaved tridiagonal
// solver alters the c array.
__global__ void cx0(double *cx, int L, int M){
    int jj = blockIdx.x*blockDim.x + threadIdx.x;
    if(jj < M){
        cx[(L-1)*M + jj] = 0;
    }
}

// Compared to the C code, we need to separate ab and bc
// calculations, to avoid a race condition due to multiple
// theads writing to the same elements of b.
__global__ void compute_ab_nobc(double *dx, double *dfactor, 
        double *MInt, double *V, double dt, int L, int M,
        double *a, double *b){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    double atemp;

    if(ii < L-1){
        atemp = MInt[ii*M + jj] * 0.5 + V[ii]/(2*dx[ii]);
        a[(ii+1)*M + jj] = -dfactor[ii+1]*atemp;
        b[ii*M + jj] += dfactor[ii]*atemp;
    }
}

__global__ void compute_bc_nobc(double *dx, double *dfactor, 
        double *MInt, double *V, double dt, int L, int M,
        double *b, double *c){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    double ctemp;

    if(ii < L-1){
        ctemp = -MInt[ii*M + jj] * 0.5 + V[ii+1]/(2*dx[ii]);
        b[(ii+1)*M + jj] += dfactor[ii+1]*ctemp;
        c[ii*M + jj] = -dfactor[ii]*ctemp;
    }
}

/*
* functions to calculate the variance
*/

__global__ void Vfunc(double *x, double nu, int L, double *output){
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if(ii < L){
        output[ii] =  x[ii] * (1.-x[ii])/nu;
    }
}

__global__ void Vfunc_tetra(double *x, double nu, int L, double *output){
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if(ii < L){
        output[ii] =  x[ii] * (1.-x[ii])/(2*nu);
    }
}

__global__ void Vfunc_hex(double *x, double nu, int L, double *output){
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if(ii < L){
        output[ii] =  x[ii] * (1.-x[ii])/(3*nu);
    }
}

/*
* mean functions and BC terms for DIPLOIDS
*/

__device__ double _Mfunc2D(double x, double y, double mxy, double gamma, double h){
    return mxy * (y-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x);
}

__global__ void Mfunc2D(double *x, double *y, double mxy, double gamma, double h, int L, int M, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    if(ii < L){
        output[ii*M + jj] = _Mfunc2D(x[ii], y[jj], mxy, gamma, h);
    }
}

__device__ double _Mfunc3D(double x, double y, double z, double mxy, double mxz,
        double gamma, double h){
    return mxy * (y-x) + mxz * (z-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x);
}

__global__ void Mfunc3D(double *x, double *y, double *z, double mxy, double mxz, double gamma, double h, int L, int M, int N, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / N) % M;
    int kk = (blockIdx.x*blockDim.x + threadIdx.x) % N;
    if(ii < L){
        output[ii*(M*N) + jj*N + kk] = _Mfunc3D(x[ii], y[jj], z[kk], mxy, mxz, gamma, h);
    }
}

__device__ double _Mfunc4D(double x, double y, double z, double a, double mxy, double mxz, double mxa,
        double gamma, double h){
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x);
}

__global__ void Mfunc4D(double *x, double *y, double *z, double *a, double mxy, double mxz, double mxa, double gamma, double h, int L, int M, int N, int O, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / O) % N;
    int ll = (blockIdx.x*blockDim.x + threadIdx.x) % O;
    if(ii < L){
        output[ii*M*N*O + jj*N*O + kk*O + ll] = _Mfunc4D(x[ii], y[jj], z[kk], a[ll], mxy, mxz, mxa, gamma, h);
    }
}

__device__ double _Mfunc5D(double x, double y, double z, double a, double b, double mxy, double mxz, double mxa, double mxb,
        double gamma, double h){
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x);
}

__global__ void Mfunc5D(double *x, double *y, double *z, double *a, double *b, double mxy, double mxz, double mxa, double mxb, double gamma, double h, int L, int M, int N, int O, int P, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O*P);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O*P)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / (O*P)) % N;
    int ll = ((blockIdx.x*blockDim.x + threadIdx.x) / P) % O;
    int mm = (blockIdx.x*blockDim.x + threadIdx.x) % P;
    if(ii < L){
        output[ii*M*N*O*P + jj*N*O*P + kk*O*P + ll*P + mm] = _Mfunc5D(x[ii], y[jj], z[kk], a[ll], b[mm], mxy, mxz, mxa, mxb, gamma, h);
    }
}

// This function works for all dimensions, because migration terms
// don't matter at the 0,0,0 and 1,1,1 corners of the regime.
__global__ void include_bc(double*dx, double nu1, double gamma, double h, int L, int M, double *b){
    double Mfirst, Mlast;
    // 0,0 entry
    Mfirst = _Mfunc2D(0, 0, 0, gamma, h);
    if(Mfirst <= 0){
        b[0] += (0.5/nu1 - Mfirst)*2./dx[0];
    }
    // -1,-1 entry
    Mlast = _Mfunc2D(1, 1, 0, gamma, h);
    if(Mlast >= 0){
        b[L*M-1] += -(-0.5/nu1 - Mlast)*2./dx[L-2];
    }
}


/*
* mean functions and BC terms for AUTOTETRAPLOIDS
*/

__device__ double _Mfunc2D_auto(double x, double y, double mxy, double gam1, double gam2, double gam3, double gam4){
    double poly = (((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                    (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                    (-6.*gam1 + 3.*gam2)) * x +
                    gam1;
    return mxy * (y-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc2D_auto(double *x, double *y, double mxy, double gam1, double gam2, double gam3, double gam4, int L, int M, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    if(ii < L){
        output[ii*M + jj] = _Mfunc2D_auto(x[ii], y[jj], mxy, gam1, gam2, gam3, gam4);
    }
}

__device__ double _Mfunc3D_auto(double x, double y, double z, double mxy, double mxz, double gam1, double gam2, double gam3, double gam4){
    double poly = (((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                    (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                    (-6.*gam1 + 3.*gam2)) * x +
                    gam1;
    return mxy * (y-x) + mxz * (z-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc3D_auto(double *x, double *y, double *z, double mxy, double mxz, double gam1, double gam2, double gam3, double gam4, int L, int M, int N, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / N) % M;
    int kk = (blockIdx.x*blockDim.x + threadIdx.x) % N;
    if(ii < L){
        output[ii*(M*N) + jj*N + kk] = _Mfunc3D_auto(x[ii], y[jj], z[kk], mxy, mxz, gam1, gam2, gam3, gam4);
    }
}

__device__ double _Mfunc4D_auto(double x, double y, double z, double a, double mxy, double mxz, double mxa, 
                                double gam1, double gam2, double gam3, double gam4){
    double poly = (((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                    (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                    (-6.*gam1 + 3.*gam2)) * x +
                    gam1;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc4D_auto(double *x, double *y, double *z, double *a, double mxy, double mxz, double mxa, double gam1, double gam2, double gam3, double gam4, int L, int M, int N, int O, double *output){   
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / O) % N;
    int ll = (blockIdx.x*blockDim.x + threadIdx.x) % O;
    if(ii < L){
        output[ii*M*N*O + jj*N*O + kk*O + ll] = _Mfunc4D_auto(x[ii], y[jj], z[kk], a[ll], mxy, mxz, mxa, gam1, gam2, gam3, gam4);
    }
}

__device__ double _Mfunc5D_auto(double x, double y, double z, double a, double b, double mxy, double mxz, double mxa, double mxb,
                                double gam1, double gam2, double gam3, double gam4){
    double poly = (((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                    (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                    (-6.*gam1 + 3.*gam2)) * x +
                    gam1;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc5D_auto(double *x, double *y, double *z, double *a, double *b, double mxy, double mxz, double mxa, double mxb, double gam1, double gam2, double gam3, double gam4, int L, int M, int N, int O, int P, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O*P);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O*P)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / (O*P)) % N;
    int ll = ((blockIdx.x*blockDim.x + threadIdx.x) / P) % O;
    int mm = (blockIdx.x*blockDim.x + threadIdx.x) % P;
    if(ii < L){
        output[ii*M*N*O*P + jj*N*O*P + kk*O*P + ll*P + mm] = _Mfunc5D_auto(x[ii], y[jj], z[kk], a[ll], b[mm], mxy, mxz, mxa, mxb, gam1, gam2, gam3, gam4);
    }
}

// This function works for all dimensions, because migration terms
// don't matter at the 0,0,0 and 1,1,1 corners of the regime.
__global__ void include_bc_auto(double*dx, double nu1, double gam1, double gam2, double gam3, double gam4, int L, int M, double *b){
    double Mfirst, Mlast;
    // 0,0 entry
    Mfirst = _Mfunc2D_auto(0, 0, 0, gam1, gam2, gam3, gam4);
    if(Mfirst <= 0){
        b[0] += (0.25/nu1 - Mfirst)*2./dx[0];
    }
    // -1,-1 entry
    Mlast = _Mfunc2D_auto(1, 1, 0, gam1, gam2, gam3, gam4);
    if(Mlast >= 0){
        b[L*M-1] += -(-0.25/nu1 - Mlast)*2./dx[L-2];
    }
}

/*
* mean functions and BC terms for ALLOTETRAPLOID SUBGENOME A
*/

/*
* These functions are a bit different from the CPU code because we need to transpose the GPU array 
*       between directions of the ADI scheme to use the much faster interleaved tridiagonal solver.
* So, we can't interchange the order in which params (i.e. mij's) are passed to these functions later on. 
* Instead, we account for the order of params when we write the functions. 
* Note that this is really only a problem for the b subgenome code in 3+ dimensions.
* 
* To help a bit with readability, I use e to denote HEs (within one population) and m to denote migration (between two populations)
*/

__device__ double _Mfunc2D_allo_a(double x, double y, double exy, double g01, double g02, 
                                  double g10, double g11, double g12, double g20, double g21, double g22){
    /*
    * x is x_a, y is x_b
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g10 + (-2.*g10 + g20)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g01 - g02 + g10 -2.*g11 + g12)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g01 + 4.*g10 -4.*g11 -2.*g20 +2.*g21)*xy;
    return exy * (y-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc2D_allo_a(double *x, double *y, double exy, double g01, double g02, double g10, double g11, double g12, 
                               double g20, double g21, double g22, int L, int M, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    if(ii < L){
        output[ii*M + jj] = _Mfunc2D_allo_a(x[ii], y[jj], exy, g01, g02, g10, g11, g12, g20, g21, g22);
    }
}

__device__ double _Mfunc3D_allo_a(double x, double y, double z, double exy, double mxz, double g01, double g02, double g10, double g11, double g12, 
                                  double g20, double g21, double g22){
    /*
    * x is x_a, y is x_b, z is a separate population
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g10 + (-2.*g10 + g20)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g01 - g02 + g10 -2.*g11 + g12)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g01 + 4.*g10 -4.*g11 -2.*g20 +2.*g21)*xy;
    return exy * (y-x) + mxz * (z-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc3D_allo_a(double *x, double *y, double *z, double exy, double mxz, double g01, double g02, double g10, double g11, double g12, 
                               double g20, double g21, double g22, int L, int M, int N, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / N) % M;
    int kk = (blockIdx.x*blockDim.x + threadIdx.x) % N;
    if(ii < L){
        output[ii*(M*N) + jj*N + kk] = _Mfunc3D_allo_a(x[ii], y[jj], z[kk], exy, mxz, g01, g02, g10, g11, g12, g20, g21, g22);
    }
}

/*
* To support demographic models of two allotetraploid populations (e.g. a1, b1, a2, b2), 
* we can simply reuse the subgenome a functions which expect x=a, y=b, z=other pop, a=other pop (apologies for the multi-use of a and b)
* Similarly, for the b subgenomes, our functions will expect x=b, y=other pop, z=other pop, a=a. 

* Then, if we have a1, b1, a2, b2 this looks like
*   a1, b1, a2, b2 (no transpositions; x dim)
*   b1, a2, b2, a1 (one transposition; y dim)
*   a2, b2, a1, b1 (two transpositions; z dim)
*   b2, a1, b1, a2 (three tranpositions; a dim)
* So, our funcs. will work for both allotet. pops. automatically

* Similarly for 5 dimensions (where the first and second or fourth and fifth dimensions are a pair of a and b subgenomes).
*/

__device__ double _Mfunc4D_allo_a(double x, double y, double z, double a, double exy, double mxz, double mxa, 
                                double g01, double g02, double g10, double g11, double g12, 
                                double g20, double g21, double g22){
    /*
    * x is x_a, y is x_b, z and a are separate populations
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g10 + (-2.*g10 + g20)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g01 - g02 + g10 -2.*g11 + g12)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g01 + 4.*g10 -4.*g11 -2.*g20 +2.*g21)*xy;
    return exy * (y-x) + mxz * (z-x) + mxa * (a-x) + + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc4D_allo_a(double *x, double *y, double *z, double *a, double exy, double mxz, double mxa, double g01, double g02, double g10, double g11, double g12, 
                                double g20, double g21, double g22, int L, int M, int N, int O, double *output){   
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / O) % N;
    int ll = (blockIdx.x*blockDim.x + threadIdx.x) % O;
    if(ii < L){
        output[ii*M*N*O + jj*N*O + kk*O + ll] = _Mfunc4D_allo_a(x[ii], y[jj], z[kk], a[ll], exy, mxz, mxa, g01, g02, g10, g11, g12, g20, g21, g22);
    }
}

__device__ double _Mfunc5D_allo_a(double x, double y, double z, double a, double b, double exy, double mxz, double mxa, double mxb,
                                double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22){
    /*
    * x is x_a; y is x_b; z, a, and b are separate populations
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g10 + (-2.*g10 + g20)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g01 - g02 + g10 -2.*g11 + g12)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g01 + 4.*g10 -4.*g11 -2.*g20 +2.*g21)*xy;
    return exy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc5D_allo_a(double *x, double *y, double *z, double *a, double *b, double exy, double mxz, double mxa, double mxb, 
                               double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, int L, int M, int N, int O, int P, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O*P);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O*P)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / (O*P)) % N;
    int ll = ((blockIdx.x*blockDim.x + threadIdx.x) / P) % O;
    int mm = (blockIdx.x*blockDim.x + threadIdx.x) % P;
    if(ii < L){
        output[ii*M*N*O*P + jj*N*O*P + kk*O*P + ll*P + mm] = _Mfunc5D_allo_a(x[ii], y[jj], z[kk], a[ll], b[mm], exy, mxz, mxa, mxb, g01, g02, g10, g11, g12, g20, g21, g22);
    }
}

// This function works for all dimensions, because migration terms
// don't matter at the 0,0,0 and 1,1,1 corners of the regime.
__global__ void include_bc_allo_a(double*dx, double nu1, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, int L, int M, double *b){
    double Mfirst, Mlast;
    // 0,0 entry
    Mfirst = _Mfunc2D_allo_a(0, 0, 0, g01, g02, g10, g11, g12, g20, g21, g22);
    if(Mfirst <= 0){
        b[0] += (0.5/nu1 - Mfirst)*2./dx[0];
    }
    // -1,-1 entry
    Mlast = _Mfunc2D_allo_a(1, 1, 0, g01, g02, g10, g11, g12, g20, g21, g22);
    if(Mlast >= 0){
        b[L*M-1] += -(-0.5/nu1 - Mlast)*2./dx[L-2];
    }
}

/*
* mean functions and BC terms for ALLOTETRAPLOID SUBGENOME B
*/

/*
* Note that for the b subgenomes, our functions will expect x=b and the last population to be the a subgenome.
*/

__device__ double _Mfunc2D_allo_b(double x, double y, double exy, double g01, double g02, 
                                  double g10, double g11, double g12, double g20, double g21, double g22){
    /*
    * x is x_b, y is x_a
    */
    double xy = x*y;
    double yy = y*y;
    double xyy = xy*y;
    double poly = g01 + (-2.*g01 + g02)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*y +
                  (2.*g10 - g20 + g01 -2.*g11 + g21)*yy +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xyy + 
                  (2.*g10 + 4.*g01 -4.*g11 -2.*g02 +2.*g12)*xy;
    return exy * (y-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc2D_allo_b(double *x, double *y, double exy, double g01, double g02, double g10, double g11, double g12, 
                               double g20, double g21, double g22, int L, int M, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    if(ii < L){
        output[ii*M + jj] = _Mfunc2D_allo_b(x[ii], y[jj], exy, g01, g02, g10, g11, g12, g20, g21, g22);
    }
}

__device__ double _Mfunc3D_allo_b(double x, double y, double z, double mxy, double exz, double g01, double g02, 
                                  double g10, double g11, double g12, double g20, double g21, double g22){
    /*
    * x is x_b, y is a separate population, z is x_a
    */
    double xz = x*z;
    double zz = z*z;
    double xzz = xz*z;
    double poly = g01 + (-2.*g01 + g02)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*z +
                  (2.*g10 - g20 + g01 -2.*g11 + g21)*zz +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xzz + 
                  (2.*g10 + 4.*g01 -4.*g11 -2.*g02 +2.*g12)*xz;
    return mxy * (y-x) + exz * (z-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc3D_allo_b(double *x, double *y, double *z, double mxy, double exz, double g01, double g02, double g10, double g11, double g12, 
                               double g20, double g21, double g22, int L, int M, int N, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / N) % M;
    int kk = (blockIdx.x*blockDim.x + threadIdx.x) % N;
    if(ii < L){
        output[ii*(M*N) + jj*N + kk] = _Mfunc3D_allo_b(x[ii], y[jj], z[kk], mxy, exz, g01, g02, g10, g11, g12, g20, g21, g22);
    }
}

__device__ double _Mfunc4D_allo_b(double x, double y, double z, double a, double mxy, double mxz, double exa, 
                                double g01, double g02, double g10, double g11, double g12, 
                                double g20, double g21, double g22){
    /*
    * x is x_b, y and z are separate populations, a is x_a
    */
    double xa = x*a;
    double aa = a*a;
    double xaa = xa*a;
    double poly = g01 + (-2.*g01 + g02)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*a +
                  (2.*g10 - g20 + g01 -2.*g11 + g21)*aa +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xaa + 
                  (2.*g10 + 4.*g01 -4.*g11 -2.*g02 +2.*g12)*xa;
    return mxy * (y-x) + mxz * (z-x) + exa * (a-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc4D_allo_b(double *x, double *y, double *z, double *a, double mxy, double mxz, double exa, double g01, double g02, double g10, double g11, double g12, 
                                double g20, double g21, double g22, int L, int M, int N, int O, double *output){   
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / O) % N;
    int ll = (blockIdx.x*blockDim.x + threadIdx.x) % O;
    if(ii < L){
        output[ii*M*N*O + jj*N*O + kk*O + ll] = _Mfunc4D_allo_b(x[ii], y[jj], z[kk], a[ll], mxy, mxz, exa, g01, g02, g10, g11, g12, g20, g21, g22);
    }
}

__device__ double _Mfunc5D_allo_b(double x, double y, double z, double a, double b, double mxy, double mxz, double mxa, double exb,
                                double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22){
    /*
    * x is x_b, y, z, and a are separate populations, b is x_a
    */
    double xb = x*b;
    double bb = b*b;
    double xbb = xb*b;
    double poly = g01 + (-2.*g01 + g02)*x + 
                  (-2.*g01 - 2.*g10 + 2.*g11)*b +
                  (2.*g10 - g20 + g01 -2.*g11 + g21)*bb +
                  (-2.*g01 + g02 - 2.*g10 + 4.*g11 -2.*g12 + g20 -2.*g21 + g22)*xbb + 
                  (2.*g10 + 4.*g01 -4.*g11 -2.*g02 +2.*g12)*xb;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + exb * (b-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc5D_allo_b(double *x, double *y, double *z, double *a, double *b, double mxy, double mxz, double mxa, double exb, 
                               double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, int L, int M, int N, int O, int P, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O*P);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O*P)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / (O*P)) % N;
    int ll = ((blockIdx.x*blockDim.x + threadIdx.x) / P) % O;
    int mm = (blockIdx.x*blockDim.x + threadIdx.x) % P;
    if(ii < L){
        output[ii*M*N*O*P + jj*N*O*P + kk*O*P + ll*P + mm] = _Mfunc5D_allo_b(x[ii], y[jj], z[kk], a[ll], b[mm], mxy, mxz, mxa, exb, g01, g02, g10, g11, g12, g20, g21, g22);
    }
}

// This function works for all dimensions, because migration terms
// don't matter at the 0,0,0 and 1,1,1 corners of the regime.
__global__ void include_bc_allo_b(double*dx, double nu1, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, int L, int M, double *b){
    double Mfirst, Mlast;
    // 0,0 entry
    Mfirst = _Mfunc2D_allo_b(0, 0, 0, g01, g02, g10, g11, g12, g20, g21, g22);
    if(Mfirst <= 0){
        b[0] += (0.5/nu1 - Mfirst)*2./dx[0];
    }
    // -1,-1 entry
    Mlast = _Mfunc2D_allo_b(1, 1, 0, g01, g02, g10, g11, g12, g20, g21, g22);
    if(Mlast >= 0){
        b[L*M-1] += -(-0.5/nu1 - Mlast)*2./dx[L-2];
    }
}


/*
* mean functions and BC terms for AUTOHEXAPLOIDS
*/

__device__ double _Mfunc2D_autohex(double x, double y, double mxy, double g1, double g2, 
                                  double g3, double g4, double g5, double g6){
    double poly = (((((-6.*g1 + 15.*g2 - 20.*g3 + 15.*g4 - 6.*g5 + g6) * x +
                      (25.*g1 - 50.*g2 + 50.*g3 - 25.*g4 + 5.*g5)) * x +
                      (-40.*g1 + 60.*g2 - 40.*g3 + 10.*g4)) * x +
                      (30.*g1 - 30.*g2 + 10.*g3)) * x +
                      (-10.*g1 + 5.*g2)) * x +
                      g1;
    return mxy * (y-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc2D_autohex(double *x, double *y, double mxy, double g1, double g2, double g3, 
                                double g4, double g5, double g6, int L, int M, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    if(ii < L){
        output[ii*M + jj] = _Mfunc2D_autohex(x[ii], y[jj], mxy, g1, g2, g3, g4, g5, g6);
    }
}

__device__ double _Mfunc3D_autohex(double x, double y, double z, double mxy, double mxz, double g1,  
                                  double g2, double g3, double g4, double g5, double g6){
    double poly = (((((-6.*g1 + 15.*g2 - 20.*g3 + 15.*g4 - 6.*g5 + g6) * x +
                      (25.*g1 - 50.*g2 + 50.*g3 - 25.*g4 + 5.*g5)) * x +
                      (-40.*g1 + 60.*g2 - 40.*g3 + 10.*g4)) * x +
                      (30.*g1 - 30.*g2 + 10.*g3)) * x +
                      (-10.*g1 + 5.*g2)) * x +
                      g1;
    return mxy * (y-x) + mxz * (z-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc3D_autohex(double *x, double *y, double *z, double mxy, double mxz, double g1, double g2, 
                               double g3, double g4, double g5, double g6, int L, int M, int N, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / N) % M;
    int kk = (blockIdx.x*blockDim.x + threadIdx.x) % N;
    if(ii < L){
        output[ii*(M*N) + jj*N + kk] = _Mfunc3D_autohex(x[ii], y[jj], z[kk], mxy, mxz, g1, g2, g3, g4, g5, g6);
    }
}

__device__ double _Mfunc4D_autohex(double x, double y, double z, double a, double mxy, double mxz, double mxa, 
                                   double g1, double g2, double g3, double g4, double g5, double g6){
    double poly = (((((-6.*g1 + 15.*g2 - 20.*g3 + 15.*g4 - 6.*g5 + g6) * x +
                      (25.*g1 - 50.*g2 + 50.*g3 - 25.*g4 + 5.*g5)) * x +
                      (-40.*g1 + 60.*g2 - 40.*g3 + 10.*g4)) * x +
                      (30.*g1 - 30.*g2 + 10.*g3)) * x +
                      (-10.*g1 + 5.*g2)) * x +
                      g1;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc4D_autohex(double *x, double *y, double *z, double *a, double mxy, double mxz, double mxa, double g1, double g2, 
                                double g3, double g4, double g5, double g6, int L, int M, int N, int O, double *output){   
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / O) % N;
    int ll = (blockIdx.x*blockDim.x + threadIdx.x) % O;
    if(ii < L){
        output[ii*M*N*O + jj*N*O + kk*O + ll] = _Mfunc4D_autohex(x[ii], y[jj], z[kk], a[ll], mxy, mxz, mxa, g1, g2, g3, g4, g5, g6);
    }
}

__device__ double _Mfunc5D_autohex(double x, double y, double z, double a, double b, double mxy, double mxz, double mxa, double mxb,
                                   double g1, double g2, double g3, double g4, double g5, double g6){
    double poly = (((((-6.*g1 + 15.*g2 - 20.*g3 + 15.*g4 - 6.*g5 + g6) * x +
                      (25.*g1 - 50.*g2 + 50.*g3 - 25.*g4 + 5.*g5)) * x +
                      (-40.*g1 + 60.*g2 - 40.*g3 + 10.*g4)) * x +
                      (30.*g1 - 30.*g2 + 10.*g3)) * x +
                      (-10.*g1 + 5.*g2)) * x +
                      g1;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + 2. * x * (1. - x) * poly;
}


__global__ void Mfunc5D_autohex(double *x, double *y, double *z, double *a, double *b, double mxy, double mxz, double mxa, double mxb, 
                               double g1, double g2, double g3, double g4, double g5, double g6, int L, int M, int N, int O, int P, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O*P);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O*P)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / (O*P)) % N;
    int ll = ((blockIdx.x*blockDim.x + threadIdx.x) / P) % O;
    int mm = (blockIdx.x*blockDim.x + threadIdx.x) % P;
    if(ii < L){
        output[ii*M*N*O*P + jj*N*O*P + kk*O*P + ll*P + mm] = _Mfunc5D_autohex(x[ii], y[jj], z[kk], a[ll], b[mm], mxy, mxz, mxa, mxb, g1, g2, g3, g4, g5, g6);
    }
}

// This function works for all dimensions, because migration terms
// don't matter at the 0,0,0 and 1,1,1 corners of the regime.
__global__ void include_bc_autohex(double*dx, double nu1, double g1, double g2, double g3, double g4, double g5, double g6, int L, int M, double *b){
    double Mfirst, Mlast;
    // 0,0 entry
    Mfirst = _Mfunc2D_autohex(0, 0, 0, g1, g2, g3, g4, g5, g6);
    if(Mfirst <= 0){
        b[0] += (0.5/(3.*nu1) - Mfirst)*2./dx[0];
    }
    // -1,-1 entry
    Mlast = _Mfunc2D_autohex(1, 1, 0, g1, g2, g3, g4, g5, g6);
    if(Mlast >= 0){
        b[L*M-1] += -(-0.5/(3.*nu1) - Mlast)*2./dx[L-2];
    }
}

/*
* mean functions and BC terms for ALLOAUTOHEXAPLOIDS (4+2 hexaploids) - *Tetraploid subgenome*
*
* These will be structured nearly identical to the allotetraploid a and b functions above.
* Here, we treat the a subgenome as the tetraploid subgenome, and the b subgenome as the diploid subgenome.
* So, for this first set of functions, x is always the tetraploid subgenome frequency, and y is the diploid subgenome frequency.
* 
* For the later functions for the diploid subgenome, x is always the diploid subgenome frequency, 
* and the last dimension of the phi array is the tetraploid subgenome frequency.
*/

__device__ double _Mfunc2D_hex_tetra(double x, double y, double exy, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_4, y is x_2 where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    */
    double xy = x*y; // qa*qb
    double xx = x*x; // qa^2
    double xxx = xx*x; // qa^3
    double yy = y*y; // qb^2
    double xyy = xy*y; // qa*qb^2
    double xxy = xx*y; // qa^2*qb
    double xxxy = xxx*y; // qa^3*qb
    double xxyy = xxy*y; // qa^2*qb^2
    double xxxyy = xxxy*y; // qa^3*qb^2
    double poly = g10 + (-6.*g10 + 3.*g20) * x +
                  (-2.*g01 - 2.*g10 + 2.*g11) * y +
                  (9.*g10 -9.*g20 + 3.*g30) * xx + 
                  (-4.*g10 + 6.*g20 - 4.*g30 + g40) * xxx + 
                  (2.*g01 - g02 + g10 - 2.*g11 + g12) * yy + 
                  (-6.*g01 + 3.*g02 - 6.*g10 + 12.*g11 - 6.*g12 + 3.*g20 - 6.*g21 + 3.*g22) * xyy +
                  (-6.*g01 - 18.*g10 + 18.*g11 + 18.*g20 -18.*g21 -6.*g30 +6.*g31) * xxy +
                  (2.*g01 + 8.*g10 - 8.*g11 - 12.*g20 + 12.*g21 + 8.*g30 - 8.*g31 - 2.*g40 + 2.*g41) * xxxy +
                  (6.*g01 - 3.*g02 + 9.*g10 - 18.*g11 + 9.*g12 - 9.*g20 + 18.*g21 - 9.*g22 + 3.*g30 - 6.*g31 + 3.*g32) * xxyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xxxyy +
                  (6.*g01 + 12.*g10 - 12.*g11 - 6.*g20 + 6.*g21) * xy;
    // note the 1/2 term in the exchange term here to correct for differences in ploidy between subgenomes
    return exy * (y-x) / 2. + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc2D_hex_tetra(double *x, double *y, double exy, double g01, double g02, double g10, double g11, double g12, 
                                double g20, double g21, double g22, double g30, double g31, double g32, 
                                double g40, double g41, double g42, int L, int M, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    if(ii < L){
        output[ii*M + jj] = _Mfunc2D_hex_tetra(x[ii], y[jj], exy, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    }
}

__device__ double _Mfunc3D_hex_tetra(double x, double y, double z, double exy, double mxz, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_4, y is x_2 where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    * z is a separate population
    */
    double xy = x*y; // qa*qb
    double xx = x*x; // qa^2
    double xxx = xx*x; // qa^3
    double yy = y*y; // qb^2
    double xyy = xy*y; // qa*qb^2
    double xxy = xx*y; // qa^2*qb
    double xxxy = xxx*y; // qa^3*qb
    double xxyy = xxy*y; // qa^2*qb^2
    double xxxyy = xxxy*y; // qa^3*qb^2
    double poly = g10 + (-6.*g10 + 3.*g20) * x +
                  (-2.*g01 - 2.*g10 + 2.*g11) * y +
                  (9.*g10 -9.*g20 + 3.*g30) * xx + 
                  (-4.*g10 + 6.*g20 - 4.*g30 + g40) * xxx + 
                  (2.*g01 - g02 + g10 - 2.*g11 + g12) * yy + 
                  (-6.*g01 + 3.*g02 - 6.*g10 + 12.*g11 - 6.*g12 + 3.*g20 - 6.*g21 + 3.*g22) * xyy +
                  (-6.*g01 - 18.*g10 + 18.*g11 + 18.*g20 -18.*g21 -6.*g30 +6.*g31) * xxy +
                  (2.*g01 + 8.*g10 - 8.*g11 - 12.*g20 + 12.*g21 + 8.*g30 - 8.*g31 - 2.*g40 + 2.*g41) * xxxy +
                  (6.*g01 - 3.*g02 + 9.*g10 - 18.*g11 + 9.*g12 - 9.*g20 + 18.*g21 - 9.*g22 + 3.*g30 - 6.*g31 + 3.*g32) * xxyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xxxyy +
                  (6.*g01 + 12.*g10 - 12.*g11 - 6.*g20 + 6.*g21) * xy;
    // note the 1/2 term in the exchange term here to correct for differences in ploidy between subgenomes
    return exy * (y-x) / 2. + mxz * (z-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc3D_hex_tetra(double *x, double *y, double *z, double exy, double mxz, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, double g30, double g31, double g32, 
                         double g40, double g41, double g42, int L, int M, int N, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / N) % M;
    int kk = (blockIdx.x*blockDim.x + threadIdx.x) % N;
    if(ii < L){
        output[ii*(M*N) + jj*N + kk] = _Mfunc3D_hex_tetra(x[ii], y[jj], z[kk], exy, mxz, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    }
}

__device__ double _Mfunc4D_hex_tetra(double x, double y, double z, double a, double exy, double mxz, double mxa, 
                                     double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                                     double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_4, y is x_2 where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    * z and a are separate populations
    */
    double xy = x*y; // qa*qb
    double xx = x*x; // qa^2
    double xxx = xx*x; // qa^3
    double yy = y*y; // qb^2
    double xyy = xy*y; // qa*qb^2
    double xxy = xx*y; // qa^2*qb
    double xxxy = xxx*y; // qa^3*qb
    double xxyy = xxy*y; // qa^2*qb^2
    double xxxyy = xxxy*y; // qa^3*qb^2
    double poly = g10 + (-6.*g10 + 3.*g20) * x +
                  (-2.*g01 - 2.*g10 + 2.*g11) * y +
                  (9.*g10 -9.*g20 + 3.*g30) * xx + 
                  (-4.*g10 + 6.*g20 - 4.*g30 + g40) * xxx + 
                  (2.*g01 - g02 + g10 - 2.*g11 + g12) * yy + 
                  (-6.*g01 + 3.*g02 - 6.*g10 + 12.*g11 - 6.*g12 + 3.*g20 - 6.*g21 + 3.*g22) * xyy +
                  (-6.*g01 - 18.*g10 + 18.*g11 + 18.*g20 -18.*g21 -6.*g30 +6.*g31) * xxy +
                  (2.*g01 + 8.*g10 - 8.*g11 - 12.*g20 + 12.*g21 + 8.*g30 - 8.*g31 - 2.*g40 + 2.*g41) * xxxy +
                  (6.*g01 - 3.*g02 + 9.*g10 - 18.*g11 + 9.*g12 - 9.*g20 + 18.*g21 - 9.*g22 + 3.*g30 - 6.*g31 + 3.*g32) * xxyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xxxyy +
                  (6.*g01 + 12.*g10 - 12.*g11 - 6.*g20 + 6.*g21) * xy;
    // note the 1/2 term in the exchange term here to correct for differences in ploidy between subgenomes
    return exy * (y-x) / 2. + mxz * (z-x) + mxa * (a-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc4D_hex_tetra(double *x, double *y, double *z, double *a, double exy, double mxz, double mxa, 
                                  double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                                  double g30, double g31, double g32, double g40, double g41, double g42, int L, int M, int N, int O, double *output){   
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / O) % N;
    int ll = (blockIdx.x*blockDim.x + threadIdx.x) % O;
    if(ii < L){
        output[ii*M*N*O + jj*N*O + kk*O + ll] = _Mfunc4D_hex_tetra(x[ii], y[jj], z[kk], a[ll], exy, mxz, mxa, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    }
}

__device__ double _Mfunc5D_hex_tetra(double x, double y, double z, double a, double b, double exy, double mxz, double mxa, double mxb,
                                     double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                                     double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_4, y is x_2 where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    * z, a, and b are separate populations
    */
    double xy = x*y; // qa*qb
    double xx = x*x; // qa^2
    double xxx = xx*x; // qa^3
    double yy = y*y; // qb^2
    double xyy = xy*y; // qa*qb^2
    double xxy = xx*y; // qa^2*qb
    double xxxy = xxx*y; // qa^3*qb
    double xxyy = xxy*y; // qa^2*qb^2
    double xxxyy = xxxy*y; // qa^3*qb^2
    double poly = g10 + (-6.*g10 + 3.*g20) * x +
                  (-2.*g01 - 2.*g10 + 2.*g11) * y +
                  (9.*g10 -9.*g20 + 3.*g30) * xx + 
                  (-4.*g10 + 6.*g20 - 4.*g30 + g40) * xxx + 
                  (2.*g01 - g02 + g10 - 2.*g11 + g12) * yy + 
                  (-6.*g01 + 3.*g02 - 6.*g10 + 12.*g11 - 6.*g12 + 3.*g20 - 6.*g21 + 3.*g22) * xyy +
                  (-6.*g01 - 18.*g10 + 18.*g11 + 18.*g20 -18.*g21 -6.*g30 +6.*g31) * xxy +
                  (2.*g01 + 8.*g10 - 8.*g11 - 12.*g20 + 12.*g21 + 8.*g30 - 8.*g31 - 2.*g40 + 2.*g41) * xxxy +
                  (6.*g01 - 3.*g02 + 9.*g10 - 18.*g11 + 9.*g12 - 9.*g20 + 18.*g21 - 9.*g22 + 3.*g30 - 6.*g31 + 3.*g32) * xxyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xxxyy +
                  (6.*g01 + 12.*g10 - 12.*g11 - 6.*g20 + 6.*g21) * xy;
    // note the 1/2 term in the exchange term here to correct for differences in ploidy between subgenomes
    return exy * (y-x) / 2. + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc5D_hex_tetra(double *x, double *y, double *z, double *a, double *b, double exy, double mxz, double mxa, double mxb, 
                                  double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, double g30, double g31, 
                                  double g32, double g40, double g41, double g42, int L, int M, int N, int O, int P, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O*P);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O*P)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / (O*P)) % N;
    int ll = ((blockIdx.x*blockDim.x + threadIdx.x) / P) % O;
    int mm = (blockIdx.x*blockDim.x + threadIdx.x) % P;
    if(ii < L){
        output[ii*M*N*O*P + jj*N*O*P + kk*O*P + ll*P + mm] = _Mfunc5D_hex_tetra(x[ii], y[jj], z[kk], a[ll], b[mm], exy, mxz, mxa, mxb, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    }
}

// This function works for all dimensions, because migration terms
// don't matter at the 0,0,0 and 1,1,1 corners of the regime.
__global__ void include_bc_hex_tetra(double*dx, double nu1, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                                   double g30, double g31, double g32, double g40, double g41, double g42, int L, int M, double *b){
    double Mfirst, Mlast;
    // 0,0 entry
    Mfirst = _Mfunc2D_hex_tetra(0, 0, 0, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    if(Mfirst <= 0){
        b[0] += (0.25/nu1 - Mfirst)*2./dx[0];
    }
    // -1,-1 entry
    Mlast = _Mfunc2D_hex_tetra(1, 1, 0, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    if(Mlast >= 0){
        b[L*M-1] += -(-0.25/nu1 - Mlast)*2./dx[L-2];
    }
}

/*
* mean functions and BC terms for ALLOAUTOHEXAPLOIDS (4+2 hexaploids) - *diploid subgenome*
*
* These will be structured nearly identical to the allotetraploid a and b functions above.
* Here, we treat the a subgenome as the tetraploid subgenome, and the b subgenome as the diploid subgenome.
* For this second set of functions, x is always the diploid subgenome frequency
* and the last dimension is the tetraploid subgenome frequency.
*/

__device__ double _Mfunc2D_hex_dip(double x, double y, double exy, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_2, y is x_4 
    * where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    */
    double xy = x*y; // qa*qb
    double yy = y*y; // qa^2
    double yyy = yy*y; // qa^3
    double yyyy = yyy*y; // qa^4
    double xyy = xy*y; // qa^2*qb
    double xyyy = xyy*y; // qa^3*qb
    double xyyyy = xyyy*y; // qa^4*qb
    double poly = g01 + (-4.*g01 - 4.*g10 + 4.*g11) * y + 
                  (-2.*g01 + g02) * x + 
                  (6.*g01 + 12.*g10 - 12.*g11 -6.*g20 + 6.*g21) * yy +
                  (-4.*g01 -12.*g10 + 12.*g11 + 12.*g20 - 12.*g21 - 4.*g30 + 4.*g31) * yyy + 
                  (g01 + 4.*g10 - 4.*g11 - 6.*g20 + 6.*g21 + 4.*g30 - 4.*g31 - g40 + g41) * yyyy +
                  (-12.*g01 + 6.*g02 - 12.*g10 + 24.*g11 - 12.*g12 + 6.*g20 - 12.*g21 + 6.*g22) * xyy + 
                  (8.*g01 - 4.*g02 + 12.*g10 - 24.*g11 + 12.*g12 - 12.*g20 + 24.*g21 - 12.*g22 + 4.*g30 - 8.*g31 + 4.*g32) * xyyy + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xyyyy + 
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12) * xy;
    return exy * (y-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc2D_hex_dip(double *x, double *y, double exy, double g01, double g02, double g10, double g11, double g12, 
                                double g20, double g21, double g22, double g30, double g31, double g32, 
                                double g40, double g41, double g42, int L, int M, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    if(ii < L){
        output[ii*M + jj] = _Mfunc2D_hex_dip(x[ii], y[jj], exy, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    }
}

__device__ double _Mfunc3D_hex_dip(double x, double y, double z, double mxy, double exz, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, 
                         double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_2, z is x_4 
    * where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    * y is a separate population
    */
    double xz = x*z; // qa*qb
    double zz = z*z; // qa^2
    double zzz = zz*z; // qa^3
    double zzzz = zzz*z; // qa^4
    double xzz = xz*z; // qa^2*qb
    double xzzz = xzz*z; // qa^3*qb
    double xzzzz = xzzz*z; // qa^4*qb
    double poly = g01 + (-4.*g01 - 4.*g10 + 4.*g11) * z + 
                  (-2.*g01 + g02) * x + 
                  (6.*g01 + 12.*g10 - 12.*g11 -6.*g20 + 6.*g21) * zz +
                  (-4.*g01 -12.*g10 + 12.*g11 + 12.*g20 - 12.*g21 - 4.*g30 + 4.*g31) * zzz + 
                  (g01 + 4.*g10 - 4.*g11 - 6.*g20 + 6.*g21 + 4.*g30 - 4.*g31 - g40 + g41) * zzzz +
                  (-12.*g01 + 6.*g02 - 12.*g10 + 24.*g11 - 12.*g12 + 6.*g20 - 12.*g21 + 6.*g22) * xzz + 
                  (8.*g01 - 4.*g02 + 12.*g10 - 24.*g11 + 12.*g12 - 12.*g20 + 24.*g21 - 12.*g22 + 4.*g30 - 8.*g31 + 4.*g32) * xzzz + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xzzzz + 
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12) * xz;
    return mxy * (y-x) + exz * (z-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc3D_hex_dip(double *x, double *y, double *z, double mxy, double exz, double g01, double g02, 
                         double g10, double g11, double g12, double g20, double g21, double g22, double g30, double g31, double g32, 
                         double g40, double g41, double g42, int L, int M, int N, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / N) % M;
    int kk = (blockIdx.x*blockDim.x + threadIdx.x) % N;
    if(ii < L){
        output[ii*(M*N) + jj*N + kk] = _Mfunc3D_hex_dip(x[ii], y[jj], z[kk], mxy, exz, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    }
}

__device__ double _Mfunc4D_hex_dip(double x, double y, double z, double a, double mxy, double mxz, double exa, 
                                     double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                                     double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_2, a is x_4 
    * where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    * y and z are separate populations
    */
    double xa = x*a; // qa*qb
    double aa = a*a; // qa^2
    double aaa = aa*a; // qa^3
    double aaaa = aaa*a; // qa^4
    double xaa = xa*a; // qa^2*qb
    double xaaa = xaa*a; // qa^3*qb
    double xaaaa = xaaa*a; // qa^4*qb
    double poly = g01 + (-4.*g01 - 4.*g10 + 4.*g11) * a + 
                  (-2.*g01 + g02) * x + 
                  (6.*g01 + 12.*g10 - 12.*g11 -6.*g20 + 6.*g21) * aa +
                  (-4.*g01 -12.*g10 + 12.*g11 + 12.*g20 - 12.*g21 - 4.*g30 + 4.*g31) * aaa + 
                  (g01 + 4.*g10 - 4.*g11 - 6.*g20 + 6.*g21 + 4.*g30 - 4.*g31 - g40 + g41) * aaaa +
                  (-12.*g01 + 6.*g02 - 12.*g10 + 24.*g11 - 12.*g12 + 6.*g20 - 12.*g21 + 6.*g22) * xaa + 
                  (8.*g01 - 4.*g02 + 12.*g10 - 24.*g11 + 12.*g12 - 12.*g20 + 24.*g21 - 12.*g22 + 4.*g30 - 8.*g31 + 4.*g32) * xaaa + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xaaaa + 
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12) * xa;
    return mxy * (y-x) + mxz * (z-x) + exa * (a-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc4D_hex_dip(double *x, double *y, double *z, double *a, double mxy, double mxz, double exa, 
                                  double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                                  double g30, double g31, double g32, double g40, double g41, double g42, int L, int M, int N, int O, double *output){   
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / O) % N;
    int ll = (blockIdx.x*blockDim.x + threadIdx.x) % O;
    if(ii < L){
        output[ii*M*N*O + jj*N*O + kk*O + ll] = _Mfunc4D_hex_dip(x[ii], y[jj], z[kk], a[ll], mxy, mxz, exa, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    }
}

__device__ double _Mfunc5D_hex_dip(double x, double y, double z, double a, double b, double mxy, double mxz, double mxa, double exb,
                                     double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                                     double g30, double g31, double g32, double g40, double g41, double g42){
    /* 
    * x is x_2, b is x_4 
    * where x_4 is the allele frequency in the tetraploid subgenome
    * and x_2 is the allele frequency in the diploid subgenome
    * in the matlab code, we denote x_4 as qa and x_2 as qb
    * y, z, and a are separate populations
    */
    double xb = x*b; // qa*qb
    double bb = b*b; // qa^2
    double bbb = bb*b; // qa^3
    double bbbb = bbb*b; // qa^4
    double xbb = xb*b; // qa^2*qb
    double xbbb = xbb*b; // qa^3*qb
    double xbbbb = xbbb*b; // qa^4*qb
    double poly = g01 + (-4.*g01 - 4.*g10 + 4.*g11) * b + 
                  (-2.*g01 + g02) * x + 
                  (6.*g01 + 12.*g10 - 12.*g11 -6.*g20 + 6.*g21) * bb +
                  (-4.*g01 -12.*g10 + 12.*g11 + 12.*g20 - 12.*g21 - 4.*g30 + 4.*g31) * bbb + 
                  (g01 + 4.*g10 - 4.*g11 - 6.*g20 + 6.*g21 + 4.*g30 - 4.*g31 - g40 + g41) * bbbb +
                  (-12.*g01 + 6.*g02 - 12.*g10 + 24.*g11 - 12.*g12 + 6.*g20 - 12.*g21 + 6.*g22) * xbb + 
                  (8.*g01 - 4.*g02 + 12.*g10 - 24.*g11 + 12.*g12 - 12.*g20 + 24.*g21 - 12.*g22 + 4.*g30 - 8.*g31 + 4.*g32) * xbbb + 
                  (-2.*g01 + g02 - 4.*g10 + 8.*g11 - 4.*g12 + 6.*g20 - 12.*g21 + 6.*g22 - 4.*g30 + 8.*g31 - 4.*g32 + g40 - 2.*g41 + g42) * xbbbb + 
                  (8.*g01 - 4.*g02 + 4.*g10 - 8.*g11 + 4.*g12) * xb;
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + exb * (b-x) + 2. * x * (1. - x) * poly;
}

__global__ void Mfunc5D_hex_dip(double *x, double *y, double *z, double *a, double *b, double mxy, double mxz, double mxa, double exb, 
                                  double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, double g30, double g31, 
                                  double g32, double g40, double g41, double g42, int L, int M, int N, int O, int P, double *output){
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / (M*N*O*P);
    int jj = ((blockIdx.x*blockDim.x + threadIdx.x) / (N*O*P)) % M;
    int kk = ((blockIdx.x*blockDim.x + threadIdx.x) / (O*P)) % N;
    int ll = ((blockIdx.x*blockDim.x + threadIdx.x) / P) % O;
    int mm = (blockIdx.x*blockDim.x + threadIdx.x) % P;
    if(ii < L){
        output[ii*M*N*O*P + jj*N*O*P + kk*O*P + ll*P + mm] = _Mfunc5D_hex_tetra(x[ii], y[jj], z[kk], a[ll], b[mm], mxy, mxz, mxa, exb, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    }
}

// This function works for all dimensions, because migration terms
// don't matter at the 0,0,0 and 1,1,1 corners of the regime.
__global__ void include_bc_hex_dip(double*dx, double nu1, double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                                   double g30, double g31, double g32, double g40, double g41, double g42, int L, int M, double *b){
    double Mfirst, Mlast;
    // 0,0 entry
    Mfirst = _Mfunc2D_hex_dip(0, 0, 0, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    if(Mfirst <= 0){
        b[0] += (0.5/nu1 - Mfirst)*2./dx[0];
    }
    // -1,-1 entry
    Mlast = _Mfunc2D_hex_dip(1, 1, 0, g01, g02, g10, g11, g12, g20, g21, g22, g30, g31, g32, g40, g41, g42);
    if(Mlast >= 0){
        b[L*M-1] += -(-0.5/nu1 - Mlast)*2./dx[L-2];
    }
}