__global__ void inject_mutations_2D(double *phi, int L, double val01, double val10){
    phi[1] += val01;
    phi[L] += val10;
}

__global__ void Vfunc(double *x, double nu, int L, double *output){
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if(ii < L){
        output[ii] =  x[ii] * (1.-x[ii])/nu;
    }
}

__device__ double _Mfunc2D(double x, double y, double m, double gamma, double h){
    return m * (y-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x);
}

__global__ void Mfunc2D(double *x, double *y, double m, double gamma, double h, int L, int M, double *output){
    int ii = blockIdx.x*blockDim.x + threadIdx.x / M;
    int jj = blockIdx.x*blockDim.x + threadIdx.x % M;
    if(ii < L){
        output[ii*M + jj] = _Mfunc2D(x[ii], y[jj], m, gamma, h);
    }
}

__global__ void cx0(double *cx, int L, int M){
    int jj = blockIdx.x*blockDim.x + threadIdx.x;
    if(jj < M){
        cx[(L-1)*M + jj] = 0;
    }
}

__global__ void include_bc(double*dx, double nu1, double m, double gamma, double h, int L, int M, double *b){
    double Mfirst, Mlast;
    // 0,0 entry
    Mfirst = _Mfunc2D(0, 0, m, gamma, h);
    if(Mfirst <= 0){
        b[0] += (0.5/nu1 - Mfirst)*2./dx[0];
    }
    // -1,-1 entry
    Mlast = _Mfunc2D(1, 1, m, gamma, h);
    if(Mlast >= 0){
        b[L*M-1] += -(-0.5/nu1 - Mlast)*2./dx[L-2];
    }
}

__global__ void compute_abc_nobc(double *dx, double *dfactor, 
        double *MInt, double *V, double dt, int L, int M,
        double *a, double *b, double *c){
    int ii = blockIdx.x*blockDim.x + threadIdx.x / M;
    int jj = blockIdx.x*blockDim.x + threadIdx.x % M;
    double atemp, ctemp;

    if(ii < L-1){
        atemp = MInt[ii*M + jj] * 0.5 + V[ii]/(2*dx[ii]);
        a[(ii+1)*M + jj] = -dfactor[ii+1]*atemp;
        b[ii*M + jj] += dfactor[ii]*atemp;

        ctemp = -MInt[ii*M + jj] * 0.5 + V[ii+1]/(2*dx[ii]);
        b[(ii+1)*M + jj] += dfactor[ii+1]*ctemp;
        c[ii*M + jj] = -dfactor[ii]*ctemp;
    }
}