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
    int ii = (blockIdx.x*blockDim.x + threadIdx.x) / M;
    int jj = (blockIdx.x*blockDim.x + threadIdx.x) % M;
    if(ii < L){
        output[ii*M + jj] = _Mfunc2D(x[ii], y[jj], m, gamma, h);
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

// We need an additional simple kernel to zero out the necessary
// values of the c array, because the Interleaved tridiagonal
// solver alters the c array.
__global__ void cx0(double *cx, int L, int M){
    int jj = blockIdx.x*blockDim.x + threadIdx.x;
    if(jj < M){
        cx[(L-1)*M + jj] = 0;
    }
}

// This function works for 2D and 3D cases, because migration terms
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
