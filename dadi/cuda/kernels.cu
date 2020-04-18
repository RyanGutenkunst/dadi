__global__ void inject_mutations_2D(double *phi, int L, double val01, double val10){
    phi[1] += val01;
    phi[L] += val10;
}
