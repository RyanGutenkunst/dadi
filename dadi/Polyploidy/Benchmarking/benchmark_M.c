#include <stdio.h>
#include <time.h>

double Mfunc1D_auto(double x, double gam1, double gam2, double gam3, double gam4) {
    double poly = ((-4.*gam1 + 6.*gam2 - 4.*gam3 + gam4) * x +
                   (9.*gam1 - 9.*gam2 + 3.*gam3)) * x +
                   (-6.*gam1 + 3.*gam2) * x +
                   gam1;
    return x * (1. - x) * 2. * poly;
}

double Mfunc1D_auto_naive(double x, double gam1, double gam2, double gam3, double gam4){
    return x * (1.-x) * 2. * (
        gam1 + (-6.*gam1 + 3.*gam2)*x  +
        (9.*gam1 -9.*gam2 + 3.*gam3)*x*x + 
        (-4.*gam1 +6.*gam2 - 4.*gam3 + gam4)*x*x*x
    );
}

double Mfunc1D(double x, double gamma, double h){
    return gamma * 2*(h + (1.-2*h)*x) * x*(1.-x);
}


int main() {
    const int N = 1000000;
    double gamma = 4.0, h = 0.5, gam1 = 1.0, gam2 = 2.0, gam3 = 3.0, gam4 = 4.0;
    double result = 0.0;
    double x;


    clock_t start = clock();
    for (int i = 0; i < N; ++i) {
        x = (i + 0.5) / N;
        result += Mfunc1D_auto(x, gam1, gam2, gam3, gam4);
    }
    clock_t end = clock();
    printf("Auto version (Horner): %.6f sec, result = %f\n", (double)(end - start) / CLOCKS_PER_SEC, result / N);

    result = 0.0;
    start = clock();
    for (int i = 0; i < N; ++i) {
        x = (i + 0.5) / N;
        result += Mfunc1D_auto_naive(x, gam1, gam2, gam3, gam4);
    }
    end = clock();
    printf("Auto version (naive): %.6f sec, result = %f\n", (double)(end - start) / CLOCKS_PER_SEC, result / N);


    result = 0.0;
    start = clock();
    for (int i = 0; i < N; ++i) {
        x = (i + 0.5) / N;
        result += Mfunc1D(x, gamma, h);
    }
    end = clock();
    printf("Diploid version: %.6f sec, result = %f\n", (double)(end - start) / CLOCKS_PER_SEC, result / N);

    return 0;
}
