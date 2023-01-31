#ifndef TRIDIAG_H
#define TRIDIAG_H
void tridiag(double a[], double b[], double c[], double r[], double u[], int n);

void tridiag_malloc(int n);
void tridiag_free(void);
/* This version of tridiag uses previously allocated memory, for slighly *
 * improved performance with repeated solution of problems of the same size.
 */
void tridiag_premalloc(double a[], double b[], double c[], double r[], double u[], int n);
#endif
