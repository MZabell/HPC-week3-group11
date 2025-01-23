#include <stdlib.h>
#include "alloc3d.h"

double ***
malloc_3d(int m, int n, int k) {

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    double ***p = (double***) malloc(m * sizeof(double **) +
                                     m * n * sizeof(double *));
    if (p == NULL) {
        return NULL;
    }

    for(int i = 0; i < m; i++) {
        p[i] = (double **) p + m + i * n ;
    }

    double *a = (double*) malloc(m * n * k * sizeof(double));
    if (a == NULL) {
	free(p);
	return NULL;
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }

    return p;
}

void init_u(double ***u, int N, double start_T) {
    #pragma omp for schedule(static) collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                u[i][j][k] = start_T;
            }
        }
    }
    #pragma omp for schedule(static)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            u[i][0][j] = 0.0;
            u[i][N-1][j] = 20.0;
            u[0][i][j] = 20.0;
            u[N-1][i][j] = 20.0;
            u[i][j][0] = 20.0;
            u[i][j][N-1] = 20.0;
        }
    }
}

void init_f(double ***f, int N) {
    #pragma omp for schedule(static) collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                f[i][j][k] = 0.0;
            }
        }
    }
    #pragma omp for schedule(static) collapse(2)
    for(int i = 0; i < 5*N/16; i++) {
        for(int j = 0; j < N/4; j++) {
            for(int k = N/6; k < N/2; k++) {
                f[i][j][k] = 200.0;
            }
        }
    }
}

void
free_3d(double ***p) {
    free(p[0][0]);
    free(p);
}
