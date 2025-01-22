//
// Created by Zahed on 22/01/2025.
//
//#include <omp.h>
/*

void mkn_offload(int m, int n, int k, double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
        }
    }
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}
*/


extern "C" {
#include <cblas.h>
#include <omp.h>
#include <stdio.h>


void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
        }
    }

//#pragma omp target teams distribute parallel for num_teams(114) thread_limit(64) \
    map(to: A[0:m][0:k], B[0:k][0:n]) map(from: C[0:m][0:n])
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }

    printf("%f\n", C[3][3]);
}





/*

void mkn_offload2(int m,int n,int k,double **A,double **B,double **C) {
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            C[i][j] = 0;
        }
    }

#pragma omp target teams distribute parallel for num_teams(114) thread_limit(64) map(to: m, n, k, A, B, C) map(from: C)
    for (int i=0;i<m;i++) {
        for (int l=0;l<k;l++) {
            double sum = 0;
            for (int j=0;j<n;j++) {
                C[i][j] += A[i][l]*B[l][j];
            }
        }
    }
}


*/



/*

void matmult_mkn_offload(int m, int n, int k, double *A, double *B, double *C) {
    int val = 0;
*/
/*#pragma omp target teams distribute parallel for map(from: val)
    for (int i = 0; i < m; i++) {
        val = val + i;
    }*//*


    printf("Returned Value: %d\n", val);

}

*/







}