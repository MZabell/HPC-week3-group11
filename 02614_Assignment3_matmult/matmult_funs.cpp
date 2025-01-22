//
// Created by Zahed on 22/01/2025.

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

#pragma omp target teams distribute parallel for num_teams(114) thread_limit(64) \
    map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n])
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }

    //printf("%f\n", C[3][3]);
}


/* ---------------- MNK -----------------------*/

void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double **C) {
#pragma omp target teams distribute parallel for num_teams(114) thread_limit(64) \
        map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n])
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int l = 0; l < k; l++) {
                sum += A[i][l] * B[l][j];
            }
            C[i][j] = sum;
        }
    }
}


}