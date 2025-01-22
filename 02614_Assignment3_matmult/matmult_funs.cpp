//
// Created by Zahed on 22/01/2025.

/**
 * TODO
 * The mkn version is not the optimal permutation when offloading to the GPU. Why?
 * What team and thread sizes should experiment with? And is there a way to automatize?
 *
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
// Minimum number of threads for full throughput (this is not optimal!): N SMs x 4 x 32


// Minimum number of threads for full occupancy (this is often optimal): N SMs x 4 x 12 x 32 =
// If there is an imbalanced amount of work, use one thread per iteration.
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