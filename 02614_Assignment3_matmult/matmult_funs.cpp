//
// Created by Zahed on 22/01/2025.

/**
 * TODO
 * The mkn version is not the optimal permutation when offloading to the GPU. Why?
 * What team and thread sizes should experiment with? And is there a way to automatize?
 *
 */


// Minimum number of threads for full throughput (this is not optimal!): N SMs x 4 x 32
// Minimum number of threads for full occupancy (this is often optimal): N SMs x 4 x 12 x 32 =
// If there is an imbalanced amount of work, use one thread per iteration.


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
        // You want adjacent threads to access adjacent memory. E.g. thread 1,2,3,... access outermost index 1,2,3...
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}



void matmult_mkn_offload_ignore(int m, int n, int k, double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
        }
    }
// Minimum number of threads for full throughput (this is not optimal!): N SMs x 4 x 32


// Minimum number of threads for full occupancy (this is often optimal): N SMs x 4 x 12 x 32 =
// If there is an imbalanced amount of work, use one thread per iteration.
#pragma omp target teams distribute parallel for collapse(2) num_teams(64)  \
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

void matmult_mnk_offload_main(int m, int n, int k, double **A, double **B, double **C) {
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


void matmult_mnk_offload2(int m, int n, int k, double **A, double **B, double **C) {
#pragma omp target teams distribute parallel for collapse(2) \
    num_teams(114) thread_limit(64) \
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


void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double **C) {

#pragma omp target teams distribute parallel for collapse(2) \
    map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n])
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            if (i == 0 && j == 0) {
                printf("Number of teams: %d\n", omp_get_num_teams());
                printf("Number of thrads: %d\n", omp_get_num_threads());
            }

            double sum = 0;
            for (int l = 0; l < k; l++) {
                sum += A[i][l] * B[l][j];
            }
            C[i][j] = sum;
        }
    }
}


/* ---------------- BLK -----------------------*/


void matmult_blk_offload(int m, int n, int k, double **A, double **B, double **C) {
#define BLK 8

#pragma omp target teams distribute parallel for num_teams(114) thread_limit(64) \
        map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n])
    for (int i = 0; i < m; i += BLK) {
        for (int j = 0; j < n; ++j) {
            if (i + BLK - 1 < m) { // If the full block fits within the range
                // Do BLK elements of C here
                double sum[BLK] = {0};
                for (int l = 0; l < k; ++l) {
                    for (int ii = 0; ii < BLK; ++ii) {
                        sum[ii] += A[i + ii][l] * B[l][j];
                    }
                }
                for (int ii = 0; ii < BLK; ++ii) {
                    C[i + ii][j] = sum[ii];
                }
            } else {
                // Do the remainder part here
                for (int ii = 0; ii < m - i; ++ii) {
                    double sum = 0;
                    for (int l = 0; l < k; ++l) {
                        sum += A[i + ii][l] * B[l][j];
                    }
                    C[i + ii][j] = sum;
                }
            }
        }
    }
}





void matmult_blk_offload2(int m, int n, int k, double **A, double **B, double **C) {
    int bs = 8;
#pragma omp target teams distribute parallel for num_teams(114) thread_limit(64) \
        map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n])
    for (int i_block = 0; i_block < m; i_block += bs) {

        for (int i = i_block; i < i_block + bs && i < m; i++) {
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





void test(double* A, int n) {
#pragma omp target teams distribute parallel for num_teams(114) thread_limit(64) \
        map(tofrom: A[0:n])

    for (int i = 0; i < n; ++i) {

    }
}

}