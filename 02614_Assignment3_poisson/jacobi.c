/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>

void
jacobi(double ***f, double ***u, double ***u_2, int N, int iter_max) {
    double ***temp;
    double delta_2 = 4.0 / (N*N);
    int iter = 0;
    while(iter < iter_max) {
        iter++;
        #pragma omp for schedule(static)
        for(int i = 1; i < N-1; i++) {
            for(int j = 1; j < N-1; j++) {
                for(int k = 1; k < N-1; k++) {
                    u_2[i][j][k] = (u[i-1][j][k] + u[i+1][j][k] +
                                    u[i][j-1][k] + u[i][j+1][k] +
                                    u[i][j][k-1] + u[i][j][k+1] +
                                    delta_2 * f[i][j][k]) / 6.0;
                }
            }
        }
        // Swap u and u_2
        temp = u;
        u = u_2;
        u_2 = temp;
    }
}

void
jacobi_tol(double ***f, double ***u, double ***u_2, int N, int iter_max, double tolerance) {
    double ***temp;
    double delta_2 = 4.0 / (N*N);
    int iter = 0;
    double sum = INFINITY;
    double tolerance_2 = tolerance * tolerance;
    while(iter < iter_max && sum > tolerance_2) {
        sum = 0.0;
        iter++;
        #pragma omp for schedule(static) reduction(+:sum)
        for(int i = 1; i < N-1; i++) {
            for(int j = 1; j < N-1; j++) {
                for(int k = 1; k < N-1; k++) {
                    u_2[i][j][k] = (u[i-1][j][k] + u[i+1][j][k] +
                                    u[i][j-1][k] + u[i][j+1][k] +
                                    u[i][j][k-1] + u[i][j][k+1] +
                                    delta_2 * f[i][j][k]) / 6.0;
                    //double c = u_2[i][j][k];
                    sum += (u_2[i][j][k] - u[i][j][k]) * (u_2[i][j][k] - u[i][j][k]); //SEGFAULT
                }
            }
        }
        // Swap u and u_2
        temp = u;
        u = u_2;
        u_2 = temp;
        //printf("iter: %i\tsum:%f\n", iter, sum);
    }
}

void
jacobi_offload(double ***f, double ***u, double ***u_2, int N, int iter_max) {
    double ***temp;
    double delta_2 = 4.0 / (N*N);
    int iter = 0;
    while(iter < iter_max) {
        iter++;
        #pragma omp target teams distribute parallel for collapse(2) \
        default(none) shared(f, u, u_2, N, delta_2) \
        num_teams(ceil(N*N/512.0)) thread_limit(512)
        for(int i = 1; i < N-1; i++) {
            for(int k = 1; k < N-1; k++) {
                for(int j = 1; j < N-1; j++) {
                    u_2[i][j][k] = (u[i-1][j][k] + u[i+1][j][k] +
                                    u[i][j-1][k] + u[i][j+1][k] +
                                    u[i][j][k-1] + u[i][j][k+1] +
                                    delta_2 * f[i][j][k]) / 6.0;
                }
            }
        }
        // Swap u and u_2
        temp = u;
        u = u_2;
        u_2 = temp;
    }
}

void
jacobi_offload_tol(double ***f, double ***u, double ***u_2, int N, int iter_max, double tolerance) {
    double ***temp;
    double delta_2 = 4.0 / (N*N);
    int iter = 0;
    double sum = INFINITY;
    double tolerance_2 = tolerance * tolerance;
    while(iter < iter_max && sum > tolerance_2) {
        iter++;
        sum = 0.0;
        #pragma omp target teams distribute parallel for collapse(2) \
        default(none) shared(f, u, u_2, N, delta_2) \
        num_teams(ceil(N*N/512.0)) thread_limit(512) reduction(+:sum)
        for(int i = 1; i < N-1; i++) {
            for(int k = 1; k < N-1; k++) {
                for(int j = 1; j < N-1; j++) {
                    u_2[i][j][k] = (u[i-1][j][k] + u[i+1][j][k] +
                                    u[i][j-1][k] + u[i][j+1][k] +
                                    u[i][j][k-1] + u[i][j][k+1] +
                                    delta_2 * f[i][j][k]) / 6.0;
                    sum += (u_2[i][j][k] - u[i][j][k]) * (u_2[i][j][k] - u[i][j][k]);
                }
            }
        }
        // Swap u and u_2
        temp = u;
        u = u_2;
        u_2 = temp;
        //#pragma omp single
        //#pragma omp taskwait
        //printf("iter: %i\tsum:%f\n", iter, sum);
    }
}

void
jacobi_offload_gpuptr(double ***f, double ***u, double ***u_2, int N, int iter_max) {
    double ***temp;
    double delta_2 = 4.0 / (N*N);
    int iter = 0;
    while(iter < iter_max) {
        iter++;
        #pragma omp target teams distribute parallel for collapse(2) \
        is_device_ptr(f, u, u_2) num_teams(ceil(N*N/512.0)) thread_limit(512)
        for(int i = 1; i < N-1; i++) {
            for(int k = 1; k < N-1; k++) {
                for(int j = 1; j < N-1; j++) {
                    u_2[i][j][k] = (u[i-1][j][k] + u[i+1][j][k] +
                                    u[i][j-1][k] + u[i][j+1][k] +
                                    u[i][j][k-1] + u[i][j][k+1] +
                                    delta_2 * f[i][j][k]) / 6.0;
                    
                }
            }
        }
        // Swap u and u_2
        temp = u;
        u = u_2;
        u_2 = temp;
    }
}

void
jacobi_offload_dual(double ***f0, double ***u0, double ***u_20,
                    double ***f1, double ***u1, double ***u_21, int N, int iter_max) {
    double ***temp0, ***temp1;
    double delta_2 = 4.0 / (N*N);
    int iter = 0;
    while(iter < iter_max) {
        iter++;
        omp_set_default_device(0);
        #pragma omp target teams distribute parallel for collapse(2) nowait \
        is_device_ptr(f0, u0, u1, u_20) num_teams(ceil(N*N/512.0)) thread_limit(512) device(0)
        for(int i = 1; i < N/2; i++) {
            for(int k = 1; k < N-1; k++) {
                double **u_ip1;
                if(i == N/2-1) {
                    u_ip1 = u1[0];
                }
                else {
                    u_ip1 = u0[i+1];
                }
                for(int j = 1; j < N-1; j++) {
                    /*double u_ip1jk;
                    if(i == N/2-1) {
                        u_ip1jk = u1[0][j][k];
                    }
                    else {
                        u_ip1jk = u0[i+1][j][k];
                    }*/
                    u_20[i][j][k] = (u0[i-1][j][k] + u_ip1[j][k] +
                                    u0[i][j-1][k] + u0[i][j+1][k] +
                                    u0[i][j][k-1] + u0[i][j][k+1] +
                                    delta_2 * f0[i][j][k]) / 6.0;
                }
            }
        }
        omp_set_default_device(1);
        #pragma omp target teams distribute parallel for collapse(2) nowait \
        is_device_ptr(f1, u0, u1, u_21) num_teams(ceil(N*N/512.0)) thread_limit(512) device(1)
        for(int i = 0; i < N/2-1; i++) {
            for(int k = 1; k < N-1; k++) {
                double **u_im1;
                if(i == 0) {
                    u_im1 = u0[N/2-1];
                }
                else {
                    u_im1 = u1[i-1];
                }
                for(int j = 1; j < N-1; j++) {
                    /*double u_im1jk;
                    if(i == 0) {
                        u_im1jk = u0[N/2-1][j][k];
                    }
                    else {
                        u_im1jk = u1[i-1][j][k];
                    }*/
                    u_21[i][j][k] = (u_im1[j][k] + u1[i+1][j][k] +
                                    u1[i][j-1][k] + u1[i][j+1][k] +
                                    u1[i][j][k-1] + u1[i][j][k+1] +
                                    delta_2 * f1[i][j][k]) / 6.0;
                }
            }
        }
        #pragma omp taskwait
        // Swap u and u_2
        temp0 = u0;
        u0 = u_20;
        u_20 = temp0;
        temp1 = u1;
        u1 = u_21;
        u_21 = temp1;
    }
}