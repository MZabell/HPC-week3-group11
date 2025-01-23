#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}



/* Collapse version of transpose */

/*
void transpose_per_elm(double **A, double **At)
{
#pragma omp target teams distribute parallel for \
                num_teams(N*N/64) thread_limit(64)
    for (int idx = 0; idx < N * N; idx++) {
        int i = idx / N; // Calculate the row index
        int j = idx % N; // Calculate the column index
        At[i][j] = A[j][i];
    }
}*/


// https://developer.nvidia.com/nsight-compute