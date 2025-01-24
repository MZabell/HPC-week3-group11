/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "alloc3d_dev.h"
#include "print.h"
#include <omp.h>
#include "jacobi.h"
#include <cuda_runtime.h>

#define N_DEFAULT 100

int
main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u = NULL;
    double 	***f = NULL;
    double 	***u_2 = NULL;

    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }

    // allocate memory
    double start_mem = omp_get_wtime();
    if ( (u = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (f = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }
    if ( (u_2 = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u_2: allocation failed");
        exit(-1);
    }
    double mem = omp_get_wtime() - start_mem;

    double start, start_calc, calc, elapsed_time, Mlups;

    // ########## CPU ##########
    start = omp_get_wtime();
    #pragma omp parallel default(none) shared(f, u, u_2, N, iter_max, start_T)
    {
        // --------Initialization-----------
        init_u(u, N+2, start_T);
        init_f(f, N+2);
        init_u(u_2, N+2, start_T);
        // --------Calculation-----------
        jacobi(f, u, u_2, N+2, iter_max);
    } // end parallel region
    elapsed_time = omp_get_wtime() - start;
    // Calculate lups
    Mlups = (double)N * N * N * iter_max / elapsed_time / 1e6;
    // Calculate memory footprint
    //int mem_footprint = N * N * N * 3 * 8;
    printf("CPU native\t%f\t%f\t%f\t%f\n", mem, elapsed_time, mem+elapsed_time, Mlups);
    //print_binary("res_bin_cpu.bin", N+2, u);


    // ########## GPU MAP CLAUSE ##########
    start = omp_get_wtime();
    // --------Initialization-----------
    init_u(u, N+2, start_T);
    init_f(f, N+2);
    init_u(u_2, N+2, start_T);
    // --------Calculation-----------
    double warmup_s = omp_get_wtime();
    #pragma omp target data map(to: u_2[:N+2][:N+2][:N+2], f[:N+2][:N+2][:N+2]) map(tofrom: u[:N+2][:N+2][:N+2])
    {
        double dummy = 0.0;
    }
    double warmup = omp_get_wtime() - warmup_s;
    #pragma omp target data map(to: u_2[:N+2][:N+2][:N+2], f[:N+2][:N+2][:N+2]) map(tofrom: u[:N+2][:N+2][:N+2])
    {
        start_calc = omp_get_wtime();
        jacobi_offload(f, u, u_2, N+2, iter_max);
        calc = omp_get_wtime() - start_calc;
    }
    elapsed_time = omp_get_wtime() - start - warmup;
    // Calculate lups
    Mlups = (double)N * N * N * iter_max / elapsed_time / 1e6;
    printf("GPU map_\t%f\t%f\t%f\t%f\n", elapsed_time - calc, calc, elapsed_time, Mlups);
    //print_binary("res_bin_gpu_map.bin", N+2, u);


    // ########## GPU MANUAL COPY ##########
    start = omp_get_wtime();
    // --------Initialization-----------
    init_u(u, N+2, start_T);
    init_f(f, N+2);
    init_u(u_2, N+2, start_T);
    // -------- GPU Memory -----------
    // allocation
    double ***u_d, ***f_d, ***u_2_d;
    double *u_ptr, *f_ptr, *u_2_ptr;
    if ( (u_d = malloc_3d_dev(N+2, N+2, N+2, &u_ptr)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (f_d = malloc_3d_dev(N+2, N+2, N+2, &f_ptr)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }
    if ( (u_2_d = malloc_3d_dev(N+2, N+2, N+2, &u_2_ptr)) == NULL ) {
        perror("array u_2: allocation failed");
        exit(-1);
    }
    // copy data
    omp_target_memcpy(u_ptr, **u, (N+2) * (N+2) * (N+2) * sizeof(double),
                      0, 0, omp_get_default_device(),
                      omp_get_initial_device());
    omp_target_memcpy(f_ptr, **f, (N+2) * (N+2) * (N+2) * sizeof(double),
                      0, 0, omp_get_default_device(),
                      omp_get_initial_device());
    omp_target_memcpy(u_2_ptr, **u_2, (N+2) * (N+2) * (N+2) * sizeof(double),
                      0, 0, omp_get_default_device(),
                      omp_get_initial_device());
    // --------Calculation-----------
    start_calc = omp_get_wtime();
    jacobi_offload_gpuptr(f_d, u_d, u_2_d, N+2, iter_max);
    calc = omp_get_wtime() - start_calc;
    // --------Copy back and free-----------
    omp_target_memcpy(**u, u_ptr, (N+2) * (N+2) * (N+2) * sizeof(double),
                      0, 0, omp_get_initial_device(),
                      omp_get_default_device());
    free_3d_dev(u_d, u_ptr);
    free_3d_dev(f_d, f_ptr);
    free_3d_dev(u_2_d, u_2_ptr);
    elapsed_time = omp_get_wtime() - start;
    // Calculate lups
    Mlups = (double)N * N * N * iter_max / elapsed_time / 1e6;
    printf("GPU memcpy\t%f\t%f\t%f\t%f\n", elapsed_time - calc, calc, elapsed_time, Mlups);
    //print_binary("res_bin_gpu_memcpy.bin", N+2, u);


    // ########## DUAL GPUS ##########
    // Enable peer-to-peer access and offload
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0); // (dev 1, future flag)
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0); // (dev 0, future flag)
    cudaSetDevice(0);
    // --------Initialization-----------
    init_u(u, N+2, start_T);
    init_f(f, N+2);
    init_u(u_2, N+2, start_T);
    // -------- GPU Memory -----------
    // allocation
    for (int i = 0; i < 2; i++) {
        start = omp_get_wtime();
        double ***u_d0, ***f_d0, ***u_2_d0;
        double *u_ptr0, *f_ptr0, *u_2_ptr0;
        omp_set_default_device(0);
        if ( (u_d0 = malloc_3d_dev((N+2)/2, N+2, N+2, &u_ptr0)) == NULL ) {
            perror("array u_d0: allocation failed");
            exit(-1);
        }
        if ( (f_d0 = malloc_3d_dev((N+2)/2, N+2, N+2, &f_ptr0)) == NULL ) {
            perror("array f_d0: allocation failed");
            exit(-1);
        }
        if ( (u_2_d0 = malloc_3d_dev((N+2)/2, N+2, N+2, &u_2_ptr0)) == NULL ) {
            perror("array u_2_d0: allocation failed");
            exit(-1);
        }
        double ***u_d1, ***f_d1, ***u_2_d1;
        double *u_ptr1, *f_ptr1, *u_2_ptr1;
        omp_set_default_device(1);
        if ( (u_d1 = malloc_3d_dev((N+2)/2, N+2, N+2, &u_ptr1)) == NULL ) {
            perror("array u_d1: allocation failed");
            exit(-1);
        }
        if ( (f_d1 = malloc_3d_dev((N+2)/2, N+2, N+2, &f_ptr1)) == NULL ) {
            perror("array f_d1: allocation failed");
            exit(-1);
        }
        if ( (u_2_d1 = malloc_3d_dev((N+2)/2, N+2, N+2, &u_2_ptr1)) == NULL ) {
            perror("array u_2_d1: allocation failed");
            exit(-1);
        }
        // copy data
        omp_set_default_device(0);
        int half_size = (N+2)/2 * (N+2) * (N+2) * sizeof(double);
        omp_target_memcpy(u_ptr0, **u, half_size,
                        0, 0, omp_get_default_device(),
                        omp_get_initial_device());
        omp_target_memcpy(f_ptr0, **f, half_size,
                        0, 0, omp_get_default_device(),
                        omp_get_initial_device());
        omp_target_memcpy(u_2_ptr0, **u_2, half_size,
                        0, 0, omp_get_default_device(),
                        omp_get_initial_device());
        omp_set_default_device(1);
        omp_target_memcpy(u_ptr1, **u, half_size,
                        0, half_size, omp_get_default_device(),
                        omp_get_initial_device());
        omp_target_memcpy(f_ptr1, **f, half_size,
                        0, half_size, omp_get_default_device(),
                        omp_get_initial_device());
        omp_target_memcpy(u_2_ptr1, **u_2, half_size,
                        0, half_size, omp_get_default_device(),
                        omp_get_initial_device());
        // --------Calculation-----------
        start_calc = omp_get_wtime();
        jacobi_offload_dual(f_d0, u_d0, u_2_d0, f_d1, u_d1, u_2_d1,
                            N+2, iter_max);
        calc = omp_get_wtime() - start_calc;
        // --------Copy back and free-----------
        omp_set_default_device(0);
        omp_target_memcpy(**u, u_ptr0, half_size,
                        0, 0, omp_get_initial_device(),
                        omp_get_default_device());
        free_3d_dev(u_d0, u_ptr0);
        free_3d_dev(f_d0, f_ptr0);
        free_3d_dev(u_2_d0, u_2_ptr0);
        omp_set_default_device(1);
        omp_target_memcpy(**u, u_ptr1, half_size,
                        half_size, 0, omp_get_initial_device(),
                        omp_get_default_device());
        free_3d_dev(u_d1, u_ptr1);
        free_3d_dev(f_d1, f_ptr1);
        free_3d_dev(u_2_d1, u_2_ptr1);
        elapsed_time = omp_get_wtime() - start;
        // Calculate lups
        Mlups = (double)N * N * N * iter_max / elapsed_time / 1e6;
    }
    printf("GPU dual\t%f\t%f\t%f\t%f\n", elapsed_time - calc, calc, elapsed_time, Mlups);
    //print_binary("res_bin_gpu_dual.bin", N+2, u);
    omp_set_default_device(0);


    // ########## CPU NORM ##########
    start = omp_get_wtime();
    #pragma omp parallel default(none) shared(f, u, u_2, N, iter_max, tolerance, start_T)
    {
        // --------Initialization-----------
        init_u(u, N+2, start_T);
        init_f(f, N+2);
        init_u(u_2, N+2, start_T);
        // --------Calculation-----------
        jacobi_tol(f, u, u_2, N+2, iter_max, tolerance);
    } // end parallel region
    elapsed_time = omp_get_wtime() - start;
    // Calculate lups
    Mlups = (double)N * N * N * iter_max / elapsed_time / 1e6;
    // Calculate memory footprint
    //int mem_footprint = N * N * N * 3 * 8;
    printf("CPU norm\t%f\t%f\t%f\t%f\n", mem, elapsed_time, elapsed_time + mem, Mlups);
    //print_binary("res_bin_cpu_norm.bin", N+2, u);


    // ########## GPU MAP CLAUSE NORM ##########
    start = omp_get_wtime();
    // --------Initialization-----------
    init_u(u, N+2, start_T);
    init_f(f, N+2);
    init_u(u_2, N+2, start_T);
    // --------Calculation-----------
    #pragma omp target data map(to: u_2[:N+2][:N+2][:N+2], f[:N+2][:N+2][:N+2]) map(tofrom: u[:N+2][:N+2][:N+2])
    {
        start_calc = omp_get_wtime();
        jacobi_offload_tol(f, u, u_2, N+2, iter_max, tolerance);
        calc = omp_get_wtime() - start_calc;
    }
    elapsed_time = omp_get_wtime() - start;
    // Calculate lups
    Mlups = (double)N * N * N * iter_max / elapsed_time / 1e6;
    printf("GPU mapnorm\t%f\t%f\t%f\t%f\n", elapsed_time - calc, calc, elapsed_time, Mlups);
    //print_binary("res_bin_gpu_map_norm.bin", N+2, u);


    // #####################################

    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N+2, u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N+2, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free_3d(u);
    free_3d(u_2);
    free_3d(f);

    return(0);
}
