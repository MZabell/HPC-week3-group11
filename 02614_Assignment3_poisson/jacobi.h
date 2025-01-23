/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

int jacobi(double ***f, double ***u, double ***u_2, int N, int iter_max);
int jacobi_tol(double ***f, double ***u, double ***u_2, int N, int iter_max, double tolerance);
int jacobi_offload(double ***f, double ***u, double ***u_2, int N, int iter_max);
int jacobi_offload_tol(double ***f, double ***u, double ***u_2, int N, int iter_max, double tolerance);
int jacobi_offload_gpuptr(double ***f, double ***u, double ***u_2, int N, int iter_max);
int jacobi_offload_dual(double ***f0, double ***u0, double ***u_20,
                        double ***f1, double ***u1, double ***u_21, int N, int iter_max);

#endif
