#ifndef __ALLOC_3D
#define __ALLOC_3D

double ***malloc_3d(int m, int n, int k);

void init_u(double ***u, int N, double start_T);

void init_f(double ***f, int N);

#define HAS_FREE_3D
void free_3d(double ***array3D);

#endif /* __ALLOC_3D */
