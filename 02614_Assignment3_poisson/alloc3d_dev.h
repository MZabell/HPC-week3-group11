#ifndef __ALLOC_3D_DEV
#define __ALLOC_3D_DEV

double ***malloc_3d_dev(int m, int n, int k, double **data);

void free_3d_dev(double ***array3D, double *data);

#endif /* __ALLOC_3D */
