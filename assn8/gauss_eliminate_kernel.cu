 /* Device code. */
#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel(float *U, int k) {
	// DIVISION
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
        int row = k * MATRIX_SIZE;
	if (tx > k) {
                  U[row+tx] = __fdiv_rn(U[row+tx],U[row+k]);
	}
	__syncthreads();
	if (tx == k) {
		U[row+k] = 1;
	}


	// ELIMINATION
        if (tx <= k) {
            return;
        }
	int elim_row = tx*MATRIX_SIZE;
	float div = U[elim_row+k];
	for (int i = k+1; i < MATRIX_SIZE; i++) {
		U[elim_row+i] = __fsub_rn(U[elim_row+i], __fmul_rn(div,U[row+i]));
	}
	U[elim_row+k] = 0;
}
