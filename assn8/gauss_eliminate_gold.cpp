#include <stdio.h>
#include <stdlib.h>
extern "C" int compute_gold(float*, const float*, unsigned int);

int compute_gold(float* U, const float* A, unsigned int matrix_dim)
{
	unsigned int i, j, k;
    float pivot;

	/* Copy contents of the A matrix into the U matrix */
    for (i = 0; i < matrix_dim; i ++)
		for(j = 0; j < matrix_dim; j++)
			U[matrix_dim * i + j] = A[matrix_dim * i + j];

	/* Perform Gaussian elimination in place on the U matrix */
	for (k = 0; k < matrix_dim; k++) {
        pivot = U[matrix_dim * k + k];
        /* DO NOT HAVE TO PERFORM THIS CHECK IN YOUR PARALLEL CODE. */
        if (pivot == 0) {
            printf("Numerical instability detected. Principle diagonal element is zero at row %d\n", matrix_dim);
            return 0;
        }

		for (j = (k + 1); j < matrix_dim; j++) /* Reduce current row */
            U[matrix_dim * k + j] = (float) U[matrix_dim * k + j] / pivot; /* Division step */

        U[matrix_dim * k + k] = 1; /* Set pivot element to 1 */
        for (i = (k+1); i < matrix_dim; i++){ /* Elimination step. */
			for (j = (k+1); j < matrix_dim; j++)
				U[matrix_dim * i + j] = U[matrix_dim * i + j] - (U[matrix_dim * i + k] * U[matrix_dim * k + j]);

			U[matrix_dim * i + k] = 0;
		}
	}	

    return 1;
}
