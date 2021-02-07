/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: January 28, 2021
 *
 * Student name(s): Kevin Connell, Casey Adams
 * Date modified: 2/6/2021
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "jacobi_solver.h"
#include <pthread.h>
/* Uncomment the line below to spit out debug information */
/* #define DEBUG */

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s matrix-size num-threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
        fprintf(stderr, "num-threads: number of worker threads to create\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    matrix_t A;                /* N x N constant matrix */
    matrix_t B;                /* N x 1 b matrix */
    matrix_t reference_x;      /* Reference solution */
    matrix_t mt_solution_x_v1; /* Solution computed by pthread code using chunking */
    matrix_t mt_solution_x_v2; /* Solution computed by pthread code using striding */

    /* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
    srand(time(NULL));
    A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
    if (A.elements == NULL)
    {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
    }

    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
    reference_x = allocate_matrix(matrix_size, 1, 0);
    mt_solution_x_v1 = allocate_matrix(matrix_size, 1, 0);
    mt_solution_x_v2 = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
    print_matrix(A);
    print_matrix(B);
    print_matrix(reference_x);
#endif

    /* Compute Jacobi solution using reference code */
    fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));
    /* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x_v1 and mt_solution_x_v2.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using chunking\n");
    gettimeofday(&start, NULL);
    compute_using_pthreads_v1(A, mt_solution_x_v1, B, max_iter, num_threads);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, mt_solution_x_v1, B); /* Display statistics */
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using striding\n");
    gettimeofday(&start, NULL);
    compute_using_pthreads_v2(A, mt_solution_x_v2, B, max_iter, num_threads);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, mt_solution_x_v2, B); /* Display statistics */
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    free(A.elements);
    free(B.elements);
    free(reference_x.elements);
    free(mt_solution_x_v1.elements);
    free(mt_solution_x_v2.elements);

    exit(EXIT_SUCCESS);
}

pthread_barrier_t barrier_v1;

struct Pthread_v1_args
{
    /**
     * Struct used to store task data for use by the chunking threads. Each thread gets
     * the same instance of this struct.
     */
    matrix_t a;
    int num_elements;
    long num_threads;
    int done;
    matrix_t *x1;
    matrix_t *x2;
    matrix_t target_mat;
    matrix_t b;
    double *ssd_vals;
    int max_iter;
};
struct Pthread_v1_args p_v1_args;

void *PThread_v1_func(void *rank)
{
    /**
     * Function that runs in each chunking thread.
     */
    long thread_rank = (long)rank; //This is the nth thread
    matrix_t a = p_v1_args.a;
    int num_elements = p_v1_args.num_elements;
    long threads = p_v1_args.num_threads;
    matrix_t *x1 = p_v1_args.x1;
    matrix_t *x2 = p_v1_args.x2;
    matrix_t b = p_v1_args.b;
    double *ssd_mat = p_v1_args.ssd_vals;
    int segment_size = num_elements / threads;

    int max_iter = p_v1_args.max_iter;
    int i, j;
    int num_cols = a.num_columns;

    /* Initialize current jacobi solution. */
    for (i = segment_size * thread_rank; i < num_elements; i++)
    {
        x1->elements[i] = b.elements[i];
    }
    /* Perform Jacobi iteration. */
    double ssd, mse;
    int num_iter = 0;
    matrix_t *x;
    matrix_t *new_x;
    while (!p_v1_args.done)
    {
        if (num_iter % 2 == 0)
        {
            x = x1;
            new_x = x2;
        }
        else
        {
            x = x2;
            new_x = x1;
        }
        for (i = segment_size * thread_rank; i < num_elements; i++)
        {

            double sum = 0.0;
            for (j = 0; j < num_cols; j++)
            {
                if (i != j)
                    sum += a.elements[i * num_cols + j] * x->elements[j];
            }

            /* Update values for the unkowns for the current row. */
            new_x->elements[i] = (b.elements[i] - sum) / a.elements[i * num_cols + i];
        }
        ssd = 0.0;
        for (i = segment_size * thread_rank; i < num_elements; i++)
        {
            ssd += (new_x->elements[i] - x->elements[i]) * (new_x->elements[i] - x->elements[i]);
        }
        num_iter++;
        ssd_mat[thread_rank] = ssd;
        pthread_barrier_wait(&barrier_v1);
        if (thread_rank == 0)
        {
            double sum = 0;
            for (i = 0; i < threads; i++)
            {
                sum += ssd_mat[i];
            }
            mse = sqrt(sum);
            if ((mse <= THRESHOLD) || (num_iter == max_iter))
            {
                p_v1_args.done = 1;
                p_v1_args.target_mat = *new_x;
            }
        }
        pthread_barrier_wait(&barrier_v1);
    }
    return NULL;
}

void compute_using_pthreads_v1(const matrix_t A, matrix_t mt_sol_x_v1, const matrix_t B, int max_iter, int num_threads)
{
    /**
     * Initializes, runs, and closes threads for chunking solution
     */
    // Create thread handles
    long thread;
    pthread_t *thread_handles;
    pthread_barrier_init(&barrier_v1, NULL, num_threads);
    thread_handles = malloc(num_threads * sizeof(pthread_t));

    // Setup struct with all necessary information for thread
    p_v1_args.a = A;
    p_v1_args.num_elements = mt_sol_x_v1.num_rows;
    p_v1_args.num_threads = num_threads;
    p_v1_args.x1 = &mt_sol_x_v1;
    matrix_t mt_sol_x2_v1 = allocate_matrix(mt_sol_x_v1.num_rows, 1, 0);
    p_v1_args.x2 = &mt_sol_x2_v1;
    p_v1_args.b = B;
    p_v1_args.done = 0;
    p_v1_args.target_mat = allocate_matrix(mt_sol_x_v1.num_rows, 1, 0);
    double ssd_vals[num_threads];
    p_v1_args.ssd_vals = ssd_vals;
    p_v1_args.max_iter = max_iter;

    // Create threads
    for (thread = 0; thread < num_threads; thread++)
    {
        pthread_create(&thread_handles[thread], NULL, PThread_v1_func, (void *)thread);
    }
    // Close threads
    for (thread = 0; thread < num_threads; thread++)
    {
        pthread_join(thread_handles[thread], NULL);
    }
    mt_sol_x_v1 = p_v1_args.target_mat;
    // Clean-up
    free(thread_handles);
    pthread_barrier_destroy(&barrier_v1);
}

pthread_barrier_t barrier_v2;

struct Pthread_v2_args
{
    /**
     * Struct used to store task data for use by striding threads. Each thread gets
     * the same instance of this struct.
     */
    matrix_t a;
    int num_elements;
    long num_threads;
    int done;
    matrix_t *x1;
    matrix_t *x2;
    matrix_t *target_mat;
    matrix_t b;
    double *ssd_vals;
    int max_iter;
};
struct Pthread_v2_args p_v2_args;

void *PThread_v2_func(void *rank)
{
    /**
     * Function run by each striding thread
     */
    long thread_rank = (long)rank; //This is the nth thread
    matrix_t a = p_v2_args.a;
    int num_elements = p_v2_args.num_elements;
    long threads = p_v2_args.num_threads;
    matrix_t *x1 = p_v2_args.x1;
    matrix_t *x2 = p_v2_args.x2;
    matrix_t b = p_v2_args.b;
    double *ssd_mat = p_v2_args.ssd_vals;

    int max_iter = p_v2_args.max_iter;
    int i, j;
    int num_cols = a.num_columns;

    /* Initialize current jacobi solution. */
    for (i = thread_rank; i < num_elements; i += threads)
        x1->elements[i] = b.elements[i];

    /* Perform Jacobi iteration. */
    double ssd, mse;
    int num_iter = 0;
    matrix_t *x;
    matrix_t *new_x;
    while (!p_v2_args.done)
    {
        if (num_iter % 2 == 0)
        {
            x = x1;
            new_x = x2;
        }
        else
        {
            x = x2;
            new_x = x1;
        }
        for (i = thread_rank; i < num_elements; i += threads)
        {

            double sum = 0.0;
            for (j = 0; j < num_cols; j++)
            {
                if (i != j)
                    sum += a.elements[i * num_cols + j] * x->elements[j];
            }

            /* Update values for the unkowns for the current row. */
            new_x->elements[i] = (b.elements[i] - sum) / a.elements[i * num_cols + i];
        }
        ssd = 0.0;
        for (i = thread_rank; i < num_elements; i += threads)
        {
            ssd += (new_x->elements[i] - x->elements[i]) * (new_x->elements[i] - x->elements[i]);
        }
        num_iter++;
        ssd_mat[thread_rank] = ssd;
        pthread_barrier_wait(&barrier_v2);
        if (thread_rank == 0)
        {
            double sum = 0;
            for (i = 0; i < threads; i++)
            {
                sum += ssd_mat[i];
            }
            mse = sqrt(sum);
            if ((mse <= THRESHOLD) || (num_iter == max_iter))
            {
                p_v2_args.done = 1;
                p_v2_args.target_mat = new_x;
                fprintf(stderr, "Iterations: %d\n", num_iter);
            }
        }
        pthread_barrier_wait(&barrier_v2);
    }
    return NULL;
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads using chunking. 
 * Result must be placed in mt_sol_x_v2. */
void compute_using_pthreads_v2(const matrix_t A, matrix_t mt_sol_x_v2, const matrix_t B, int max_iter, int num_threads)
{
    /**
     * Initializes, runs, and closes threads for chunking solution
     */
    // Create thread handles
    long thread;
    pthread_t *thread_handles;
    pthread_barrier_init(&barrier_v2, NULL, num_threads);
    thread_handles = malloc(num_threads * sizeof(pthread_t));

    // Setup struct with all necessary information for thread
    p_v2_args.a = A;
    p_v2_args.num_elements = mt_sol_x_v2.num_rows;
    p_v2_args.num_threads = num_threads;
    p_v2_args.x1 = &mt_sol_x_v2;
    matrix_t mt_sol_x2_v2 = allocate_matrix(mt_sol_x_v2.num_rows, 1, 0);
    p_v2_args.x2 = &mt_sol_x2_v2;
    p_v2_args.b = B;
    p_v2_args.done = 0;
    p_v2_args.target_mat = p_v2_args.x2;
    double ssd_vals[num_threads];
    p_v2_args.ssd_vals = ssd_vals;
    p_v2_args.max_iter = max_iter;

    // Create threads
    for (thread = 0; thread < num_threads; thread++)
    {
        pthread_create(&thread_handles[thread], NULL, PThread_v2_func, (void *)thread);
    }
    // Close threads
    for (thread = 0; thread < num_threads; thread++)
    {
        pthread_join(thread_handles[thread], NULL);
    }
    mt_sol_x_v2 = *p_v2_args.target_mat;
    // Clean-up
    free(thread_handles);
    pthread_barrier_destroy(&barrier_v2);
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

    M.elements = (float *)malloc(size * sizeof(float));
    for (i = 0; i < size; i++)
    {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }

    return M;
}

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
    for (i = 0; i < M.num_rows; i++)
    {
        for (j = 0; j < M.num_columns; j++)
        {
            fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }

        fprintf(stderr, "\n");
    }

    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand() / (float)RAND_MAX;
    return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
    float diag_element;
    float sum;
    for (i = 0; i < M.num_rows; i++)
    {
        sum = 0.0;
        diag_element = M.elements[i * M.num_rows + i];
        for (j = 0; j < M.num_columns; j++)
        {
            if (i != j)
                sum += fabsf(M.elements[i * M.num_rows + j]);
        }

        if (diag_element <= sum)
            return -1;
    }

    return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(int num_rows, int num_columns)
{
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
    fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
    for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);

    /* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
    for (i = 0; i < num_rows; i++)
    {
        row_sum = 0.0;
        for (j = 0; j < num_columns; j++)
        {
            row_sum += fabs(M.elements[i * M.num_rows + j]);
        }

        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
    }

    /* Check if matrix is diagonal dominant */
    if (check_if_diagonal_dominant(M) < 0)
    {
        free(M.elements);
        M.elements = NULL;
    }

    return M;
}
