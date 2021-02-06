/* Implementation of the SAXPY loop.
 *
 * Compile as follows: gcc -o saxpy saxpy.c -O3 -Wall -std=c99 -lpthread -lm
 *
 * Author: Naga Kandasamy
 * Date created: April 14, 2020
 * Date modified: January 19, 2021
 *
 * Student names: Kevin Connell, Casey Adams
 * Date: 1/27/2021
 *
 * */

#define _REENTRANT /* Make sure the library functions are MT (muti-thread) safe */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

/* Function prototypes */
void compute_gold(float *, float *, float, int);
void compute_using_pthreads_v1(float *, float *, float, int, int);
void compute_using_pthreads_v2(float *, float *, float, int, int);
int check_results(float *, float *, int, float);

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s num-elements num-threads\n", argv[0]);
        fprintf(stderr, "num-elements: Number of elements in the input vectors\n");
        fprintf(stderr, "num-threads: Number of threads\n");
		exit(EXIT_FAILURE);
	}

    int num_elements = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

	/* Create vectors X and Y and fill them with random numbers between [-.5, .5] */
    fprintf(stderr, "Generating input vectors\n");
    int i;
	float *x = (float *)malloc(sizeof(float) * num_elements);
    float *y1 = (float *)malloc(sizeof(float) * num_elements);              /* For the reference version */
	float *y2 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 1 */
	float *y3 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 2 */

	srand(time(NULL)); /* Seed random number generator */
	for (i = 0; i < num_elements; i++) {
		x[i] = rand()/(float)RAND_MAX - 0.5;
		y1[i] = rand()/(float)RAND_MAX - 0.5;
        y2[i] = y1[i]; /* Make copies of y1 for y2 and y3 */
        y3[i] = y1[i];
	}

    float a = 2.5;  /* Choose some scalar value for a */

	/* Calculate SAXPY using the reference solution. The resulting values are placed in y1 */
    fprintf(stderr, "\nCalculating SAXPY using reference solution\n");
	struct timeval start, stop;
	gettimeofday(&start, NULL);

    compute_gold(x, y1, a, num_elements);

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Compute SAXPY using pthreads, version 1. Results must be placed in y2 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 1\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v1(x, y2, a, num_elements, num_threads);

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Compute SAXPY using pthreads, version 2. Results must be placed in y3 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 2\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v2(x, y3, a, num_elements, num_threads);

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Check results for correctness */
    fprintf(stderr, "\nChecking results for correctness\n");
    float eps = 1e-12;                                      /* Do not change this value */
    if (check_results(y1, y2, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else
        fprintf(stderr, "TEST FAILED\n");

    if (check_results(y1, y3, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else
        fprintf(stderr, "TEST FAILED\n");

	/* Free memory */
	free((void *)x);
	free((void *)y1);
    free((void *)y2);
	free((void *)y3);

    exit(EXIT_SUCCESS);
}

/* Compute reference soution using a single thread */
void compute_gold(float *x, float *y, float a, int num_elements)
{
	int i;
    for (i = 0; i < num_elements; i++)
        y[i] = a * x[i] + y[i];
}








// ---------- ALL CODE FOR PTHREADS V1 ---------- //
void *PThread_v1_func(void* rank);

struct Pthread_v1_args {
    /**
     * Struct used to store task data for use by each thread. Each thread gets
     * the same instance of this struct.
     */
    float a;
    int num_elements;
    long num_threads;
    float *x;
    float *y;
}; struct Pthread_v1_args p_v1_args;

/* Calculate SAXPY using pthreads, version 1. Place result in the Y vector */
void compute_using_pthreads_v1(float *x, float *y, float a, int num_elements, int num_threads) {
    // Create thread handles
    long thread;
    pthread_t* thread_handles;
    thread_handles = malloc(num_threads*sizeof(pthread_t));

    // Setup struct with all necessary information for thread
    p_v1_args.a = a;
    p_v1_args.num_elements = num_elements;
    p_v1_args.num_threads = num_threads;
    p_v1_args.x = x;
    p_v1_args.y = y;

    // Create threads
    for (thread = 0; thread < num_threads; thread++) {
        pthread_create(&thread_handles[thread], NULL, PThread_v1_func, (void*) thread);
    }

    // Close threads
    for (thread = 0; thread < num_threads; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    // Clean-up
    free(thread_handles);

}

void *PThread_v1_func(void* rank) {
    /**
     * Function that runs on each thread for pthreads v1.
     * Runs saxby code on a chunk from segment_size*rank to segment_size*(rank+1).
     * The last thread will run from segment_size*(num_threads-1) to num_elements.
     * Information regarding values for a, num_elements, num_threads, and x/y vectors comes from
     * a global instance of the Pthread_v1_args struct.
     */
    // Create variables
    long thread_rank = (long) rank; //This is the nth thread
    float a = p_v1_args.a;
    int num_elements = p_v1_args.num_elements;
    long threads = p_v1_args.num_threads;
    float *x = p_v1_args.x;
    float *y = p_v1_args.y;
    int i;
    int segment_size = num_elements/threads;

    // If this is not the last thread, pick the chunk from segment_size*rank to segment_size*(rank+1)
    if (thread_rank < threads-1) {
        int end = segment_size*(thread_rank+1);
        for (i = segment_size*thread_rank; i < end; i++) {
            y[i] = a * x[i] + y[i];
        }

    // If this IS the last thread, pick the chunk from segment_size*(num_threads-1) to num_elements
    // This protects against arrays that aren't evenly divisible by the thread count.
    } else {
        for (i = segment_size*thread_rank; i < num_elements; i++) {
            y[i] = a * x[i] + y[i];
        }
    }
    return NULL;
}










// ---------- ALL CODE FOR PTHREADS V2 ---------- //

void *PThread_v2_func(void* rank);

struct Pthread_v2_args {
    /**
     * Struct used to store task data for use by each thread. Each thread gets
     * the same instance of this struct.
     */
    float a;
    int num_elements;
    long num_threads;
    float *x;
    float *y;
}; struct Pthread_v2_args p_v2_args;

/* Calculate SAXPY using pthreads, version 2. Place result in the Y vector */
void compute_using_pthreads_v2(float *x, float *y, float a, int num_elements, int num_threads) {
    // Create thread handles
    long thread;
    pthread_t* thread_handles;
    thread_handles = malloc(num_threads*sizeof(pthread_t));

    // Setup struct with all necessary information for thread
    p_v2_args.a = a;
    p_v2_args.num_elements = num_elements;
    p_v2_args.num_threads = num_threads;
    p_v2_args.x = x;
    p_v2_args.y = y;

    // Create threads
    for (thread = 0; thread < num_threads; thread++) {
        pthread_create(&thread_handles[thread], NULL, PThread_v2_func, (void*) thread);
    }

    // Close threads
    for (thread = 0; thread < num_threads; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    // Clean-up
    free(thread_handles);

}

void *PThread_v2_func(void* rank) {
    /**
     * Function that runs on each thread for pthreads v2.
     * Runs saxby code on every nth element of the arrays where n is the number
     * of threads. The first element is in position = thread_rank.
     */
    // Create variables
    long thread_rank = (long) rank; //This is the nth thread
    float a = p_v2_args.a;
    int num_elements = p_v2_args.num_elements;
    long threads = p_v2_args.num_threads;
    float *x = p_v2_args.x;
    float *y = p_v2_args.y;
    int i;
    for (i = thread_rank; i < num_elements; i += threads) {
        y[i] = a * x[i] + y[i];
    }

    return NULL;
}



/* END calculate SAXPY using pthreads, version 2. Place result in the Y vector */







/* Perform element-by-element check of vector if relative error is within specified threshold */
int check_results(float *A, float *B, int num_elements, float threshold)
{
    int i;
    for (i = 0; i < num_elements; i++) {
        if (fabsf((A[i] - B[i])/A[i]) > threshold)
            return -1;
    }

    return 0;
}
