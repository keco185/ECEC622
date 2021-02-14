/* Implementation of PSO using OpenMP.
 *
 * Author: Naga Kandasamy
 * Date: February 5, 2021
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "pso.h"
#include <omp.h>
int pso_get_best_fitness_omp(swarm_t *swarm, int num_threads);

/* Solve PSO */
int pso_solve_omp(char *function, swarm_t *swarm, 
                    float xmax, float xmin, int max_iter, int thread_count)
{
    int i, j, iter, g;
    float w, c1, c2;
    float r1, r2;
    float curr_fitness;
    particle_t *particle, *gbest;

    w = 0.79;
    c1 = 1.49;
    c2 = 1.49;
    iter = 0;
    g = -1;
    while (iter < max_iter) {
        # pragma omp parallel for num_threads(thread_count) private(i, j, r1, r2, curr_fitness, particle) shared(w, c1, c2, swarm, function, xmax, xmin)
        for (i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            gbest = &swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
            for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                unsigned int seed = (unsigned int) omp_get_thread_num();
                r1 = (float)rand_r(&seed)/(float)RAND_MAX;
                r2 = (float)rand_r(&seed)/(float)RAND_MAX;
                /* Update particle velocity */
                particle->v[j] = w * particle->v[j]\
                                 + c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                 + c2 * r2 * (gbest->x[j] - particle->x[j]);
                /* Clamp velocity */
                if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) 
                    particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

                /* Update particle position */
                particle->x[j] = particle->x[j] + particle->v[j];
                if (particle->x[j] > xmax)
                    particle->x[j] = xmax;
                if (particle->x[j] < xmin)
                    particle->x[j] = xmin;
            } /* State update */
            
            /* Evaluate current fitness */
            pso_eval_fitness(function, particle, &curr_fitness);

            /* Update pbest */
            if (curr_fitness < particle->fitness) {
                particle->fitness = curr_fitness;
                for (j = 0; j < particle->dim; j++)
                    particle->pbest[j] = particle->x[j];
            }
        } /* Particle loop */

        /* Identify best performing particle */
        g = pso_get_best_fitness_omp(swarm, thread_count);
        for (i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            particle->g = g;
        }

#ifdef SIMPLE_DEBUG
        /* Print best performing particle */
        fprintf(stderr, "\nIteration %d:\n", iter);
        pso_print_particle(&swarm->particle[g]);
#endif
        iter++;
    } /* End of iteration */
    return g;
}

/* Return index of best performing particle */
int pso_get_best_fitness_omp(swarm_t *swarm, int num_threads)
{
    float best_fitness = INFINITY;
    int g = -1;
    #pragma omp parallel num_threads(num_threads)
    {
        int local_g = -1;
        float local_best_fitness = INFINITY;
        particle_t *particle;
        #pragma omp for nowait
        for (int i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            if (particle->fitness < local_best_fitness) {
                local_best_fitness = particle->fitness;
                local_g = i;
            }
        }
        #pragma omp critical
        {
            if (local_best_fitness < best_fitness) {
                g = local_g;
                best_fitness = local_best_fitness;
            }
        }
    }
    return g;
}


/* Initialize PSO using OpenMP */
swarm_t *pso_init_omp(char *function, int dim, int swarm_size, 
                  float xmin, float xmax, int num_threads)
{
    int i, j, g;
    int status;
    float fitness;
    swarm_t *swarm;
    particle_t *particle;

    swarm = (swarm_t *)malloc(sizeof(swarm_t));
    swarm->num_particles = swarm_size;
    swarm->particle = (particle_t *)malloc(swarm_size * sizeof(particle_t));
    if (swarm->particle == NULL)
        return NULL;
    #pragma omp parallel for num_threads(num_threads) private(fitness, i, j, status, particle)
    for (i = 0; i < swarm->num_particles; i++) {
        particle = &swarm->particle[i];
        particle->dim = dim; 
        /* Generate random particle position */
        particle->x = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
           particle->x[j] = uniform(xmin, xmax);

       /* Generate random particle velocity */ 
        particle->v = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
            particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

        /* Initialize best position for particle */
        particle->pbest = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
            particle->pbest[j] = particle->x[j];

        /* Initialize particle fitness */
        status = pso_eval_fitness(function, particle, &fitness);
        if (status < 0) {
            fprintf(stderr, "Could not evaluate fitness. Unknown function provided.\n");
            continue;
        }
        particle->fitness = fitness;

        /* Initialize index of best performing particle */
        particle->g = -1;
    }

    /* Get index of particle with best fitness */
    g = pso_get_best_fitness_omp(swarm, num_threads);
    for (i = 0; i < swarm->num_particles; i++) {
        particle = &swarm->particle[i];
        particle->g = g;
    }

    return swarm;
}

int optimize_using_omp(char *function, int dim, int swarm_size, 
                       float xmin, float xmax, int max_iter, int num_threads)
{
     /* Initialize PSO */
    swarm_t *swarm;
    srand(time(NULL));
    swarm = pso_init_omp(function, dim, swarm_size, xmin, xmax, num_threads);
    if (swarm == NULL) {
        fprintf(stderr, "Unable to initialize PSO\n");
        exit(EXIT_FAILURE);
    }

    /* Solve PSO */
    int g = pso_solve_omp(function, swarm, xmax, xmin, max_iter, num_threads);
    if (g >= 0) {
        //fprintf(stderr, "Solution:\n");
        //pso_print_particle(&swarm->particle[g]);
    }

    pso_free(swarm);
    return g;
}
