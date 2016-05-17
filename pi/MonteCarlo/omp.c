#include <omp.h>
#include <stdio.h>
#include "dsfmt/dSFMT.h"

#ifdef NUM_THREADS
#else
    #define NUM_THREADS 8
#endif 

int main(int argc, char* argv[]) {
    printf("running %d threads\n",NUM_THREADS);
    omp_set_num_threads(NUM_THREADS);
    int i, inside = 0, seed;
    double x, y, pi;
    const long n_steps = 1000000000;
    #pragma omp parallel private(x,y)
    {
        int id = omp_get_thread_num();
        dsfmt_t dsfmt;
        seed = 142857 + id;
        dsfmt_init_gen_rand(&dsfmt, seed);
        int t = 0;
        for (long i = id; i < n_steps; i+=NUM_THREADS) {
            x = dsfmt_genrand_close_open(&dsfmt);
            y = dsfmt_genrand_close_open(&dsfmt);
            if (x * x + y * y < 1.0) {
                t++;
            }
        }
        inside += t;
    } 
    pi = (double)inside / n_steps * 4;
    printf("%.10g\n", pi);
    return 0;
}
