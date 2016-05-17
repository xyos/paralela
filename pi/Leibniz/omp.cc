#include <omp.h>
#include <stdio.h>

static long n_steps = 5000000000;
#ifdef NUM_THREADS
#else
    #define NUM_THREADS 16
#endif 
int main(int argc, char** argv){
    long i;
    double pi, sum[NUM_THREADS][8];
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel
    {
        int id;
        double e;
        id = omp_get_thread_num();
        e = (id%2 == 1) ? 1.0 : -1.0;
        #pragma for schedule(static)
        for (long i = id; i < n_steps; i+=NUM_THREADS) {
            sum[id][0] +=  1.0/(2*i-1)*e;
        }
    }
    for(i=0, pi=0.0; i< NUM_THREADS; i++){
        pi += sum[i][0];
    }
    printf ("%.10g \n", pi*4-4);
    return 0;
}
