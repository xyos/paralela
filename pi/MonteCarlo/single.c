#include <stdio.h>
#include <stdlib.h>
#include "dsfmt/dSFMT.h"

int main(int argc, char* argv[]) {
    int i, inside, seed;
    double x, y, pi;
    const long n_steps = 1000000000;
    dsfmt_t dsfmt;

    seed = 142857;
    inside = 0;
    dsfmt_init_gen_rand(&dsfmt, seed);
    for (i = 0; i < n_steps; i++) {
        x = dsfmt_genrand_close_open(&dsfmt);
        y = dsfmt_genrand_close_open(&dsfmt);
        if (x * x + y * y < 1.0) {
            inside++;
        }
    }
    pi = (double)inside / n_steps * 4;
    printf("%.10g\n", pi);
    return 0;
}
