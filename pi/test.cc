#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/time.h>

#define L 10000000 // Iterations

double piWork(long start,long end) {
        double sum = 0;
        for (int i = start; i < end ; i++){
            int j = 2*i-1 ;                   //denominator term
            double o = 1.0/ j ; 
            o = (i%2 == 1)? o : -1*o ;        //Odd terms are subtracted, even terms are added
            sum += o ;
        }  
        return sum;
}


static double *glob_sum;

int main(void) {
    struct timeval tv1, tv2;
    glob_sum = (double *)mmap(NULL, sizeof *glob_sum, PROT_READ | PROT_WRITE, 
                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    *glob_sum = 0;
    gettimeofday(&tv1, NULL);
    if (fork() == 0) {
        *glob_sum += piWork(1,floor(L/2));
        exit(EXIT_SUCCESS);
    } else {
        *glob_sum += piWork(ceil(L/2),L);
        wait(NULL);
        printf("%.17g\n", *glob_sum*4);
        munmap(glob_sum, sizeof *glob_sum);
    }
    if (tv1.tv_usec > tv2.tv_usec) {
        tv2.tv_sec--;
        tv2.tv_usec += 1000000;
    }
    gettimeofday(&tv2, NULL);
    printf("Time - %ld.%ld\n", tv2.tv_sec - tv1.tv_sec, tv2.tv_usec - tv1.tv_usec);
    return 0;
}
