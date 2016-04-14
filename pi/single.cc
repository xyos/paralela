
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define L 10000000 // Iterations



double piWork(long end) {
    double sum = 0;
    for (int i = 1; i < end ; i++){
        int j = 2*i-1 ;                   //denominator term
        double o = 1.0/ j ; 
        o = (i%2 == 1)? o : -1*o ;        //Odd terms are subtracted, even terms are added
        sum += o ;
    }  
    return sum;
}

int main ()
{

    struct timeval tv1, tv2;

    long end = (long)L;
    gettimeofday(&tv1, NULL);
    double pi = piWork(L);
    gettimeofday(&tv2, NULL);

    /* Results and cleanup */
    printf ("Sum = %.17g \n", pi*4);
    if (tv1.tv_usec > tv2.tv_usec) {
        tv2.tv_sec--;
        tv2.tv_usec += 1000000;
    }
    printf("Time - %ld.%ld\n", tv2.tv_sec - tv1.tv_sec, tv2.tv_usec - tv1.tv_usec);
}
