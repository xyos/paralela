#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>

#define L 80000000 // Iterations

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
pthread_mutex_t *glob_mutex;

int main(void) {
    struct timeval tv1, tv2;
    glob_sum = (double *)mmap(NULL, sizeof *glob_sum, PROT_READ | PROT_WRITE, 
                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    *glob_sum = 0;

    pthread_mutexattr_t attrmutex;

    /* Initialise attribute to mutex. */
    pthread_mutexattr_init(&attrmutex);
    pthread_mutexattr_setpshared(&attrmutex, PTHREAD_PROCESS_SHARED);

    glob_mutex = (pthread_mutex_t *)mmap(NULL, sizeof *glob_mutex, PROT_READ | PROT_WRITE, 
                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    /* Initialise mutex. */
    pthread_mutex_init(glob_mutex, &attrmutex);

    gettimeofday(&tv1, NULL);
    if (fork() == 0) {
        double sum = piWork(1,floor(L/2));
        pthread_mutex_lock(glob_mutex);
        *glob_sum += sum;
        pthread_mutex_unlock(glob_mutex);
        exit(EXIT_SUCCESS);
    } else {
        double sum = piWork(ceil(L/2),L);
        pthread_mutex_lock(glob_mutex);
        *glob_sum += sum;
        pthread_mutex_unlock(glob_mutex);
        wait(NULL);
    }
    printf ("Sum = %.17g \n", *glob_sum*4);
    if (tv1.tv_usec > tv2.tv_usec) {
        tv2.tv_sec--;
        tv2.tv_usec += 1000000;
    }
    gettimeofday(&tv2, NULL);
    printf("Time - %ld.%ld\n", tv2.tv_sec - tv1.tv_sec, tv2.tv_usec - tv1.tv_usec);
    munmap(glob_sum, sizeof *glob_sum);
    pthread_mutex_destroy(glob_mutex);
    pthread_mutexattr_destroy(&attrmutex); 
    return 0;
}
