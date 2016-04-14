#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

typedef struct
{
        int *a;
        int length;
        double sum;
} MyData;

#define N 8 // threads
#define L 10000000 // data for each thread

MyData mData;
pthread_t myThread[N];
pthread_mutex_t mutex;


void *piWork(void *arg) {
        long offset = (long)arg;
        double sum = 0;
        int start = offset * mData.length;
        int end = start + mData.length;

        for (int i = start; i < end ; i++) {
            int j = 2*mData.a[i]-1 ;            //denominator term
            double o = 1.0/ j ; 
            o = (mData.a[i]%2 == 1)? o : -1*o ; //Odd terms are subtracted, even terms are added
            sum += o ;
        }  

        /* mutex lock/unlock */
        pthread_mutex_lock(&mutex);
        mData.sum += sum;
        pthread_mutex_unlock(&mutex);

        pthread_exit((void*) 0);
}

int main () {
        void *status;

        /* fill the structure */
        int *a = (int*) malloc (N*L*sizeof(int));
        for (int i = 0; i < N*L; i++) a[i] = i + 1;
        mData.length = L;
        mData.a = a;
        mData.sum = 0;

        struct timeval tv1, tv2;

        pthread_mutex_init(&mutex, NULL);

        /* Each thread has its own  set of data to work on. */
        gettimeofday(&tv1, NULL);
        for(long i=0; i < N; i++)
                pthread_create(&myThread[i], NULL, piWork, (void *)i);

        /* Wait on child threads */
        for(int i=0; i < N; i++) pthread_join(myThread[i], &status);
        gettimeofday(&tv2, NULL);

        /* Results and cleanup */
        printf ("Sum = %.17g \n", mData.sum*4);
        if (tv1.tv_usec > tv2.tv_usec) {
            tv2.tv_sec--;
            tv2.tv_usec += 1000000;
        }
        printf("Time - %ld.%ld\n", tv2.tv_sec - tv1.tv_sec, tv2.tv_usec - tv1.tv_usec);
        free (a);
        pthread_mutex_destroy(&mutex);
        pthread_exit(NULL);
}
