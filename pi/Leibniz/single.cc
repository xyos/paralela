
#include <stdio.h>
#include <stdlib.h>

static long n_steps = 5000000000;

double piWork(long end) {
    double sum = 0,
           x;
    double j;
    for (long i = 1; i < end ; i++){
        j = 2*i-1 ;  //denominator term
        x = 1.0/ j ; 
        sum += (i%2 == 1)? x : -x;
    }  
    return sum;
}

int main ()
{
    double pi = piWork(n_steps);
    printf ("Sum = %.10g \n", pi*4);
}
