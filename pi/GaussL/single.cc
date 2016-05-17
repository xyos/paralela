#include <stdio.h>
#include <math.h>

static long iterations = 3;

int main()
{
    bool run=true;
    double a=1,
           aa,
           b=pow(2,-.5),
           t=.25,
           p=1,
           r;
    for(long i=0;i<iterations;i++) {
        aa=(a+b)/2;
        b=pow((a*b),.5);
        t=t-p*pow((a-aa),2);
        p=2*p;
        a=aa;
    }
    r=(pow((a+b),2))/(4*t);
    printf ("%.10g \n", r);
    return 0;
}
