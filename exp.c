#include <stdio.h>
#include <sys/time.h>
#include <time.h>
int main ()
{
    struct timeval i, f;
    gettimeofday (&i, NULL);
    for (int i = 0; i < 2e9; i++);
    gettimeofday (&f, NULL);
    printf ("%lf\n", ((double) (f.tv_usec - i.tv_usec)) / 1e6);
    // gettimeofday (&v, NULL);
    // printf ("%lf\n", ((double) (v.tv_usec)) / 1e6);
    return 0;
}