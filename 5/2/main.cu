#include <stdio.h>
#include "matrix.cuh"
int main ()
{
    srand (time (NULL));
    Matrix A (4, 3);
    A.init ();
    Matrix AT = ~A;
    printf ("Matrix A:\n");
    A.display ();
    printf ("Matrix AT:\n");
    AT.display ();
    Matrix P = A * AT;
    printf ("Matrix P:\n");
    P.display ();
    // Matrix D = C * C;
    // printf("Matrix D:\n");
    // D.display();
    // C + 
    // Matrix TA = ~A, TB = ~B;
    // printf ("\033[4;31mMatrix A:\033[m\n");
    // A.display ();
    // printf ("\033[4;31mMatrix B:\033[m\n");
    // B.display ();
    // printf ("\033[4;31mMatrix TA:\033[m\n");
    // TA.display ();
    // printf ("\033[4;31mMatrix TB:\033[m\n");
    // TB.display ();
    // Matrix PAB = A * B;
    // Matrix PTATB = TA * TB;
    // printf ("\033[4;31mMatrix PAB:\033[m\n");
    // PAB.display ();
    // printf ("\033[4;31mMatrix PTATB:\033[m\n");
    // PTATB.display ();
    // Matrix D = PAB - PTATB;
    // printf ("\033[4;31mMatrix D:\033[m\n");
    // D.display ();
    
    cudaDeviceReset ();
    return 0;
}