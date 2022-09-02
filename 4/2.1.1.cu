#include <stdio.h>
#include <cuda_runtime.h>
// #define N 8
typedef struct
{
    int rows, cols;
    double *device_pointer, *host_pointer;
} Matrix;

__global__ void MatrixMulKernel (float *MatA, float *MatB, float *MatC, int Width)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (Row < Width && Col < Width)
    {
        // printf ("{%d,%d}", Row, Col);
        float Pvalue = 0;
        for (int k = 0; k < Width; k++)
        {
            // printf ("(%.0f,%.0f)", MatA[Row * Width + k], MatB[k * Width + Col]);
            Pvalue += MatA[Row * Width + k] * MatB[k * Width + Col];
        }
        MatC[Row * Width + Col] = Pvalue;
        // printf ("=<%f>\n", Pvalue);
    }
}
void displayMatrix (float *A, int nx, int ny, int widthField)
{
    int idx;
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            idx = i * ny + j;
            printf (" %*.0f ", widthField, A[idx]);
        }
        printf ("\n");
    }
}
__global__ void initial_Matrix (float *ip, const int rows, const int cols)
{
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    if (r < rows && c < cols)
    {
        ip[r * cols + rows] = (float) (rand () % 100);
    }
    return;
}
void allocate_Matrix (Matrix *m, int rows, int cols)
{
    m->rows = rows;
    m->cols = cols;
    cudaMalloc (&(m->device_pointer), rows * cols * sizeof (double));
    m->host_pointer = (double *) malloc (rows * cols * sizeof (double));
    return;
}
void transfer_Matrix_h2d (Matrix *m)
{
    cudaMemcpy (m->device_pointer, m->host_pointer, m->cols * m->rows * sizeof (double), cudaMemcpyHostToDevice);
    return;
}
void transfer_Matrix_d2h (Matrix *m)
{
    cudaMemcpy (m->device_pointer, m->host_pointer, m->cols * m->rows * sizeof (double), cudaMemcpyHostToDevice);
    return;
}
int main ()
{
    // int Width = N;
    // int nx = Width;
    // int ny = Width;
    // int nxy = nx * ny;
    Matrix A, B, C;
    allocate_Matrix (&A, 4, 8);
    allocate_Matrix (&B, 8, 2);
    allocate_Matrix (&C, 2, 4);

    // int nBytes = nxy * sizeof (float);
    // printf ("Matrix size: nx %d ny %d\n", nx, ny);
    
    float *h_A, *h_B, *h_C;
    h_A = (float *) (malloc (nBytes));
    h_B = (float *) malloc (nBytes);
    h_C = (float *) malloc (nBytes);
    
    initialData (h_A, nxy);
    initialData (h_B, nxy);
    
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc ((void **) &d_MatA, nBytes);
    cudaMalloc ((void **) &d_MatB, nBytes);
    cudaMalloc ((void **) &d_MatC, nBytes);


    cudaMemcpy ((void *) d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy ((void *) d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    int bdimx = 16;
    int bdimy = 16;

    dim3 block (bdimx, bdimy, 1);
    dim3 grid ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);

    MatrixMulKernel <<<grid, block>>> (d_MatA, d_MatB, d_MatC, Width);
    cudaDeviceSynchronize ();

    cudaMemcpy (h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    printf ("Matrix A is=\n");
    displayMatrix (h_A, nx, ny, 2);
    printf ("Matrix B is=\n");
    displayMatrix (h_B, nx, ny, 2);
    printf ("The Product of Matrix A and Matrix B is=\n");
    displayMatrix (h_C, nx, ny, 5);

    cudaFree (d_MatA);
    cudaFree (d_MatB);
    cudaFree (d_MatC);

    free (h_A);
    free (h_B);
    free (h_C);

    cudaDeviceReset ();

    return 0;

}