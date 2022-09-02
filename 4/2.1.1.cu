#include <stdio.h>
#include <cuda_runtime.h>

// macros:
#define widthField 2
#define precisionField 0
struct Matrix;
__global__ void init_GPU (Matrix M);
struct Matrix
{
    int rows, cols;
    double *device_pointer, *host_pointer;
    int flag = 0;
    Matrix ()
    {
        rows = cols = 0;
        device_pointer = host_pointer = NULL;
        return;
    }
    Matrix (int rows, int cols)
    {
        // init (rows, cols);
        alloc (rows, cols);
        
    }
    void alloc (int r, int c)
    {
        rows = r;
        cols = c;
        cudaMalloc (&device_pointer, rows * cols * sizeof (double));
        host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
        return;
    }
    void display ()
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                printf (" %*.*lf ", widthField, precisionField, host_pointer[i * cols + j]);
            }
            printf ("\n");
        }
    }
    void init (int rows, int cols)
    {
        dim3 block (1, 1, 1);
        dim3 grid (rows, cols, 1);
        init_GPU <<<grid, block>>> (*this);
        cudaDeviceSynchronize ();
        d2h ();
        return;
    }
    void h2d ()
    {
        cudaMemcpy (device_pointer, host_pointer, cols * rows * sizeof (double), cudaMemcpyHostToDevice);
        return;
    }
    void d2h ()
    {
        cudaMemcpy (host_pointer, device_pointer, cols * rows * sizeof (double), cudaMemcpyDeviceToHost);
        return;
    }
};

// __global__ void MatrixMulKernel (float *MatA, float *MatB, float *MatC, int Width)
// {
//     int Row = blockIdx.y * blockDim.y + threadIdx.y;
//     int Col = blockIdx.x * blockDim.x + threadIdx.x;
//     if (Row < Width && Col < Width)
//     {
//         // printf ("{%d,%d}", Row, Col);
//         float Pvalue = 0;
//         for (int k = 0; k < Width; k++)
//         {
//             // printf ("(%.0f,%.0f)", MatA[Row * Width + k], MatB[k * Width + Col]);
//             Pvalue += MatA[Row * Width + k] * MatB[k * Width + Col];
//         }
//         MatC[Row * Width + Col] = Pvalue;
//         // printf ("=<%f>\n", Pvalue);
//     }
// }
// void display_Matrix (Matrix M)
// {
//     int idx;
//     for (int i = 0; i < M.rows; i++)
//     {
//         for (int j = 0; j < M.cols; j++)
//         {
//             idx = i * M.cols + j;
//             printf (" %*.*lf ", widthField, precisionField, M.host_pointer[idx]);
//         }
//         printf ("\n");
//     }
// }
// __device__ int random_int ()
// {
//     static int i = 12345678;
//     i *= 0xf9f9f9f9, i++;
//     return i;
// }
__global__ void init_GPU (Matrix M)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x; // x = rows
    int c = threadIdx.y + blockIdx.y * blockDim.y; // y = cols
    if (r < M.rows && c < M.cols)
    {
        // printf ("")
        M.device_pointer[r * M.cols + c] = (double) (r * M.cols + c) * r;
        // printf ("%lf ", M.device_pointer[r * M.cols + M.rows]);
    }
    return;
}

// void initialize_Matrix (Matrix M)
// {
//     dim3 block (1, 1, 1);
//     dim3 grid (M.rows, M.cols, 1);
//     initialize_Matrix_GPU <<<grid, block>>> (M);
//     cudaDeviceSynchronize ();
//     return;
// }

// void allocate_Matrix (Matrix *m, int rows, int cols)
// {
//     m->rows = rows;
//     m->cols = cols;
//     cudaMalloc (&(m->device_pointer), rows * cols * sizeof (double));
//     m->host_pointer = (double *) malloc (rows * cols * sizeof (double));
//     return;
// }
int main ()
{
    // int Width = N;
    // int nx = Width;
    // int ny = Width;
    // int nxy = nx * ny;
    Matrix A, B, C;
    // allocate_Matrix (&A, 4, 8);
    // transfer_Matrix_h2d (A);
    // initialize_Matrix (A);
    // transfer_Matrix_d2h (A);
    // display_Matrix (A);

    
    // allocate_Matrix (&B, 8, 2);
    // allocate_Matrix (&C, 2, 4);
    
    // initialize_Matrix <<<

    // int nBytes = nxy * sizeof (float);
    // printf ("Matrix size: nx %d ny %d\n", nx, ny);
    
    // float *h_A, *h_B, *h_C;
    // h_A = (float *) (malloc (nBytes));
    // h_B = (float *) malloc (nBytes);
    // h_C = (float *) malloc (nBytes);
    
    // initialData (h_A, nxy);
    // initialData (h_B, nxy);
    
    // float *d_MatA, *d_MatB, *d_MatC;
    // cudaMalloc ((void **) &d_MatA, nBytes);
    // cudaMalloc ((void **) &d_MatB, nBytes);
    // cudaMalloc ((void **) &d_MatC, nBytes);


    // cudaMemcpy ((void *) d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    // cudaMemcpy ((void *) d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // int bdimx = 16;
    // int bdimy = 16;

    // dim3 block (bdimx, bdimy, 1);
    // dim3 grid ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);

    // MatrixMulKernel <<<grid, block>>> (d_MatA, d_MatB, d_MatC, Width);
    // cudaDeviceSynchronize ();

    // cudaMemcpy (h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // printf ("Matrix A is=\n");
    // displayMatrix (h_A, nx, ny, 2);
    // printf ("Matrix B is=\n");
    // displayMatrix (h_B, nx, ny, 2);
    // printf ("The Product of Matrix A and Matrix B is=\n");
    // displayMatrix (h_C, nx, ny, 5);

    // cudaFree (d_MatA);
    // cudaFree (d_MatB);
    // cudaFree (d_MatC);

    // free (h_A);
    // free (h_B);
    // free (h_C);

    // cudaDeviceReset ();

    return 0;

}