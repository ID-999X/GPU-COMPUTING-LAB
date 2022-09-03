#include <stdio.h>
#include <cuda_runtime.h>

// macros:
#define widthField 2
#define precisionField 0
struct Matrix;
__global__ void init_GPU (double *p, int rows, int cols);
__global__ void mul_GPU (double *m1, double *m2, double *p, int rows, int x, int cols);
struct Matrix
{
    int rows, cols;
    double *device_pointer, *host_pointer;
    int flag = 0;
    Matrix () : rows (0), cols (0), device_pointer (NULL), host_pointer (NULL)
    {
        return;
    }
    Matrix (int r, int c) : Matrix ()
    {
        rows = r;
        cols = c;
        alloc ();
        return;
    }
    Matrix (const Matrix &M)
    {
        printf ("\033[90mMatrix (const Matrix &M)\033[m\n");
        rows = M.rows;
        cols = M.cols;
        cudaMalloc (&device_pointer, rows * cols * sizeof (double));
        cudaMemcpy (device_pointer, M.device_pointer, rows * cols * sizeof (double), cudaMemcpyDeviceToDevice);
        host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
        memcpy (host_pointer, M.host_pointer, rows * cols * sizeof (double));
        return;
    }
    Matrix (Matrix &&M)
    {
        printf ("\033[90mMatrix (const Matrix &&M)\033[m\n");
        rows = M.rows;
        cols = M.cols;
        device_pointer = M.device_pointer;
        host_pointer = M.host_pointer;
        M.rows = M.cols = 0;
        M.device_pointer = M.host_pointer = NULL;
        // M.clear ();
        return;
    }
    Matrix operator = (Matrix &M)
    {
        printf ("\033[90mMatrix operator = (Matrix &M)\033[m\n");
        clear ();
        rows = M.rows;
        cols = M.cols;
        cudaMalloc (&device_pointer, rows * cols * sizeof (double));
        cudaMemcpy (device_pointer, M.device_pointer, rows * cols * sizeof (double), cudaMemcpyDeviceToDevice);
        host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
        memcpy (host_pointer, M.host_pointer, rows * cols * sizeof (double));
        return *this;
    }
    Matrix operator = (Matrix &&M)
    {
        printf ("\033[90mMatrix operator = (Matrix &&M)\033[m\n");
        rows = M.rows;
        cols = M.cols;
        device_pointer = M.device_pointer;
        host_pointer = M.host_pointer;
        M.rows = M.cols = 0;
        M.device_pointer = M.host_pointer = NULL;
        return *this;
    }
    ~Matrix ()
    {
        printf ("\033[90m~Matrix () : %p, %p\033[m\n", device_pointer, host_pointer);
        if (NULL != device_pointer)
        {
            cudaFree (device_pointer);
        }
        if (NULL != host_pointer)
        {
            free (host_pointer);
        }
        rows = cols = 0;
        device_pointer = host_pointer = NULL;
        return;
    }
    void alloc ()
    {
        cudaMalloc (&device_pointer, rows * cols * sizeof (double));
        host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
        // printf ("hello");
        return;
    }
    void clear ()
    {
        // printf ("%p, %p\n", device_pointer, host_pointer);
        if (NULL != device_pointer)
        {
            cudaFree (device_pointer);
        }
        if (NULL != host_pointer)
        {
            free (host_pointer);
        }
        rows = cols = 0;
        device_pointer = host_pointer = NULL;
        return;
    }
    // display works on host matrix
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
        return;
    }
    void display (const Matrix &M)
    {
        if (cols == M.cols && rows == M.rows)
        {
            for (int i = 0; i < rows; i++)
            {
                // first matrix: 
                for (int j = 0; j < cols; j++)
                {
                    printf (" %*.*lf ", widthField, precisionField, host_pointer[i * cols + j]);
                }
                printf (" |  "); // seperator
                // second matrix: 
                for (int j = 0; j < cols; j++)
                {
                    printf (" %*.*lf ", widthField, precisionField, M.host_pointer[i * cols + j]);
                }
                printf ("\n");
            }
        }
        return;
    }
    void init ()
    {
        dim3 block (1, 1, 1);
        dim3 grid (rows, cols, 1);
        init_GPU <<<grid, block>>> (device_pointer, rows, cols);
        cudaDeviceSynchronize ();
        // printf ("\033[31mhere\033[m");
        device2host ();
        // printf ("here");
        return;
    }
    void host2device ()
    {
        cudaMemcpy (device_pointer, host_pointer, cols * rows * sizeof (double), cudaMemcpyHostToDevice);
        return;
    }
    void device2host ()
    {
        cudaMemcpy (host_pointer, device_pointer, cols * rows * sizeof (double), cudaMemcpyDeviceToHost);
        return;
    }
    Matrix operator * (const Matrix &M)
    {
        if (cols != M.rows)
        {
            printf ("Matrix1 (%dX%d); Matrix2 (%dX%d)\n", rows, cols, M.rows, M.cols);
            return Matrix ();
        }
        Matrix p (rows, M.cols);
        dim3 block (1, 1, 1);
        dim3 grid (rows, M.cols, 1);
        mul_GPU <<< block, grid>>> (device_pointer, M.device_pointer, p.device_pointer, rows, cols, M.cols);
        cudaDeviceSynchronize ();
        p.device2host ();
        p.display ();
        return p;
    }
};

__global__ void init_GPU (double *p, int rows, int cols)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x; // x = rows
    int c = threadIdx.y + blockIdx.y * blockDim.y; // y = cols
    // printf ("%d;%d;%d;%d\n", r, c, M.rows, M.cols);
    if (r < rows && c < cols)
    {
        // printf ("<%d>", r * M.cols + c);
        p[r * cols + c] = ((double) (r * cols + c));
        // printf ("%lf ", M.device_pointer[r * M.cols + c]);
    }
    return;
}
__global__ void mul_GPU (double *m1, double *m2, double *p, int rows, int x, int cols)
{
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = blockIdx.y * blockDim.y + threadIdx.y;
    if (Row < rows && Col < cols)
    {
        // printf ("{%d,%d}", Row, Col);
        double a = 0;
        for (int k = 0; k < x; k++)
        {
            // printf ("(%.0f,%.0f)", m1[Row * cols + k], m2[k * rows + Col]);
            a += m1[Row * x + k] * m2[k * cols + Col];
        }
        p[Row * cols + Col] = a;
        // printf ("=<%f>\n", a);
    }
    return;
}
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
    Matrix A (2, 8), B (8, 8);
    A.init ();
    A.display ();
    printf ("-----------------\n");
    B.init ();
    B.display ();
    printf ("-----------------\n");
    Matrix C = A * B;
    // C.display ();
    // C.display ();
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