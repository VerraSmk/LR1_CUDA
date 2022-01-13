#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <time.h> 
#define BLOCK_SIZE  16          // submatrix size

__global__ void matMultCuda(float* a, float* b, int n, float* c)
{
    int bx = blockIdx.x;        // block index
    int by = blockIdx.y;

    int tx = threadIdx.x;       // thread index
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * n;
    float sum = 0.0f;           // computed subelement

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        // Shared memory for the sub-matrix of A
        __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
        // Shared memory for the sub-matrix of B
        __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from global memory to shared memory;
        as[ty][tx] = a[ia + n * ty + tx];
        bs[ty][tx] = b[ib + n * ty + tx];

        __syncthreads();    // Synchronize to make sure the matrices are loaded

                            // Multiply the two matrices together;
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += as[ty][k] * bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory;
    // each thread writes one element
    int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    c[ic + n * ty + tx] = sum;
}

double calcCuda(float* a, float* b, float* c, int N, bool flag)
{
    clock_t start2 = clock();
    int numBytes = N * N * sizeof(float);
    // allocate device memory
    float* adev = NULL;
    float* bdev = NULL;
    float* cdev = NULL;

    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    // set kernel launch configuration
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    // create cuda event handles
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // asynchronously issue work to the GPU (all to stream 0)
    cudaEventRecord(start, 0);
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    matMultCuda << < blocks, threads >> > (adev, bdev, N, cdev);

    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    // print the events gpu times
    if (flag)
        printf("Time spent executing by the GPU events: %.2f millseconds\n", gpuTime);

    // release resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    clock_t end2 = clock();
    double millseconds2 = (double)(end2 - start2);
    // print the gpu times
    if (flag)
        printf("Time spent executing by the GPU: %.2f millseconds\n", millseconds2);
    return millseconds2;
}

double calcCPU(float* a, float* b, float* c, int N)
{
    clock_t start3 = clock();
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            c[i * N + j] = 0;
            for (int k = 0; k < N; ++k)
                c[i * N + j] += a[i * N + k] * b[k * N + j];
        }
    }
    clock_t end3 = clock();
    double millseconds = (double)(end3 - start3);
    // print the cpu times
    printf("Time spent executing by the CPU: %.2f millseconds\n", millseconds);
    return millseconds;
}

int main(int argc, char* argv[])
{
    float* a = new float[64 * 64];
    float* b = new float[64 * 64];
    float* cpu = new float[64 * 64];
    float* gpu = new float[64 * 64];
    //Run to initialize cuda
    calcCuda(a, b, gpu, 64, false);
    //Main
    for (int i = 6; i < 12; i++)
    {
        int N = pow(2, i);       // matrix size is N*N
        printf("Experiment for matrix size: %u \n", N);
        // allocate host memory
        float* a = new float[N * N];
        float* b = new float[N * N];
        float* cpu = new float[N * N];
        float* gpu = new float[N * N];

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                a[i * N + j] = rand() % 100;
                b[i * N + j] = rand() % 100;
            }
        printf("Acceleration factor: %.2f \n", calcCPU(a, b, cpu, N) / calcCuda(a, b, gpu, N, true));
        bool rel = true;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                if (cpu[i * N + j] != gpu[i * N + j]) { rel = false; break; }
            }
        printf("Relevance: %s \n", rel ? "true" : "false");
        delete a;
        delete b;
        delete cpu;
        delete gpu;
    }
    return 0;
}