#include "Matrix.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>
#define BLOCK_SIZE 32
// Get a matrix element
__device__ float getElement(const float* data, const int n, int row, int col) {
    return data[row * n + col];
}
// Set a matrix element
__device__ void setElement(float* data, int n, int row, int col, float value) {
    data[row * n + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ float* GetSubMatrix(float* data, int n, int row, int col) {
    float *sub_data;
    sub_data = &data[n * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return sub_data;
}
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(float* a_data, float* b_data, float* c_data, int n) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    float* c_sub = GetSubMatrix(c_data, n, blockRow, blockCol);
    // Each thread computes one element of c_sub
    // by accumulating results into c_value
    float c_value = 0.0;
    // Thread row and column within c_sub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of a_ and b_ that are
    // required to compute c_sub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (n / BLOCK_SIZE); ++m) {
        // Get sub-matrix a_sub of A
        float* a_sub = GetSubMatrix(a_data, n, blockRow, m);
        // Get sub-matrix b_sub of B
        float* b_sub = GetSubMatrix(b_data, n, m, blockCol);
        // Shared memory used to store a_sub and b_sub respectively
        __shared__ float a_s[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float b_s[BLOCK_SIZE][BLOCK_SIZE];
        // Load a_sub and b_sub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        a_s[row][col] = getElement(a_sub, n, row, col);
        b_s[row][col] = getElement(b_sub, n, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply a_sub and b_sub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            c_value += a_s[row][e] * b_s[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of a_ and b_ in the next iteration
        __syncthreads();
    }
    // Write c_sub to device memory
    // Each thread writes one element
    setElement(c_sub, n, row, col, c_value);
}
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Mnn A, const Mnn B, Mnn C) {
    // Load _n and _nn to device memory
    Clock::time_point t1 = Clock::now();
    int* d_n;
    int* d_nn;
    size_t int_size = sizeof(int);
    cudaError_t err = cudaMalloc(&d_n, int_size);
    //printf("CUDA malloc d_n: %s\n",cudaGetErrorString(err));
    int n = A.get_n();
    cudaMemcpy(d_n, &n, int_size, cudaMemcpyHostToDevice);
    err = cudaMalloc(&d_nn, int_size);
    //printf("CUDA malloc d_nn: %s\n",cudaGetErrorString(err));
    int nn = A.get_data_size();
    cudaMemcpy(d_n, &nn, int_size, cudaMemcpyHostToDevice);
    // Load A data to device memory
    size_t size = A.get_data_size() * sizeof(float);
    float* a_data;
    err = cudaMalloc(&a_data, size);
    //printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
    cudaMemcpy(a_data, A._data, size, cudaMemcpyHostToDevice);
    // Load B data to device memory
    float* b_data;
    err = cudaMalloc(&b_data, size);
    //printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
    cudaMemcpy(b_data, B._data, size, cudaMemcpyHostToDevice);
    // Allocate C in device memory
    float* c_data;
    err = cudaMalloc(&c_data, size);
    //printf("CUDA malloc C: %s\n",cudaGetErrorString(err));
    cudaMemcpy(c_data, C._data, size, cudaMemcpyHostToDevice);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n / dimBlock.x, n / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(a_data, b_data, c_data, n);
    err = cudaThreadSynchronize();
    //printf("Run kernel: %s\n", cudaGetErrorString(err));
    // Read C from device memory
    err = cudaMemcpy(C._data, c_data, size, cudaMemcpyDeviceToHost);
    //printf("Copy C off of device: %s\n",cudaGetErrorString(err));
    // Free device memory
    cudaFree(d_n);
    cudaFree(d_nn);
    cudaFree(a_data);
    cudaFree(b_data);
    cudaFree(c_data);
    Clock::time_point t2 = Clock::now();
    duration<float> time = duration_cast<duration<float>>(t2-t1);
    printf("---- CUDA time: %f seconds \n", time.count());
}

int main(int argc, const char *argv[])
{
    int devID = 0;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&deviceProp, devID);
    int block_size = (deviceProp.major < 2) ? 16 : 32;
    printf("block size must be: %d\n", block_size);
    if(argc != 2){
        printf("usage: `./mmul n` \n");
        exit(0);
    }
    unsigned int m_size  = atoi(argv[1]);
    if(m_size%BLOCK_SIZE != 0){
        printf("n must be multiple of %d\n", BLOCK_SIZE);
        exit(0);
    }
    printf("running for n:%d \n\n", m_size);
    Mnn a(m_size), b(m_size), f(m_size);
    a.randomize();
    b.randomize();
    Mnn c = a * b;
    vector<int> threads = {2,4,8,16,32,64,128,256,512,1024};
    for(const auto& n_threads: threads){
        Mnn d = a.threadMult(b,n_threads);
        printf("TEST: %s\n", (c == d) ? "PASS": "FAIL");
    }
    vector<int> forks = {2,4,8,16,32,64,128};
    for(const auto& n_forks: forks){
        Mnn e = a.forkMult(b,n_forks);
        printf("TEST: %s\n", (c == e) ? "PASS": "FAIL");
    }
    MatMul(a, b, f);
    printf("TEST: %s\n", (c == f) ? "PASS": "FAIL");
    return 0;
}
