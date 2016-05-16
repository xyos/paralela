#include "Matrix.hpp"
#include <vector>
#include <iostream>

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Mnn A, const Mnn B, Matrix C) {
    // Load _n and _nn to device memory
    int d_n;
    int d_nn;
    cudaError_t err = cudaMalloc(d_n, sizeof(int));
    printf("CUDA malloc d_n: %s\n",cudaGetErrorString(err));
    cudaMemcpy(d_n, A.get_n(), sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err = cudaMalloc(d_nn, sizeof(int));
    printf("CUDA malloc d_nn: %s\n",cudaGetErrorString(err));
    cudaMemcpy(d_n, A.get_data_size(), sizeof(int), cudaMemcpyHostToDevice);
    // Load A data to device memory
    size_t size = A.get_data_size() * sizeof(float);
    float* a_data;
    cudaError_t err = cudaMalloc(&a_data, size);
    printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
    cudaMemcpy(a_data, A._data, size, cudaMemcpyHostToDevice);
    // Load B data to device memory
    float* b_data;
    cudaError_t err = cudaMalloc(&b_data, size);
    printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
    cudaMemcpy(b_data, B._data, size, cudaMemcpyHostToDevice);
    // Allocate C in device memory
    float* c_data;
    cudaError_t err = cudaMalloc(&c_data, size);
    printf("CUDA malloc C: %s\n",cudaGetErrorString(err));
    cudaMemcpy(c_data, C._data, size, cudaMemcpyHostToDevice);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(a_data, b_data, c_data);
    err = cudaThreadSynchronize();
    printf("Run kernel: %s\n", cudaGetErrorString(err));
    // Read C from device memory
    err = cudaMemcpy(C._data, c_data, size, cudaMemcpyDeviceToHost);
    printf("Copy C off of device: %s\n",cudaGetErrorString(err));
    // Free device memory
    cudaFree(d_n);
    cudaFree(d_nn);
    cudaFree(a_data);
    cudaFree(b_data);
    cudaFree(c_data);
}
// Get a matrix element
__device__ float GetElement(const float* data, const int n, int row, int col) {
    return data[row * n + col];
}
// Set a matrix element
__device__ void SetElement(float* data, int n, int row, int col, float value) {
    data[row * n + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ float* GetSubMatrix(float* data, int n, int row, int col) {
    float* sub_data;
    sub_data = data[n * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return sub_data;
}
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(float* a_data, float* b_data, float* c_data, int n) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    float* c_sub = GetSubMatrix(c_data, blockRow, blockCol);
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
    for (int m = 0; m < (n / b_LOCK_SIZE); ++m) {
        // Get sub-matrix a_sub of A
        Matrix a_sub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix b_sub of B
        Matrix b_sub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store a_sub and b_sub respectively
        __shared__ float a_s[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float b_s[BLOCK_SIZE][BLOCK_SIZE];
        // Load a_sub and b_sub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        a_s[row][col] = GetElement(a_sub, row, col);
        b_s[row][col] = GetElement(b_sub, row, col);
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
    SetElement(c_sub, row, col, c_value);
}

int main(int argc, const char *argv[])
{
    unsigned int m_size  = atoi(argv[1]);
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
        e.print();
        printf("TEST: %s\n", (c == e) ? "PASS": "FAIL");
    }
    MatMul(a, b, f);
    return 0;
}
