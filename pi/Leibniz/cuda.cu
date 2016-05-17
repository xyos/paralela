// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define BLOCKS 512
#define THREADS 512
#define PI 3.141592654  // known value of pi

__global__ void gpu_monte_carlo(float *estimate) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	float j,x;
        j = 2*tid-1 ;  //denominator term
        x = 1.0/ j ; 
	estimate[tid] = 4.0*(tid%2 == 1)? x : -x; // return estimate of pi
}

int main (int argc, char *argv[]) {
	clock_t start, stop;
	float host[BLOCKS * THREADS];
	float *dev;

	printf("# of trials = %d, # of blocks = %d, # of threads/block = %d.\n", BLOCKS * THREADS,
BLOCKS, THREADS);

	start = clock();

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts
	

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results 

	float pi_gpu;
	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	stop = clock();

	printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);
	printf("CUDA estimate of PI = %.10g [error of %.10g]\n", pi_gpu*4.0, pi_gpu - PI);
	
	return 0;
}
