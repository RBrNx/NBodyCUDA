//GPU code goes here
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void initValues(int *input, int *output, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < size) { output[idx] = input[idx] * 2; }
}

int main(){
	const int size = 1000000;
	srand(time(NULL));

	int blockSize;
	int minGridSize;
	int gridSize;

	int* h_Array = (int*) malloc(size * sizeof(int));
	int* h_testArray = (int*)malloc(size * sizeof(int));

	int* d_InputArray; cudaMalloc((void**)&d_InputArray, size * sizeof(int));
	int* d_OutputArray; cudaMalloc((void**)&d_OutputArray, size * sizeof(int));

	//Test
	for (int i = 0; i < size; i++){
		h_Array[i] = i;
		h_testArray[i] = h_Array[i] * 2;
	}

	cudaMemcpy(d_InputArray, h_Array, size * sizeof(int), cudaMemcpyHostToDevice);

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initValues, 0, size);

	gridSize = (size + blockSize - 1) / blockSize;

	initValues <<<gridSize, blockSize >>>(d_InputArray, d_OutputArray, size);

	cudaMemcpy(h_Array, d_OutputArray, size*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++){
		if (h_Array[i] != h_testArray[i]){
			printf("Error at %i ! Host = %i, Device = %i \n", i, h_testArray[i], h_Array[i]);
		}
	}

	int random = rand() % size;
	printf("Random Number: %i, Host Value at %i, Device Value at %i \n", random, h_testArray[random], h_Array[random]);

	printf("Test Passed \n");

}