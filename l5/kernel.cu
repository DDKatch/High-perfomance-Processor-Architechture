#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <stdlib.h>
#include "utils.h"
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

#define ROW_SIZE 1024 // MATRIX = ROW_SIZE * ROW_SIZE
#define ROW_THREADS 32 // block threads = ROW_THREADS * ROW_THREADS

using namespace std;

void showMatrix(int* matrix){
	for (int i = 0; i < ROW_SIZE; i++){
		for (int j = 0; j < ROW_SIZE; j++){
			printf("%4.0d ", (matrix + i * ROW_SIZE)[j]);
		}
		printf("\n");
	}
}

int* initMatrix(){
	int* matrix = (int*)malloc(ROW_SIZE * ROW_SIZE * sizeof(int));
	for (int i = 0; i < ROW_SIZE * ROW_SIZE; i++)
		matrix[i] = 2;

	return matrix;
}

__global__ void matrixMulKernel(int* matr1, int* matr2, int* matr3, int* matr4){

	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int x = blockDim.x*blockIdx.x + threadIdx.x;

	if (x < ROW_SIZE && y < ROW_SIZE){
		int offset = y*ROW_SIZE + x;
		int roffset = x*ROW_SIZE + y;
		matr1[offset] = matr1[offset] * matr2[roffset] + matr3[offset] * matr4[roffset];
	}
}

int* matrixMulAndSumCuda(int* matrix){

	int* d_matr1 = nullptr;
	int* d_matr2 = nullptr;
	int* d_matr3 = nullptr;
	int* d_matr4 = nullptr;

	dim3 blockSize;
	dim3 gridSize;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	matrix = initMatrix();
	int tmatr1[ROW_SIZE * ROW_SIZE];
	int tmatr2[ROW_SIZE * ROW_SIZE];
	int tmatr3[ROW_SIZE * ROW_SIZE];
	int tmatr4[ROW_SIZE * ROW_SIZE];

	for (int i = 0; i < ROW_SIZE; i++)
		for (int j = 0; j < ROW_SIZE; j++){
			tmatr1[i * ROW_SIZE + j] = matrix[i * ROW_SIZE + j];
			tmatr2[i * ROW_SIZE + j] = matrix[i * ROW_SIZE + j];
			tmatr3[i * ROW_SIZE + j] = matrix[i * ROW_SIZE + j];
			tmatr4[i * ROW_SIZE + j] = matrix[i * ROW_SIZE + j];
		}

	checkCudaErrors(cudaSetDevice(0));
	//size_t pitch = ROW_SIZE*sizeof(int);          // size of row in bytes
	//checkCudaErrors(cudaMallocPitch((void **)&dev_matrix, &pitch, (size_t)ROW_SIZE*sizeof(int), (size_t)ROW_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_matr1, ROW_SIZE * ROW_SIZE * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_matr1, tmatr1, ROW_SIZE * ROW_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_matr2, ROW_SIZE * ROW_SIZE * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_matr2, tmatr2, ROW_SIZE * ROW_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_matr3, ROW_SIZE * ROW_SIZE * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_matr3, tmatr3, ROW_SIZE * ROW_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_matr4, ROW_SIZE * ROW_SIZE * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_matr4, tmatr4, ROW_SIZE * ROW_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	
	blockSize = dim3(ROW_THREADS, ROW_THREADS, 1);
	gridSize = dim3(ROW_SIZE * ROW_SIZE / blockSize.x, ROW_SIZE * ROW_SIZE / blockSize.y, 1);

	cudaEventRecord(start);
	matrixMulKernel <<< gridSize, blockSize >>> (d_matr1, d_matr2, d_matr3, d_matr4);
	cudaEventRecord(stop);

	checkCudaErrors(cudaMemcpy(matrix, d_matr1, ROW_SIZE * ROW_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "CUDA time simple (ms): " << milliseconds << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_matr1);
	cudaFree(d_matr2);
	cudaFree(d_matr3);
	cudaFree(d_matr4);
	return matrix;
}

int* cpuMulAndSumMatr(int* matrix){
	
	int tmatr1[ROW_SIZE][ROW_SIZE];
	int tmatr2[ROW_SIZE][ROW_SIZE];
	int tmatr3[ROW_SIZE][ROW_SIZE];
	int tmatr4[ROW_SIZE][ROW_SIZE];

	for (int y = 0; y < ROW_SIZE; y++)
		for (int x = 0; x < ROW_SIZE; x++){
			tmatr1[y][x] = matrix[y*ROW_SIZE + x];
			tmatr2[y][x] = matrix[y*ROW_SIZE + x];
			tmatr3[y][x] = matrix[y*ROW_SIZE + x];
			tmatr4[y][x] = matrix[y*ROW_SIZE + x];
		}

	for (int y = 0; y < ROW_SIZE; y++)
		for (int x = 0; x < ROW_SIZE; x++){
			tmatr1[y][x] *= tmatr2[x][y];
			tmatr3[y][x] *= tmatr4[x][y];
			tmatr1[y][x] += tmatr3[y][x];
		}

	for (int y = 0; y < ROW_SIZE; y++)
		for (int x = 0; x < ROW_SIZE; x++)
			matrix[y*ROW_SIZE + x] = tmatr1[y][x];

	return matrix;
}

__host__ int main()
{
	int* matrix = initMatrix();
	int* cpumatrix = initMatrix();

	matrix = matrixMulAndSumCuda(matrix);
	checkCudaErrors(cudaDeviceReset());

	//showMatrix(matrix);

	/*__int64 start;
	start = __rdtsc();

	cpumatrix = cpuMultMatr(cpumatrix);

	cout << "CPU time: " << __rdtsc() - start << '\n';*/

	auto start = Clock::now();

	cpumatrix = cpuMulAndSumMatr(cpumatrix);
	auto end = Clock::now();

	cout << "CPU time (ms): " << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

	//showMatrix(cpumatrix);

	bool equal = true;
	for (int i = 0; i < ROW_SIZE; i++)
		for (int j = 0; j < ROW_SIZE; j++)
			if (matrix[i * ROW_SIZE + j] != cpumatrix[i * ROW_SIZE + j])
				equal = false;

	if (equal)
		cout << "equal\n";
	else
		cout << "not equal\n";

	system("pause");

    return 0;
}


//
//#include <iostream>
//#include <algorithm>
//using namespace std;
//#define N 1024
//#define RADIUS 3
//#define BLOCK_ROW_SIZE 16
//__global__ void stencil_1d(int *in, int *out) {
//	__shared__ int temp[BLOCK_ROW_SIZE + 2 * RADIUS];
//	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
//	int lindex = threadIdx.x + RADIUS;
//	// Read input elements into shared memory
//	temp[lindex] = in[gindex];
//	if (threadIdx.x < RADIUS) {
//		temp[lindex - RADIUS] = in[gindex - RADIUS];
//		temp[lindex + BLOCK_ROW_SIZE] = in[gindex + BLOCK_ROW_SIZE];
//	}
//	// Synchronize (ensure all the data is available)
//	__syncthreads();
//	// Apply the stencil
//	int result = 0;
//	for (int offset = -RADIUS; offset <= RADIUS; offset++)
//		result += temp[lindex + offset];
//	// Store the result
//	out[gindex] = result;
//}
//void fill_ints(int *x, int n) {
//	fill_n(x, n, 1);
//}
//int main(void) {
//	int *in, *out; // host copies of a, b, c
//	int *d_in, *d_out; // device copies of a, b, c
//	int size = (N + 2 * RADIUS) * sizeof(int);
//	// Alloc space for host copies and setup values
//	in = (int *)malloc(size); fill_ints(in, N + 2 * RADIUS);
//	out = (int *)malloc(size); fill_ints(out, N + 2 * RADIUS);
//	// Alloc space for device copies
//	checkCudaErrors(cudaMalloc((void **)&d_in, size));
//	checkCudaErrors(cudaMalloc((void **)&d_out, size));
//	// Copy to device
//	checkCudaErrors(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice));
//	// Launch stencil_1d() kernel on GPU
//	stencil_1d << <N / BLOCK_ROW_SIZE, BLOCK_ROW_SIZE >> >(d_in + RADIUS, d_out + RADIUS);
//	// Copy result back to host
//	checkCudaErrors(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));
//	// Cleanup
//	free(in); free(out);
//	cudaFree(d_in); cudaFree(d_out);
//	return 0;
//}
