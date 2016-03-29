#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <Windows.h>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define SIZE 2000

float** make_matrix(int random_range)
{
	float** matrix = (float**)_aligned_malloc(SIZE * sizeof(float*), 16);
	for (int i = 0; i < SIZE; i++)
		matrix[i] = (float*)_aligned_malloc(SIZE * sizeof(float), 16);

	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++)
			random_range == 0 ? matrix[i][j] = 0 : matrix[i][j] = rand() % random_range;

	return matrix;
}

void showMatrix(float** matrix)
{
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
			printf("%3.2f  ", matrix[i][j]);
		printf("\n");
	}
	printf("\n");
}

void main()
{
	float **firstMatrix, **secondMatrix, **resultMatrixv, **resultMatrixvi, **resultMatrixov, **resultMatrixovi;

	firstMatrix = make_matrix(10);
	secondMatrix = make_matrix(10);
	resultMatrixv = make_matrix(0);
	resultMatrixvi = make_matrix(0);
	resultMatrixov = make_matrix(0);
	resultMatrixovi = make_matrix(0);

	srand(time(NULL));
			
			//simple vectorized

	clock_t start = clock();
	for (int i = 0; i < SIZE; i++)
	{
		float *temp = resultMatrixv[i];
		for (int j = 0; j < SIZE; j++)
		{
			float temp1 = firstMatrix[i][j];
			float *temp2 = secondMatrix[j];

			for (int k = 0; k < SIZE; k++)
				temp[k] += temp1 * temp2[k];
		}
	}
	clock_t end = clock();
	printf("vectorize %f\n", (float)(end - start) / CLK_TCK);
	//showMatrix(resultMatrixv);

			//intrinsics

	start = clock();
	for (int i = 0; i < SIZE; i++)
	{
		float *temp = resultMatrixvi[i];

		for (int j = 0; j < SIZE; j++)
		{
			float temp1 = firstMatrix[i][j];
			float *temp2 = secondMatrix[j];

			/*
			float* t1 = (float*)_aligned_malloc(SIZE * sizeof(float), 16);

			for (int k = 0; k < SIZE; k++)
			{
			t1[k] = temp1;
			//temp[k] += temp1 * temp2[k]; //asm
			}

			__asm{
			mov eax, temp
			mov ebx, t1
			mov edx, temp2
			movsd xmm0,[eax]
			movsd xmm1,[ebx]
			movsd xmm2,[edx]
			mulps xmm1, xmm2
			addss xmm0, xmm1
			movsd [eax], xmm0
			}						чет работает только с одним столбцом

			*/
			//asm
			

			__m128 t0, t1, t2;
			t1 = _mm_load1_ps(&temp1);
			for (int t = 4; t < SIZE ; t+=8)
			{
				t0 = _mm_load_ps(temp + t);
				t2 = _mm_load_ps(temp2 + t);
				t2 = _mm_mul_ps(t2, t1);
				t0 = _mm_add_ps(t0, t2);
				_mm_store_ps(temp + t, t0);	

				t0 = _mm_load_ps(temp + t - 4 );
				t2 = _mm_load_ps(temp2 + t - 4);
				t2 = _mm_mul_ps(t2, t1);
				t0 = _mm_add_ps(t0, t2);
				_mm_store_ps(temp + t - 4, t0);
			}			
		}
	}
	end = clock();
	printf("vectorize intrinsics %f\n", (float)(end - start) / CLK_TCK);
	//showMatrix(resultMatrixvi);

			//omp vectorized

	start = clock();
	#pragma omp parallel for num_threads(2)
	for (int i = 0; i < SIZE; i++)
	{
		float *temp = resultMatrixov[i];
		float *fm = firstMatrix[i];
		for (int j = 0; j < SIZE; j++)
		{
			float temp1 = fm[j];
			float *temp2 = secondMatrix[j];
			
			for (int k = 0; k < SIZE; k++)
				temp[k] += temp1 * temp2[k];
		}
	}
	end = clock();

	printf("omp %f\n", (float)(end - start) / CLK_TCK);
	//showMatrix(resultMatrixov);
	
	__m128 t0, t1, t2;
	start = clock();
	#pragma omp parallel for num_threads(2)
	for (int i = 0; i < SIZE; i++)
	{
		float *temp = resultMatrixovi[i];
		float *fm = firstMatrix[i];
		for (int j = 0; j < SIZE; j++)
		{
			float temp1 = fm[j];
			float *temp2 = secondMatrix[j];

			/*
			float* t1 = (float*)_aligned_malloc(SIZE * sizeof(float), 16);

			for (int k = 0; k < SIZE; k++)
			{
			t1[k] = temp1;
			//temp[k] += temp1 * temp2[k]; //asm
			}

			__asm{
			mov eax, temp
			mov ebx, t1
			mov edx, temp2
			movsd xmm0,[eax]
			movsd xmm1,[ebx]
			movsd xmm2,[edx]
			mulps xmm1, xmm2
			addss xmm0, xmm1
			movsd [eax], xmm0
			}						чет работает только с одним столбцом

			*/
			//asm

			
			t1 = _mm_load1_ps(&temp1);
			for (int t = 4; t < SIZE; t += 8)
			{
				t0 = _mm_load_ps(temp + t);
				t2 = _mm_load_ps(temp2 + t);
				t2 = _mm_mul_ps(t2, t1);
				t0 = _mm_add_ps(t0, t2);
				_mm_store_ps(temp + t, t0);

				t0 = _mm_load_ps(temp + t - 4);
				t2 = _mm_load_ps(temp2 + t - 4);
				t2 = _mm_mul_ps(t2, t1);
				t0 = _mm_add_ps(t0, t2);
				_mm_store_ps(temp + t - 4, t0);
			}
		}
	}
	end = clock();

	printf("omp intrinsics %f\n", (float)(end - start) / CLK_TCK);
	//showMatrix(resultMatrixovi);


	bool equal = true;
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++)
			if (resultMatrixov[i][j] != resultMatrixovi[i][j] ||
				resultMatrixv[i][j] != resultMatrixvi[i][j] ||
				resultMatrixov[i][j] != resultMatrixv[i][j])
				equal = false;
	
	if (equal)
		printf("equal\n");
	else
		printf("not equal\n");

	system("pause");
}