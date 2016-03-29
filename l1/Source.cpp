#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <Windows.h>

#define SIZE 1000

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

void main()
{
	float **firstMatrix, **secondMatrix, **resultMatrix;

	firstMatrix = make_matrix(100);
	secondMatrix = make_matrix(100);
	resultMatrix = make_matrix(0);

	srand(time(NULL));

	clock_t start = clock();
	for (int i = 0; i < SIZE; i++)
	{
		float *temp = resultMatrix[i];
		for (int j = 0; j < SIZE; j++)
		{
			float temp1 = firstMatrix[i][j];
			float *temp2 = secondMatrix[j];

#pragma loop(no_vector) 
			for (int k = 0; k < SIZE; k++)
			{
				temp[k] += temp1 * temp2[k];
			}
		}
	}
	clock_t end = clock();

	printf("time %f\n", (float)(end - start) / CLK_TCK);

	resultMatrix = make_matrix(0);

	srand(time(NULL));

	start = clock();
	for (int i = 0; i < SIZE; i++)
	{
		float *temp = resultMatrix[i];
		for (int j = 0; j < SIZE; j++)
		{
			float temp1 = firstMatrix[i][j];
			float *temp2 = secondMatrix[j];

			for (int k = 0; k < SIZE; k++)
			{
				temp[k] += temp1 * temp2[k];
			}
		}
	}
	end = clock();

	printf("vectorization time %f\n", (float)(end - start) / CLK_TCK);
	system("pause");
}