#include <stdlib.h>
#include <stdio.h>
#include <windows.h>

#define OFFSET  262144 // CACHE SIZE 1 lvl (in cells) 2*32KB = 8192
#define N  64
#define TYPE __int64

int BLOCK_SIZE = OFFSET / N;

void init(int n, TYPE* arr) {

	for (int j = 0; j < n - 1; j++) {
		for (int i = 0; i < BLOCK_SIZE; i++) {
			arr[i + j * OFFSET] = (TYPE)&arr[i + (j + 1) * OFFSET];
		}
	}
	for (int i = 0; i < BLOCK_SIZE - 1; i++) {
		arr[i + (n - 1) * OFFSET] = (TYPE)&arr[i + 1];
	}
	arr[BLOCK_SIZE - 1 + (n - 1)*OFFSET] = (TYPE)&arr[0]; // arr[SIZE - 1] = (int)&arr[0];
}

int main()
{
	unsigned int start, end;
	TYPE arr[OFFSET*N];
	
	//SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	
	for (int i = 1; i < N; i++) {
		init(i, arr);
		__asm {
			lea ebx, arr
			
			rdtsc
			mov start, eax
			
			mov ecx, BLOCK_SIZE
			main_loop:
				mov ebx, [ebx]
				dec ecx
				cmp ecx, 0
			jne main_loop
			
			rdtsc
			mov end, eax
		}
		printf("time %2d = %4.4lf \n", i, (double)(end - start) / BLOCK_SIZE);
	}

	system("pause");

	return 0;
}