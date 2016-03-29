/*

TASK 3
test function SUB
- latency
- throughput

*/

#include <stdio.h>
#include <stdlib.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
//#include "latthpt.h"
#include <intrin.h>

//#define COM addpd
//#define COM_NAME "ADDPD"


#define N 10000000
#define K 6

void test_latency() {
	__asm mov ebx, 100000000000
	//clock_t start = clock();
	__int64 start = __rdtsc();
	for (int i = 0; i < N; i++) {
		__asm {
			sub ebx, eax;
			sub ebx, eax;
			sub ebx, eax;
			sub ebx, eax;
			sub ebx, eax;
			sub ebx, eax;
		}
		// K ���
	}
	int a;
	__asm mov a, ebx
	__int64 end = __rdtsc();

	printf("latency %f\n", (double)(end - start) / (K*N));
}

void test_throughput() {
	__asm mov ebx, 100000000000
	//clock_t start = clock();
	__int64 start = __rdtsc();
	for (int i = 0; i < N; i++) {
		__asm {
			sub ebx, eax;
			sub ecx, eax;
			sub edx, eax;
			sub ebx, eax;
			sub ecx, eax;
			sub edx, eax;
		}
		// K ���
	}
	int a;
	__asm mov a, ebx
	__int64 end = __rdtsc();

	printf("throughput %f\n", (double)(end - start) / (K*N));
}

int main() {

	test_latency();
	test_throughput();

	system("pause");
	return 0;
}


// int main()
// {
// 	//
// 	// Initialize latency/throughput macros.
// 	//
//
// 	LatThpt_Init();
//
// 	//
// 	// Perform xmm integer latency tests.
// 	//
//
// 	LatThpt_PrepInt128();
//
// 	printf("\n");
// 	printf("XMM i128 Latency:\n");
// 	printf("----------------\n");
// 	printf(COM_NAME);
// 	printf(" \n");
// 	printf("Reg<-Reg    \n");
//
// 	testLatc_Xmm( COM );
// 	printf10(LatThpt_GetClocks());
//
// 	//
// 	// Perform xmm integer throughput tests.
// 	//
//
// 	LatThpt_PrepInt128();
//
// 	printf("\n\n");
// 	printf("XMM i128 Throughput:  \n");
// 	printf("--------------------  \n");
// 	printf(COM_NAME);
// 	printf(" \n");
// 	printf("Reg<-Reg    \n");
//
// 	testThpt_Xmm( COM );
// 	printf10(LatThpt_GetClocks());
// 	printf(" \n");
//
// 	//
// 	// Initialize latency/throughput macro resources.
// 	//
//
// 	LatThpt_Free();
//
// 	system("pause");
// 	return 0;
// }
