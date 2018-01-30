/**
 * Costa Rica Institute of Technology
 * School of Computing
 * ce5504: High Performance Computing
 * Instructor Esteban Meneses, PhD (emeneses@ic-itcr.ac.cr)
 * OpenMP parallel Strassen algorithm for matrix multiplication.
 * The parallel programming pattern used is Fork-join.
 */

#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "timer.h"
#include "io.h"


// Function to allocate matrix. It initializes all entries to 0.
int **allocMatrix(int N){
	int **ptr;
	
	ptr = new int*[N];
	for (int i=0; i<N; i++){
		ptr[i] = new int[N];
	}
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			ptr[i][j] = 0;
	return ptr;
}

// Function to free matrix
void freeMatrix(int** ptr, int N){
	for (int i=0; i<N; i++) {
		delete [] ptr[i];
	}
	delete [] ptr;
}

// Computes a substep in Strassen's algorithm: R = (X+Y)(Z+W)
void substep(int **X, int **Y, int **Z, int **W, int **R, int N){
	int **first, **second;

	// allocate memory
	first = allocMatrix(N); second = allocMatrix(N);

	// computing the first term
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			first[i][j] = X[i][j] + Y[i][j];
	
	// computing the second term
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			second[i][j] = Z[i][j] + W[i][j];

	// computing the final product
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			for(int k=0; k<N; k++)
				R[i][j] += first[i][k]*second[k][j];
	
	// free memory
	freeMatrix(first,N); freeMatrix(second,N);
}

// Computes the operation of matrices: R=X+Y+Z-W
void step(int **X, int **Y, int **Z, int **W, int **R, int N){	
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			R[i][j] = X[i][j] + Y[i][j] + Z[i][j] - W[i][j];
}

// Computes the additive inverse of a matrix, Y = -X
void inverse(int **X, int **Y, int N){
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			Y[i][j] = -X[i][j];
}

// Extracts a submatrix Y = X[initI:initI+N][initJ:initJ+N]
void extract(int **X, int **Y, int initI, int initJ, int N){
	for(int i=initI; i<initI+N; i++)
		for(int j=initJ; j<initJ+N; j++)
			Y[i-initI][j-initJ] = X[i][j];
}

// Inserts a submatrix Y into X, X[initI:initI+N][initJ:initJ+N] = Y
void insert(int **Y, int **X, int initI, int initJ, int N){
	for(int i=initI; i<initI+N; i++)
		for(int j=initJ; j<initJ+N; j++)
			X[i][j] = Y[i-initI][j-initJ];
}

// Function implementing Strassen's algorithm
void strassen(int **A, int **B, int **C, int N){
	int **A11, **A12, **A21, **A22, **B11, **B12, **B21, **B22, **C11, **C12, **C21, **C22;
	int **mA11, **mA22, **mB11, **mB22;
	int **M1, **M2, **M3, **M4, **M5, **M6, **M7, **ZERO;

	// allocating space for sub-matrices
	A11 = allocMatrix(N/2); A12 = allocMatrix(N/2); A21 = allocMatrix(N/2); A22 = allocMatrix(N/2);
	B11 = allocMatrix(N/2); B12 = allocMatrix(N/2); B21 = allocMatrix(N/2); B22 = allocMatrix(N/2);
	C11 = allocMatrix(N/2); C12 = allocMatrix(N/2); C21 = allocMatrix(N/2); C22 = allocMatrix(N/2);
	mA11 = allocMatrix(N/2); mA22 = allocMatrix(N/2); mB11 = allocMatrix(N/2); mB22 = allocMatrix(N/2);
	M1 = allocMatrix(N/2); M2 = allocMatrix(N/2); M3 = allocMatrix(N/2); M4 = allocMatrix(N/2);
	M5 = allocMatrix(N/2); M6 = allocMatrix(N/2); M7 = allocMatrix(N/2); ZERO = allocMatrix(N/2);

	// extracting all relevant submatrices
	extract(A,A11,0,0,N/2); extract(A,A12,0,N/2,N/2); extract(A,A21,N/2,0,N/2); extract(A,A22,N/2,N/2,N/2);
	extract(B,B11,0,0,N/2); extract(B,B12,0,N/2,N/2); extract(B,B21,N/2,0,N/2); extract(B,B22,N/2,N/2,N/2);

	// computing additive inverse of submatrices
	inverse(A11,mA11,N/2); inverse(A22,mA22,N/2); inverse(B11,mB11,N/2); inverse(B22,mB22,N/2);

	// performing all substeps to compute matrices M
	#pragma omp task
	substep(A11,A22,B11,B22,M1,N/2);				// (A11 + A22)(B11 + B22)	
	#pragma omp task
	substep(A21,A22,ZERO,B11,M2,N/2);				// (A21 + A22)B11
	#pragma omp task
	substep(ZERO,A11,B12,mB22,M3,N/2);				// A11(B12 - B22)
	#pragma omp task
	substep(ZERO,A22,B21,mB11,M4,N/2);				// A22(B21 - B11)
	#pragma omp task
	substep(A11,A12,ZERO,B22,M5,N/2);				// (A11 + A12)B22
	#pragma omp task
	substep(A21,mA11,B11,B12,M6,N/2);				// (A21 - A11)(B11 + B12)
	substep(A12,mA22,B21,B22,M7,N/2);				// (A12 - A22)(B21 + B22)
	#pragma omp taskwait

	// performing all steps to compute submatrices C
	step(M1,M4,M7,M5,C11,N/2);						// M1 + M4 - M5 + M7
	step(M3,M5,ZERO,ZERO,C12,N/2);					// M3 + M5
	step(M2,M4,ZERO,ZERO,C21,N/2);					// M2 + M4
	step(M1,M3,M6,M2,C22,N/2);						// M1 - M2 +M3 +M6

	// inserting submatrices into final result	
	insert(C11,C,0,0,N/2); insert(C12,C,0,N/2,N/2);	insert(C21,C,N/2,0,N/2); insert(C22,C,N/2,N/2,N/2);	

	// deallocating space
	freeMatrix(A11,N/2); freeMatrix(A12,N/2); freeMatrix(A21,N/2); freeMatrix(A22,N/2);
	freeMatrix(B11,N/2); freeMatrix(B12,N/2); freeMatrix(B21,N/2); freeMatrix(B22,N/2);
	freeMatrix(C11,N/2); freeMatrix(C12,N/2); freeMatrix(C21,N/2); freeMatrix(C22,N/2);
	freeMatrix(mA11,N/2); freeMatrix(mA22,N/2); freeMatrix(mB11,N/2); freeMatrix(mB22,N/2);
	freeMatrix(M1,N/2); freeMatrix(M2,N/2); freeMatrix(M3,N/2); freeMatrix(M4,N/2);
	freeMatrix(M5,N/2); freeMatrix(M6,N/2); freeMatrix(M7,N/2); freeMatrix(ZERO, N/2);

}


// Main method      
int main(int argc, char* argv[]) {
	int N;
	int **A, **B, **C;
	double elapsedTime;

	// checking parameters
	if (argc != 2 && argc != 4) {
		cout << "Parameters: <N> [<fileA> <fileB>]" << endl;
		return 1;
	}
	N = atoi(argv[1]);

	// allocating matrices
	A = new int*[N];
	B = new int*[N];
	C = new int*[N];
	for (int i=0; i<N; i++){
		A[i] = new int[N];
		B[i] = new int[N];
		C[i] = new int[N];
	}

	// reading files (optional)
	if(argc == 4){
		readMatrixFile(A,N,argv[2]);
		readMatrixFile(B,N,argv[3]);
	}

	// starting timer
	timerStart();

	#pragma omp parallel
	{
		#pragma omp single
		strassen(A,B,C,N); // calling Strassen algorithm function
	}

	// testing the results is correct
	if(argc == 4){
		printMatrix(C,N);
	}
	
	// stopping timer
	elapsedTime = timerStop();

	cout << "Duration: " << elapsedTime << " seconds" << std::endl;

	// releasing memory
	for (int i=0; i<N; i++) {
		delete [] A[i];
		delete [] B[i];
		delete [] C[i];
	}
	delete [] A;
	delete [] B;
	delete [] C;

	return 0;	
}

