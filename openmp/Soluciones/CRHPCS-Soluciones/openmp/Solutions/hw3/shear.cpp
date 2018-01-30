/**
 * Costa Rica Institute of Technology
 * School of Computing
 * ce5504: High Performance Computing
 * Instructor Esteban Meneses, PhD (emeneses@ic-itcr.ac.cr)
 * Input/output operations for matrices.
 * OpenMP parallel shear sort.
 */

#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include "timer.h"
#include "io.h"

#define MAX_VALUE 10000

void sort_row(int **A, int M, int i, int order);
void sort_col(int **A, int M, int j);

// Shear Sort function
void shear_sort(int **A, int M){
    int N = M*M, total;
    total = ceil(log2(N));
    
    for(int stage=0; stage<total; stage++){
        
        // sorting rows
		#pragma omp parallel for
        for(int i=0; i<M; i++){
            sort_row(A,M,i,int(pow(-1,i)));
        }
        
        // sorting columns
		#pragma omp parallel for
        for(int j=0; j<M; j++){
            sort_col(A,M,j);
        }
    }
}

// Function to sort row (uses bubble sort)
void sort_row(int **A, int M, int i, int order){
    for(int step=0; step<M; step++) {
        for(int j=1; j<M; j++){
            if(A[i][j-1]*order > A[i][j]*order) {
                int tmp = A[i][j-1];
                A[i][j-1] = A[i][j];
                A[i][j] = tmp;
            }
        }
    }
}

// Function to sort column (uses bubble sort)
void sort_col(int **A, int M, int j){
    for(int step=0; step<M; step++) {
        for(int i=1; i<M; i++){
            if(A[i-1][j] > A[i][j]) {
                int tmp = A[i-1][j];
                A[i-1][j] = A[i][j];
                A[i][j] = tmp;
            }
        }
    }
}

// Main method      
int main(int argc, char* argv[]) {
	int N, M;
	int **A;
	double elapsedTime;

	// checking parameters
	if (argc != 2 && argc != 3) {
		cout << "Parameters: <N> [<file>]" << endl;
		return 1;
	}
	N = atoi(argv[1]);
	M = (int) sqrt(N); 
	if(N != M*M){
		cout << "N has to be a perfect square!" << endl;
		exit(1);
	}	

	// allocating matrix A
	A = new int*[M];
	for (int i=0; i<M; i++){
		A[i] = new int[M];
	}

	// reading files (optional)
	if(argc == 3){
		readMatrixFile(A,M,argv[2]);
	} else {
		srand48(time(NULL));
		for(int i=0; i<M; i++){
			for(int j=0; j<M; j++){
				A[i][j] = lrand48() % MAX_VALUE;
			}
		}
	}
	
	// starting timer
	timerStart();

	// calling shear sort function
	shear_sort(A,M);

	// testing the results is correct
	if(argc == 3){
		printMatrix(A,M);
	}
	
	// stopping timer
	elapsedTime = timerStop();

	cout << "Duration: " << elapsedTime << " seconds" << std::endl;

	// releasing memory
	for (int i=0; i<M; i++) {
		delete [] A[i];
	}
	delete [] A;

	return 0;	
}

