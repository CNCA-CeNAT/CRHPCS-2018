#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Main routine
int main (int argc, char *argv[]){
    int rank,size,length;
    char name[25];
    int position, i, j, a[2];
    char buff[1000];
    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

   if (rank == 0)
	{
	   i = 102;
	   j = 666;
	   position = 0;
	   MPI_Pack(&i, 1, MPI_INT, buff, 1000, &position, MPI_COMM_WORLD);
	   MPI_Pack(&j, 1, MPI_INT, buff, 1000, &position, MPI_COMM_WORLD);
	   MPI_Send(buff, position, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
	}

  else
	{
		MPI_Recv(a,2,MPI_INT,0,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Soy %d y recibí %d y %d \n", rank, a[0], a[1]);
	}

    // finalize MPI
    MPI_Finalize();
    return 0;
}

