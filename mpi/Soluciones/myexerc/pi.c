#include <mpi.h>
#include <math.h>
#include <stdio.h>


double function(double x)
{
	return 4/(1 + (x*x));
}



int main(int argc, char *argv[])
{
	int rank, size, N;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double my_init, my_end, my_interval, my_subtot, my_total, i;
	N = 10000;
	my_init = (double)rank * 1.0/((double)size);
	my_end = my_init + 1.0/(double)size;
	my_interval = 1.0/((double)size * (double)N);
	my_subtot = 0;
	my_total = 0;	
	printf("my_init = %lf , my_end = %lf, my_interval = %lf \n", my_init, my_end, my_interval);
	for (i = my_init; i < my_end; i+= my_interval)
	{
		my_subtot += my_interval * function(i);
	}
	MPI_Reduce(&my_subtot, &my_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if (rank == 0)
	{
		printf("PI is approximately %lf \n", my_total);
	}
	

	MPI_Finalize();
	return 0;
}



