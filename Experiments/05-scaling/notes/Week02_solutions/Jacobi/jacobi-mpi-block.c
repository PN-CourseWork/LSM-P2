//  
//   Inspired by Advanced MPI (Archer Training Material,EPCC)
//  
//   http://www.archer.ac.uk/training/course-material/2016/09/160929_AdvMPI_EPCC/index.php
//   
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "mpi.h"

// Boundary values 
#define TOP 1.0
#define BOTTOM 10.0
#define LEFT 1.0
#define RIGHT 1.0
//
// The maximum number of iterations
#define MAX_ITERATIONS 5000000
// The convergence to terminate at
#define CONVERGENCE_ACCURACY 1e-4
// How often to report the norm
#define REPORT_NORM_PERIOD 1000

int nx, ny, ny2;
int max_iter = MAX_ITERATIONS;
int report_period = REPORT_NORM_PERIOD;

void initialise(double**, double**, int, int, int);
double* allocate_matrix_as_array(int nrows, int ncols);
double** allocate_matrix(int nrows, int ncols, double* arr_A);

int main(int argc, char * argv[]) {	
	int size, myrank;
	MPI_Status *status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc < 3) {
		if (myrank==0) fprintf(stderr, "You must provide the size in X and size in Y as arguments to this code\n");
		return -1;
	}
	nx=atoi(argv[1]);
	ny=atoi(argv[2]);
        ny2=ny+2;
	if (argc == 4) { 
	    max_iter = atoi(argv[3]);
	    report_period = max_iter + 1;
	}

	if (myrank==0) printf("Solving to accuracy of %.0e, global system size is x=%d y=%d (at most %d iterations)\n", CONVERGENCE_ACCURACY, nx, ny, max_iter);
	int local_nx=nx/size;
	if (local_nx * size < nx) {
		if (myrank < nx - local_nx * size) local_nx++;
	}

	double * grid1d =  allocate_matrix_as_array(local_nx+2,ny+2);
	double * grid_new1d = allocate_matrix_as_array(local_nx+2,ny+2);

        double **grid = allocate_matrix(local_nx+2,ny+2,grid1d);
        double **grid_new = allocate_matrix(local_nx+2,ny+2,grid_new1d);
	double start_time;

        MPI_Win win;
        MPI_Aint disp;

	initialise(grid, grid_new, local_nx, myrank, size);

	double rnorm=0.0, bnorm=0.0, norm, tmpnorm=0.0;
	MPI_Request requests[]={MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        // Initial nor factor
	int i,j,k;
	for (i=1;i<=local_nx;i++) {
		for (j=1;j<ny+1;j++) {		
			//tmpnorm=tmpnorm+pow(grid[i][j]*4-grid[i][j-1]-grid[i][j+1]-grid[i-1][j]-grid[i+1][j],2);
			double tmp = grid[i][j]*4-grid[i][j-1]
			             -grid[i][j+1]-grid[i-1][j]-grid[i+1][j];
			tmpnorm=tmpnorm+(tmp*tmp);
		}
	}
	MPI_Allreduce(&tmpnorm, &bnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	bnorm=sqrt(bnorm);

        //printf("bnorm=%lf\n",bnorm);
	start_time=MPI_Wtime();
	for (k=0;k<max_iter;k++) {
 
		//printf("Iteration %d starting.\n", k);
		if (myrank > 0) {
			MPI_Recv(&grid[0][1], ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&grid[1][1], ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD);
		}
		if (myrank < size-1) {
			MPI_Send(&grid[local_nx][1], ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(&grid[local_nx+1][1], ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		tmpnorm=0.0;
		//printf("Updating grid...\n");
		for (i=1;i<=local_nx;i++) {
			for (j=1;j<ny+1;j++) {
				//tmpnorm=tmpnorm+pow(grid[i][j]*4-grid[i][j-1]-grid[i][j+1]-grid[i-1][j]-grid[i+1][j],2);
			    double tmp = grid[i][j]*4-grid[i][j-1]
			             -grid[i][j+1]-grid[i-1][j]-grid[i+1][j];
			    tmpnorm=tmpnorm+(tmp*tmp);
			}
		}
		MPI_Allreduce(&tmpnorm, &rnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		norm=sqrt(rnorm)/bnorm;
		if (norm < CONVERGENCE_ACCURACY) break;		
		for (i=1;i<=local_nx;i++) {
			for (j=1;j<ny+1;j++) {
				grid_new[i][j]=0.25*(grid[i][j-1]+grid[i][j+1]+grid[i-1][j]+grid[i+1][j]);
			}
		}
		memcpy(grid1d, grid_new1d, sizeof(double) * (local_nx + 2) * ny2);

		if (k % report_period == 0 && myrank==0) printf("Iteration= %d Relative Norm=%e\n", k, norm);
	}
	if (myrank==0) printf("\nTerminated on %d iterations, Relative Norm=%e, Total time=%e seconds\n", k, norm,
			MPI_Wtime() - start_time);
	free(grid1d);
	free(grid_new1d);
	free(grid);
	free(grid_new);
	MPI_Finalize();
	return 0;
}

/**
 * Initialises the arrays, such that grid contains the boundary conditions at the start and end points and all other
 * points are zero. grid_new is set to equal grid
 */
void initialise(double ** grid, double ** grid_new, int local_nx, int myrank, int size) {
	int i, j;

        for (j=0;j<=local_nx+1;j++) {
                grid_new[j][0]=grid[j][0]=LEFT;
        }
        for (j=0;j<=local_nx+1;j++) {
                grid_new[j][ny2-1]=grid[j][ny2-1]=RIGHT;
        }

	for (j=0;j<=ny+1;j++) {
		grid_new[0][j]=grid[0][j]=myrank==0 ? TOP: 0;
	}
	for (j=0;j<=ny+1;j++) {
		grid_new[local_nx+1][j]=grid[local_nx+1][j]= myrank==size-1 ? BOTTOM: 0;		
	}
	for (i=1;i<=local_nx;i++) {
		for (j=1;j<ny+1;j++) {
			grid_new[i][j]=grid[i][j]=0;
		}
	}	
}


/* Allocate a double matrix with one malloc */
double* allocate_matrix_as_array(int nrows, int ncols) {
  double *arr_A;
  /* Allocate enough raw space */
  arr_A = (double*) malloc(nrows*ncols*sizeof(double));
  assert(arr_A != NULL);
  return arr_A;
}

/* Easy way to access a 1d dynamic array */
double** allocate_matrix(int nrows, int ncols, double* arr_A) {
  double **A;
  int ii;
  /* Prepare pointers for each matrix row */
  A = (double**) malloc(sizeof(double)*nrows);
  assert(A != NULL);
  /* Initialize the pointers */
  for (ii=0; ii<nrows; ++ii) {
    A[ii] = &(arr_A[ii*ncols]);
  }
  return A;
}

