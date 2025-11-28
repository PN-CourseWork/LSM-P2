/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 *
 * Sample MPI "hello world" application in C
 */

#include <stdio.h>
#include <unistd.h>
#include "mpi.h"

int 
main(int argc, char* argv[]) {

    int rank, size;
    #define HOST_NAME_MAX 64
    char pname[MPI_MAX_PROCESSOR_NAME];
    int pnamelen;
    pid_t mypid;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Get_processor_name(pname, &pnamelen);
    mypid = getpid();
    printf("Hello, world, I am %d of %d, running on %s (PID: %d).\n", 
           rank, size, pname, mypid);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
