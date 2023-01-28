/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h> // for printf
#include <stdlib.h> // for atoi
#include <cstring> // for memcpy
#include <mpi.h>
#include <omp.h>

#define ROOT 0

// HEADERS
 #include <nccl.h>
// #include <rccl.h>

// PORTS
 #define PORT_CUDA
// #define PORT_HIP

// CAPABILITIES
 #define CAP_MPI
 #define CAP_MPI_staged
 #define CAP_NCCL
 #define CAP_IPC

#include "commbench.h"

#define MEASURE(numiter, comm)                                          \
      MPI_Barrier(comm);                                                \
      for(int iter = -1; iter < numiter; iter++) {                      \
        double time = MPI_Wtime();                                      \
        alltoall.start();                                               \
        alltoall.wait();                                                \
        MPI_Barrier(comm);                                              \
        time = MPI_Wtime() - time;                                      \
        if(iter < 0) {                                                  \
          if(myid == ROOT)                                              \
            printf("warmup time: %e\n", time);                          \
        }                                                               \
        else {                                                          \
          if(myid == ROOT)                                              \
            printf("time: %e\n", time);                                 \
          totalTime += time;                                            \
          totalData += 2 * (numproc - 1) * count * sizeof(Type) / 1.e9; \
        }                                                               \
      }                                                                 \

// USER DEFINED TYPE
struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};

int main(int argc, char *argv[])
{
  // INITIALIZE MPI+OPENMP
  int myid;
  int numproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  size_t count = atoi(argv[1]);
  int numiter = atoi(argv[2]);
  int groupsize = atoi(argv[3]);
  int subgroupsize = atoi(argv[4]);
  double ratio = atof(argv[5]);


  {
    using namespace CommBench;

    Bench<Type> bench(groupsize, subgroupsize, MPI_COMM_WORLD);

    bench.init(count, MPI_staged, MPI, ratio);

    size_t sendcount[numproc];
    size_t recvcount[numproc];
    size_t sendoffset[numproc] = {0};
    size_t recvoffset[numproc];
    int block = 0;
    for (int p = 0; p < numproc; p++) {
      sendcount[p] = (p != myid ? count : 0);
      recvcount[p] = (p != myid ? count : 0);
      if(recvcount[p]) {
        recvoffset[p] = block * count;
        block++;
      }
    }

    Type *sendbuf;
    Type *recvbuf;
#ifdef PORT_CUDA
    cudaMalloc(&sendbuf, count * sizeof(Type));
    cudaMalloc(&recvbuf, count * (numproc - 1) * sizeof(Type));
#elif defined(PORT_HIP)
    hipMalloc(&sendbuf, count * sizeof(Type));
    hipMalloc(&recvbuf, count * (numproc - 1) * sizeof(Type));
#else
    sendbuf = new Type[count];
    recvbuf = new Type[count * (numproc - 1)];
#endif

#ifdef CAP_MPI
    {
      Comm<Type> alltoall(sendbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, MPI_COMM_WORLD, MPI);

      double totalTime = 0;
      double totalData = 0;

      MEASURE(numiter, MPI_COMM_WORLD);

      if(myid == ROOT)
        printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- GPU-Aware MPI\n", totalTime, totalData, totalData / totalTime);
    }
#endif

#ifdef CAP_MPI_staged
    {
      Comm<Type> alltoall(sendbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, MPI_COMM_WORLD, MPI_staged);
    }
#endif

#ifdef CAP_NCCL
    {
      Comm<Type> alltoall(sendbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, MPI_COMM_WORLD, NCCL);

      double totalTime = 0;
      double totalData = 0;

      MEASURE(numiter, MPI_COMM_WORLD)

      if(myid == ROOT)
        printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- NCCL\n", totalTime, totalData, totalData / totalTime);
    }
#endif

#ifdef CAP_IPC
    {
      Comm<Type> alltoall(sendbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, MPI_COMM_WORLD, IPC);

      double totalTime = 0;
      double totalData = 0;

      MEASURE(numiter, MPI_COMM_WORLD)

      if(myid == ROOT)
        printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- IPC\n", totalTime, totalData, totalData / totalTime);
    }
#endif

  }

  // FINALIZE
  MPI_Finalize();

  return 0;
}

