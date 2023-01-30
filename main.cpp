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
// #define CAP_MPI_staged
 #define CAP_NCCL
 #define CAP_IPC
 #define CAP_self

#include "bench.h"

#define MEASURE_MACRO(numiter)                                           \
      for(int iter = -1; iter < numiter; iter++) {                       \
        MPI_Barrier(MPI_COMM_WORLD);                                     \
        double time = MPI_Wtime();                                       \
        across.start();                                                  \
        across.wait();                                                   \
        MPI_Barrier(MPI_COMM_WORLD);                                     \
        time = MPI_Wtime() - time;                                       \
        if(iter < 0) {                                                   \
          if(myid == ROOT)                                               \
            printf("warmup time: %e\n", time);                           \
        }                                                                \
        else {                                                           \
          if(myid == ROOT)                                               \
            printf("time: %e\n", time);                                  \
          totalTime += time;                                             \
          totalData += 2 * (numgroup - 1) * count * sizeof(Type) / 1.e9; \
        }                                                                \
      }

#define MEASURE_SELF_MACRO(numiter)                                      \
      for(int iter = -1; iter < numiter; iter++) {                       \
        MPI_Barrier(MPI_COMM_WORLD);                                     \
        double time = MPI_Wtime();                                       \
        across.start();                                                  \
        across.wait();                                                   \
        MPI_Barrier(MPI_COMM_WORLD);                                     \
        time = MPI_Wtime() - time;                                       \
        if(iter < 0) {                                                   \
          if(myid == ROOT)                                               \
            printf("warmup time: %e\n", time);                           \
        }                                                                \
        else {                                                           \
          if(myid == ROOT)                                               \
            printf("time: %e\n", time);                                  \
          totalTime += time;                                             \
          totalData += 2 * numgroup * count * sizeof(Type) / 1.e9;       \
        }                                                                \
      }       

#define TEST_CUDA_MACRO                                                                        \
      if(myid == ROOT)                                                                         \
        printf("TEST CUDA\n");                                                                 \
      cudaMemcpy(sendbuf_d, sendbuf, count * sizeof(Type), cudaMemcpyHostToDevice);            \
      cudaMemset(recvbuf_d, -1, numgroup * count * sizeof(Type));                              \
      MPI_Barrier(MPI_COMM_WORLD);                                                             \
      across.start();                                                                          \
      across.wait();                                                                           \
      MPI_Barrier(MPI_COMM_WORLD);                                                             \
      cudaMemcpy(recvbuf, recvbuf_d, numgroup * count * sizeof(Type), cudaMemcpyDeviceToHost);
#define TEST_HIP_MACRO                                                                         \
      if(myid == ROOT)                                                                         \
        printf("TEST HIP\n");                                                                  \
      hipMemcpy(sendbuf_d, sendbuf, count * sizeof(Type), hipMemcpyHostToDevice);              \
      hipMemset(recvbuf_d, -1, numgroup * count * sizeof(Type));                               \
      MPI_Barrier(MPI_COMM_WORLD);                                                             \
      across.start();                                                                          \
      across.wait();                                                                           \
      MPI_Barrier(MPI_COMM_WORLD);                                                             \
      hipMemcpy(recvbuf, recvbuf_d, numgroup * count * sizeof(Type), hipMemcpyDeviceToHost);
#define TEST_CPU_MACRO                                                                         \
      if(myid == ROOT)                                                                         \
        printf("TEST CPU\n");                                                                  \
      memcpy(sendbuf_d, sendbuf, count * sizeof(Type));                                        \
      memset(recvbuf_d, -1, numgroup * count * sizeof(Type));                                  \
      MPI_Barrier(MPI_COMM_WORLD);                                                             \
      across.start();                                                                          \
      across.wait();                                                                           \
      MPI_Barrier(MPI_COMM_WORLD);                                                             \
      memcpy(recvbuf, recvbuf_d, numgroup * count * sizeof(Type));
#define TEST_CROSS_MACRO \
      for(int group = 0; group < numgroup; group++) \
        if(group != mygroup) \
          for(size_t i = 0; i < count; i++) \
            if(recvbuf[group * count + i].data[0] != group * groupsize + mygroup) \
              test = false;
#define TEST_SELF_MACRO                                                            \
      for(int group = 0; group < numgroup; group++)                                \
        for(size_t i = 0; i < count; i++)                                          \
          if(recvbuf[group * count + i].data[0] != myid)                           \
            test = false; 

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
  int numthread;
  #pragma omp parallel
  if(omp_get_thread_num() == 0)
    numthread = omp_get_num_threads();

  size_t count = atoi(argv[1]);
  int numiter = atoi(argv[2]);
  int groupsize = atoi(argv[3]);
  int numgroup = numproc / groupsize;
  int mygroup = myid / groupsize;
  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of iterations %d\n", numiter);
    printf("Number of proc. per group: %d\n", groupsize);
    printf("Number of groups: %d\n", numgroup);
    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Peer-to-peer count %ld ( %ld Bytes)\n", count, count * sizeof(Type));
    printf("send buffer: %lu (%.2f GB) recv buffer: %lu (%.2f GB)\n", count, count * sizeof(Type) / 1.e9, count * numgroup, count * numgroup * sizeof(Type) / 1.e9);
    printf("\n");
  }

  Type *sendbuf = new Type[count];
  Type *recvbuf = new Type[count * numgroup];
  Type *sendbuf_d;
  Type *recvbuf_d;

  for(int i = 0; i < count; i++) 
    sendbuf[i].data[0] = myid;


  MPI_Comm comm;
  MPI_Comm_split(MPI_COMM_WORLD, myid % groupsize, mygroup, &comm);

#ifdef PORT_CUDA
  if(myid == ROOT)
    printf("CUDA PORT\n");
  // SET DEVICE
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  cudaSetDevice(device);
  // MEMORY MANAGEMENT
  cudaMalloc(&sendbuf_d, count * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * numgroup * sizeof(Type));
  // DONE
  // REPORT
  if(myid == ROOT){
    system("nvidia-smi");
    int deviceCount;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceCount(&deviceCount);
    printf("Device %d Count: %d\n", device, deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);
    printf("Device %d name: %s\n",0,deviceProp.name);
    printf("Clock Frequency: %f GHz\n",deviceProp.clockRate/1.e9);
    printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
    printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Warp size: %d\n",deviceProp.warpSize);
    printf("32-bit Reg. per block: %d\n",deviceProp.regsPerBlock);
    printf("\n");
  }
#elif defined PORT_HIP
  if(myid == ROOT)
    printf("HIP PORT\n");
  //DEVICE MANAGEMENT
  int deviceCount;
  hipGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  if(myid == ROOT)
    printf("deviceCount: %d\n", deviceCount);
  hipSetDevice(device);
  // MEMORY MANAGEMENT
  hipMalloc(&sendbuf_d, count * sizeof(Type));
  hipMalloc(&recvbuf_d, count * numgroup * sizeof(Type));
  // DONE
  // REPORT
  if(myid == ROOT)
    system("rocm-smi");
#else
  if(myid == ROOT)
    printf("CPU VERSION\n");
  // MEMORY MANAGEMENT
  sendbuf_d = new Type[count];
  recvbuf_d = new Type[count * numgroup];
  // DONE
#endif

  CommBench::Bench<Type> bench_across(MPI_COMM_WORLD, groupsize, CommBench::across, CommBench::MPI, count);

  CommBench::Bench<Type> bench_within(MPI_COMM_WORLD, groupsize, CommBench::within, CommBench::NCCL, count);

  return 0;

  // MEASURE CROSS COMMUNICATION
  {
    size_t sendcount[numgroup];
    size_t sendoffset[numgroup];
    size_t recvcount[numgroup];
    size_t recvoffset[numgroup];
    for (int group = 0; group < numgroup; group++)
      if(group != mygroup) {
        sendcount[group] = count;
        sendoffset[group] = 0;
        recvcount[group] = count;
        recvoffset[group] = group * count;
      }
      else {
        sendcount[group] = 0;
        recvcount[group] = 0;
      }
#ifdef CAP_MPI
    {
      CommBench::Comm<Type> across(sendbuf_d, sendcount, sendoffset, recvbuf_d, recvcount, recvoffset, comm, CommBench::MPI);
      double totalTime = 0;
      double totalData = 0;
      MEASURE_MACRO(numiter)
      if(myid == ROOT)
        printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- GPU-Aware MPI\n", totalTime, totalData, totalData / totalTime * groupsize);
#ifdef PORT_CUDA
      TEST_CUDA_MACRO
#elif defined PORT_HIP
      TEST_HIP_MACRO
#else
      TEST_CPU_MACRO
#endif
      bool test = true; 
      TEST_CROSS_MACRO
      if(test && myid == ROOT)
        printf("TEST PASSED!\n");
      else if(myid == ROOT)
        printf("TEST FAILED!!!\n");
    }
#endif

#ifdef CAP_MPI_staged
    {
      Comm<Type> across(sendbuf_d, sendcount, sendoffset, recvbuf_d, recvcount, recvoffset, comm, CommBench::MPI_staged);
      double totalTime = 0;
      double totalData = 0;
      MEASURE_MACRO(numiter)
      if(myid == ROOT)
        printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- CPU-Staged MPI\n", totalTime, totalData, totalData / totalTime * groupsize);
#ifdef PORT_CUDA
      TEST_CUDA_MACRO
#elif defined PORT_HIP
      TEST_HIP_MACRO
#endif
      bool test = true;
      TEST_CROSS_MACRO
      if(test && myid == ROOT)
        printf("TEST PASSED!\n");
      else if(myid == ROOT)
        printf("TEST FAILED!!!\n");
    }
#endif

#ifdef CAP_NCCL
    {
      CommBench::Comm<Type> across(sendbuf_d, sendcount, sendoffset, recvbuf_d, recvcount, recvoffset, comm, CommBench::NCCL);
      double totalTime = 0;
      double totalData = 0;
      MEASURE_MACRO(numiter)
      if(myid == ROOT)
        printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- NCCL\n", totalTime, totalData, totalData / totalTime * groupsize);
#ifdef PORT_CUDA
      TEST_CUDA_MACRO
#elif defined PORT_HIP
      TEST_HIP_MACRO
#endif
      bool test = true;
      TEST_CROSS_MACRO
      if(test && myid == ROOT)
        printf("TEST PASSED!\n");
      else if(myid == ROOT)
        printf("TEST FAILED!!!\n");
    }
#endif

#ifdef CAP_IPC
    {
      CommBench::Comm<Type> across(sendbuf_d, sendcount, sendoffset, recvbuf_d, recvcount, recvoffset, comm, CommBench::IPC);
      double totalTime = 0;
      double totalData = 0;
      MEASURE_MACRO(numiter)
      if(myid == ROOT)
        printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- IPC\n", totalTime, totalData, totalData / totalTime * groupsize);
#ifdef PORT_CUDA
      TEST_CUDA_MACRO
#elif defined PORT_HIP
      TEST_HIP_MACRO
#endif
      bool test = true;
      TEST_CROSS_MACRO
      if(test && myid == ROOT)
        printf("TEST PASSED!\n");
      else if(myid == ROOT)
        printf("TEST FAILED!!!\n");
    }
#endif
  }

  // MEASURE SELF COMMUNICATION
  {
    size_t sendcount[numgroup];
    size_t sendoffset[numgroup];
    size_t recvcount[numgroup];
    size_t recvoffset[numgroup];
    for (int group = 0; group < numgroup; group++) {
      sendcount[group] = count;
      sendoffset[group] = 0;
      recvcount[group] = count;
      recvoffset[group] = group * count;
    }
#ifdef CAP_self
    {
      CommBench::Comm<Type> across(sendbuf_d, sendcount, sendoffset, recvbuf_d, recvcount, recvoffset, comm, CommBench::self);

      double totalTime = 0;
      double totalData = 0;
      MEASURE_SELF_MACRO(numiter)
      if(myid == ROOT)
        printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- self\n", totalTime, totalData, totalData / totalTime);
#ifdef PORT_CUDA
      TEST_CUDA_MACRO
#elif defined PORT_HIP
      TEST_HIP_MACRO
#else
      TEST_CPU_MACRO
#endif
      bool test = true;
      TEST_SELF_MACRO
      if(test && myid == ROOT)
        printf("TEST PASSED!\n");
      else if(myid == ROOT)
        printf("TEST FAILED!!!\n");
    }
#endif
  }

  // FINALIZE
  MPI_Finalize();

  return 0;
} // main()

