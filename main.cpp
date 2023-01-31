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
// #define CAP_IPC
// #define CAP_self

#include "bench.h"

void setup_gpu();

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
  int subgroupsize = atoi(argv[4]);
  double ratio = atof(argv[5]);
  int numgroup = numproc / groupsize;
  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of iterations %d\n", numiter);
    printf("Number of proc. per group: %d\n", groupsize);

    printf("Number of proc. per subgroup: %d\n", subgroupsize);
    printf("Ratio %f\n", ratio);

    printf("Number of groups: %d\n", numgroup);
    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Peer-to-peer count %ld ( %ld Bytes)\n", count, count * sizeof(Type));
    printf("send buffer: %lu (%.2f GB) recv buffer: %lu (%.2f GB)\n", count, count * sizeof(Type) / 1.e9, count * numgroup, count * numgroup * sizeof(Type) / 1.e9);
    printf("\n");
  }

  setup_gpu();

  MPI_Comm comm_group;
  MPI_Comm_split(MPI_COMM_WORLD, myid / groupsize, myid % groupsize, &comm_group);

  {
    CommBench::Bench<Type> global(MPI_COMM_WORLD, groupsize, CommBench::across, CommBench::MPI, count);
    CommBench::Bench<Type> local(comm_group, subgroupsize, CommBench::across, CommBench::MPI, count * ratio);
    //CommBench::Bench<Type> self(MPI_COMM_WORLD, subgroupsize, CommBench::within, CommBench::self, count * ratio);

    double totalTime = 0;
    double totalData = 0;
    double localTime = 0;
    double localData = 0;
    for(int iter = -1; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      global.start();
      local.start();
      // self.start();
      // self.wait();
      local.wait();
      MPI_Barrier(comm_group);
      double localtime = MPI_Wtime() - time;
      global.wait();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else {
        if(myid == ROOT)
          printf("time: %e\n", time);
        totalTime += time;
        totalData += 2 * (numgroup - 1) * count * sizeof(Type) / 1.e9;
        localTime += localtime;
        localData += 2 * (groupsize - 1) * count * sizeof(Type) / 1.e9 * ratio;
      }
    }
    if(myid == ROOT) {
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- Global\n\n", totalTime, totalData, totalData / totalTime * groupsize);
      printf("localTime: %e localData: %.2e GB (%e GB/s) --- Local\n\n", localTime, localData, localData / localTime * subgroupsize);
    }
  }

  return 0;

  {
    CommBench::Bench<Type> local(comm_group, subgroupsize, CommBench::across, CommBench::MPI, count * ratio);
    double totalTime = 0;
    double totalData = 0;
    for(int iter = -1; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      local.start();
      local.wait();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else {
        if(myid == ROOT)
          printf("time: %e\n", time);
        totalTime += time;
        totalData += 2 * (groupsize - 1) * count * sizeof(Type) / 1.e9 * ratio;
      }
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- Local\n\n", totalTime, totalData, totalData / totalTime * subgroupsize);
  }

  {
    CommBench::Bench<Type> self(MPI_COMM_WORLD, subgroupsize, CommBench::within, CommBench::self, count * ratio);
    double totalTime = 0;
    double totalData = 0;
    for(int iter = -1; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      self.start();
      self.wait();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else {
        if(myid == ROOT)
          printf("time: %e\n", time);
        totalTime += time;
        totalData += 2 * subgroupsize * count * sizeof(Type) / 1.e9 * ratio;
      }
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- Self\n\n", totalTime, totalData, totalData / totalTime);
  }

  // FINALIZE
  MPI_Finalize();

  return 0;
} // main()

void setup_gpu() {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

#ifdef PORT_CUDA
  if(myid == ROOT)
    printf("CUDA PORT\n");
  // SET DEVICE
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  cudaSetDevice(device);
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
  // DONE
  // REPORT
  if(myid == ROOT)
    system("rocm-smi");
#else
  if(myid == ROOT)
    printf("CPU VERSION\n");
  // DONE
#endif
}
