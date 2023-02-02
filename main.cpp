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
// #include <nccl.h>
// #include <rccl.h>

// PORTS
// #define PORT_CUDA
// #define PORT_HIP

// CAPABILITIES
 #define MPI
// #define MPI_Staged
// #define NCCL
// #define IPC

// USER DEFINED TYPE
struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};

int main(int argc, char *argv[])
{
  // INITIALIZE MPI+OpenMP
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
  int warmup = atoi(argv[2]);
  int numiter = atoi(argv[3]);
  int groupsize = atoi(argv[4]);
  int numgroup = numproc / groupsize;
  int mygroup = myid / groupsize;
  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of warmup %d\n", warmup);
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
    printf("CUDA VERSION\n");
  // SET DEVICE
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  cudaSetDevice(device);
  // MEMORY MANAGEMENT
  cudaMalloc(&sendbuf_d, count * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * numgroup * sizeof(Type));
  cudaMemcpy(sendbuf_d, sendbuf, count * sizeof(Type), cudaMemcpyHostToDevice);
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
    printf("HIP VERSION\n");
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
  hipMemcpy(sendbuf_d, sendbuf, count * sizeof(Type), hipMemcpyHostToDevice);
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
  memcpy(sendbuf_d, sendbuf, count * sizeof(Type));
  // DONE
#endif

#ifdef MPI
  {
    if(myid == ROOT)
      printf("ENABLE GPU-Aware MPI\n");
    double totalTime = 0;
    double totalData = 0;
    for(int iter = -warmup; iter < numiter; iter++)
    {
#if !defined PORT_CUDA && !defined PORT_HIP
      memset(sendbuf_d, 0, count * sizeof(Type));
      memset(recvbuf_d, 0, numgroup * count * sizeof(Type));
#endif
      MPI_Request sendrequest[numgroup];
      MPI_Request recvrequest[numgroup];
      int sendproc = 0;
      int recvproc = 0;
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      for(int group = 0; group < numgroup; group++)
        if(group != mygroup) {
          MPI_Irecv(recvbuf_d + group * count, count * sizeof(Type), MPI_BYTE, group, MPI_ANY_TAG, comm, recvrequest + recvproc);
          recvproc++;
          MPI_Isend(sendbuf_d                , count * sizeof(Type), MPI_BYTE, group, 0, comm, sendrequest + sendproc);
          sendproc++;
        }
      MPI_Waitall(recvproc, recvrequest, MPI_STATUSES_IGNORE);
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0)
      {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else
      {
        if(myid == ROOT)
          printf("time: %e\n", time);
        totalTime += time;
        totalData += 2 * (numgroup - 1) * count * sizeof(Type) / 1.e9;
      }
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- GPU-Aware MPI\n", totalTime, totalData, totalData / totalTime * groupsize);
  }
#endif

#ifdef MPI_Staged
  { 
    if(myid == ROOT)
      printf("ENABLE CPU-Staged MPI\n");
    // ********************************************* SETUP CPU-Staged MPI **********************************************************
    Type *sendbuf_h;
    Type *recvbuf_h;
#ifdef PORT_CUDA
    cudaMallocHost(&sendbuf_h, count * numgroup * sizeof(Type));
    cudaMallocHost(&recvbuf_h, count * numgroup * sizeof(Type));
    cudaStream_t stream[numgroup];
    for(int group = 0; group < numgroup; group++)
      cudaStreamCreate(stream + group);
#elif defined PORT_HIP
    hipHostMalloc(&sendbuf_h, count * numgroup * sizeof(Type), 0);
    hipHostMalloc(&recvbuf_h, count * numgroup * sizeof(Type), 0);
    hipStream_t stream[numgroup];
    for(int group = 0; group < numgroup; group++)
      hipStreamCreate(stream + group);
#endif
    MPI_Request sendrequest[numgroup];
    MPI_Request recvrequest[numgroup];
    bool sendcomplete[numgroup];
    bool recvcomplete[numgroup];
    // *****************************************************************************************************************************
    double totalTime = 0;
    double totalData = 0;
    for(int iter = -warmup; iter < numiter; iter++)
    { 
      memset(sendcomplete, 0, numgroup);
      memset(recvcomplete, 0, numgroup);
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      for (int group = 0; group < numgroup; group++)
        if(group != mygroup) {
          MPI_Irecv(recvbuf_h + group * count, count * sizeof(Type), MPI_BYTE, group, MPI_ANY_TAG, comm, recvrequest + group);
#ifdef PORT_CUDA
          cudaMemcpyAsync(sendbuf_h + group * count, sendbuf_d, count * sizeof(Type), cudaMemcpyDeviceToHost, stream[group]);
#elif defined PORT_HIP
          hipMemcpyAsync(sendbuf_h + group * count, sendbuf_d, count * sizeof(Type), hipMemcpyDeviceToHost, stream[group]);
#endif
        }
      // SEND LOOP
      bool done_send = false;
      while(!done_send) {
        done_send = true;
        for(int group = 0; group < numgroup; group++)
          if(group != mygroup && !sendcomplete[group]) {
#ifdef PORT_CUDA
            if(cudaStreamQuery(stream[group]) == cudaSuccess) {
#elif defined PORT_HIP
            if(hipStreamQuery(stream[group]) == hipSuccess) {
#endif
              MPI_Isend(sendbuf_h + group * count, count * sizeof(Type), MPI_BYTE, group, 0, comm, sendrequest + group);
              sendcomplete[group] = true;
            }
            done_send = false;
          }
      }
      // MEMCPY LOOP
      bool done_recv = false;
      while(!done_recv) {
        done_recv = true;
        for(int group = 0; group < numgroup; group++)
          if(group != mygroup && !recvcomplete[group]) {
            int flag = 0;
            MPI_Test(recvrequest + group, &flag, MPI_STATUS_IGNORE);
            if(flag) {
#ifdef PORT_CUDA
              cudaMemcpyAsync(recvbuf_d + group * count, recvbuf_h + group * count, count * sizeof(Type), cudaMemcpyHostToDevice, stream[group]);
#elif defined PORT_HIP
              hipMemcpyAsync(recvbuf_d + group * count, recvbuf_h + group * count, count * sizeof(Type), hipMemcpyHostToDevice, stream[group]);
#endif
              recvcomplete[group] = true;
            }
            done_recv = false;
          }
      }
#ifdef PORT_CUDA
      cudaDeviceSynchronize();
#elif defined PORT_HIP
      hipDeviceSynchronize();
#endif
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0)
      {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else
      {
        if(myid == ROOT)
          printf("time: %e\n", time);
        totalTime += time;
        totalData += 2 * (numgroup - 1) * count * sizeof(Type) / 1.e9;
      }
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- CPU-Staged MPI\n", totalTime, totalData, totalData / totalTime * groupsize);

    for(int group = 0; group < numgroup; group++)
#ifdef PORT_CUDA
      cudaStreamDestroy(stream[group]);
#elif defined PORT_HIP
      hipStreamDestroy(stream[group]);
#endif
  }
#endif

#ifdef IPC
  {
#ifdef PORT_CUDA
    if(myid == ROOT)
      printf("ENABLE CUDA IPC\n");
    // ********************************************* SETUP CUDA IPC ****************************************************************
    Type *recvbuf_p[numgroup];
    {
      cudaIpcMemHandle_t peerhandle[numgroup];
      for(int group = 0; group < numgroup; group++) {
        cudaIpcMemHandle_t myhandle;
        Type *temp = recvbuf_d + group * count;
        cudaIpcGetMemHandle(&myhandle, temp);
        MPI_Gather(&myhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, peerhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, group, comm);
      }
      for(int group = 0; group < numgroup; group++)
        cudaIpcOpenMemHandle((void**)(recvbuf_p + group), peerhandle[group], cudaIpcMemLazyEnablePeerAccess);
    }
    cudaStream_t stream_ipc[numgroup];
    for(int group = 0; group < numgroup; group++)
      cudaStreamCreate(stream_ipc + group);
    // *****************************************************************************************************************************
    double totalData = 0;
    double totalTime = 0;
    for(int iter = -warmup; iter < numiter; iter++)
    {
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      for(int group = 0; group < numgroup; group++)
        if(mygroup != group)
          cudaMemcpyAsync(recvbuf_p[group] + mygroup * count, sendbuf_d, count * sizeof(Type), cudaMemcpyDeviceToDevice, stream_ipc[group]);
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0)
      { 
        if(myid == ROOT)
          printf("warmup time %e\n", time);
      }
      else
      {
        if(myid == ROOT)
          printf("time %e\n", time);
        totalTime += time;
        totalData += 2 * (numgroup - 1) * count * sizeof(Type) / 1.e9;
      }
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- IPC\n", totalTime, totalData, totalData / totalTime * groupsize);
    for(int group = 0; group < numgroup; group++)
      cudaStreamDestroy(stream_ipc[group]);
#elif defined PORT_HIP
    if(myid == ROOT)
      printf("ENABLE HIP IPC\n");
    // ********************************************** SETUP HIP IPC ****************************************************************
    Type *recvbuf_p[numgroup];
    {
      hipIpcMemHandle_t peerhandle[numgroup];
      for(int group = 0; group < numgroup; group++) {
        hipIpcMemHandle_t myhandle;
        Type *temp = recvbuf_d + group * count;
        hipIpcGetMemHandle(&myhandle, temp);
        MPI_Gather(&myhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, peerhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, group, comm);
      }
      for(int group = 0; group < numgroup; group++)
        hipIpcOpenMemHandle((void**)(recvbuf_p + group), peerhandle[group], hipIpcMemLazyEnablePeerAccess);
    }
    hipStream_t stream_ipc[numgroup];
    for(int group = 0; group < numgroup; group++)
      hipStreamCreate(stream_ipc + group);
    // *****************************************************************************************************************************
    double totalData = 0;
    double totalTime = 0;
    for(int iter = -warmup; iter < numiter; iter++)
    {
      hipDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      for(int group = 0; group < numgroup; group++)
        if(mygroup != group)
          hipMemcpyAsync(recvbuf_p[group] + mygroup * count, sendbuf_d, count * sizeof(Type), hipMemcpyDeviceToDevice, stream_ipc[group]);
      hipDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0)
      {
        if(myid == ROOT)
          printf("warmup time %e\n", time);
      }
      else
      {
        if(myid == ROOT)
          printf("time %e\n", time);
        totalTime += time;
        totalData += 2 * (numgroup - 1) * count * sizeof(Type) / 1.e9;
      }
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- IPC\n", totalTime, totalData, totalData / totalTime * groupsize);
    for(int group = 0; group < numgroup; group++)
      hipStreamDestroy(stream_ipc[group]);
#endif
  }
#endif

#ifdef NCCL
  {
    if(myid == ROOT)
      printf("ENABLE NCCL\n");
    // ************************************************* SETUP NCCL ****************************************************************
    ncclComm_t comm_nccl;
    ncclUniqueId id;
    if(myid / groupsize == 0)
      ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
    ncclCommInitRank(&comm_nccl, numgroup, id, myid / groupsize);
    // *****************************************************************************************************************************
    double totalTime = 0;
    double totalData = 0;
    for(int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      ncclGroupStart();
      for(int group = 0; group < numgroup; group++)
        if(myid / groupsize != group) {
            ncclSend(sendbuf_d,                count * sizeof(Type), ncclInt8, group, comm_nccl, 0);
            ncclRecv(recvbuf_d + group * count, count * sizeof(Type), ncclInt8, group, comm_nccl, 0);
        }
      ncclGroupEnd();
#ifdef PORT_CUDA
      cudaDeviceSynchronize();
#elif defined PORT_HIP
      hipDeviceSynchronize();
#endif
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0)
      {
        if(myid == ROOT)
          printf("warmup time %e\n", time);
      }
      else
      {
        if(myid == ROOT)
          printf("time %e\n", time);
        totalTime += time;
        totalData += 2 * (numgroup - 1) * count * sizeof(Type) / 1.e9;
      }
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- NCCL\n", totalTime, totalData, totalData / totalTime * groupsize);

    ncclCommDestroy(comm_nccl);
  }
#endif

  // RELEASE GPU POINTERS
#ifdef PORT_CUDA
  cudaFree(sendbuf_d);
  cudaFree(recvbuf_d);
#elif defined PORT_HIP
  hipFree(sendbuf_d);
  hipFree(recvbuf_d);
#else
  delete[] sendbuf_d;
  delete[] recvbuf_d;
#endif
  // RELEASE CPU POINTERS
  delete[] sendbuf;
  delete[] recvbuf;

  // FINALIZE
  MPI_Finalize();

  return 0;
}

