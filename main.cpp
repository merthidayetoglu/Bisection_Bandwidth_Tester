// Copyright Mert Hidayetoglu
// Jan 2023, Stanford, CA

#include <cstdio> // for printf
#include <cstring> // for memcpy
#include <mpi.h>
#include <omp.h>

#include <hip/hip_runtime.h>
#include <rccl.h>
//#include <nccl.h>

#define ROOT 0

#define SCI_HIP
//#define SCI_CUDA

#define MPI
//#define MPI_Staged
//#define NCCL
//#define IPC

// USER DEFINED TYPE
struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};

template <typename T>
void copy(const T *sendbuf, const int senddim, T *recvbuf) {
#ifdef SCI_CUDA
  cudaMemcpy(recvbuf, sendbuf, senddim * sizeof(T), cudaMemcpyDeviceToDevice);
#elif defined SCI_HIP
  hipMemcpy(recvbuf, sendbuf, senddim * sizeof(T), hipMemcpyDeviceToDevice);
#else
  memcpy(recvbuf, sendbuf, senddim * sizeof(T));
#endif
}

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
  if (omp_get_thread_num() == 0)
    numthread = omp_get_num_threads();
  long count = atoi(argv[1]);
  int numiter = atoi(argv[2]);
  int groupsize = atoi(argv[3]);
  int numnode = numproc / groupsize;
  int mynode = myid / groupsize;
  // PRINT NUMBER OF PROCESSES AND THREADS
  if (myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of iterations %d\n", numiter);
    printf("Number of proc. per node: %d\n", groupsize);
    printf("Number of nodes: %d\n", numnode);
    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Peer-to-peer count %ld ( %.2e MB)\n", count, count * sizeof(Type) / 1.e6);
    printf("\n");
  }

  Type *sendbuf = new Type[count];
  Type *recvbuf = new Type[count * numnode];
  Type *sendbuf_d;
  Type *recvbuf_d;

  for(int i = 0; i < count; i++) 
    sendbuf[i].data[0] = myid;

  MPI_Comm comm;
  MPI_Comm_split(MPI_COMM_WORLD, myid % groupsize, mynode, &comm);

#ifdef SCI_CUDA
  if(myid == ROOT)
    printf("CUDA VERSION\n");
  // SET DEVICE
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  cudaSetDevice(device);
  // MEMORY MANAGEMENT
  cudaMalloc(&sendbuf_d, count * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * numnode * sizeof(Type));
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
#elif defined SCI_HIP
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
  hipMalloc(&recvbuf_d, count * numnode * sizeof(Type));
  hipMemcpy(sendbuf_d, sendbuf, count * sizeof(Type), hipMemcpyHostToDevice);
  // DONE
  // REPORT
  if(myid == ROOT) {
    system("rocm-smi");
  }
#else
  if(myid == ROOT)
    printf("CPU VERSION\n");
  // MEMORY MANAGEMENT
  sendbuf_d = new Type[count];
  recvbuf_d = new Type[count * numnode];
  memcpy(sendbuf_d, sendbuf, count * sizeof(Type));
  // DONE
#endif

#ifdef MPI
  {
    if(myid == ROOT)
      printf("ENABLE CUDA-Aware MPI\n");
    {
      MPI_Request sendrequest[numnode];
      MPI_Request recvrequest[numnode];
      int sendproc = 0;
      int recvproc = 0;
      double time = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);
      for(int node = 0; node < numnode; node++)
        if(node != mynode) {
          MPI_Irecv(recvbuf_d + node * count, count * sizeof(Type), MPI_BYTE, node, MPI_ANY_TAG, comm, recvrequest + recvproc);
          recvproc++;
        }
      for(int node = 0; node < numnode; node++)
        if(node != mynode) {
          MPI_Isend(sendbuf_d               , count * sizeof(Type), MPI_BYTE, node, 0, comm, sendrequest + sendproc);
          sendproc++;
        }
      MPI_Waitall(recvproc, recvrequest, MPI_STATUSES_IGNORE);
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(myid == ROOT)
        printf("warmup time: %e\n", time);
    }
    double totalTime = 0;
    double totalData = 0;
    for(int iter = 0; iter < numiter; iter++)
    {
#if !defined SCI_CUDA && !defined SCI_HIP
      memset(sendbuf_d, 0, count * sizeof(Type));
      memset(recvbuf_d, 0, numnode * count * sizeof(Type));
#endif
      MPI_Request sendrequest[numnode];
      MPI_Request recvrequest[numnode];
      int sendproc = 0;
      int recvproc = 0;
      double time = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);
      for(int node = 0; node < numnode; node++)
        if(node != mynode) {
          MPI_Irecv(recvbuf_d + node * count, count * sizeof(Type), MPI_BYTE, node, MPI_ANY_TAG, comm, recvrequest + recvproc);
          recvproc++;
        }
      for(int node = 0; node < numnode; node++)
        if(node != mynode) {
          MPI_Isend(sendbuf_d               , count * sizeof(Type), MPI_BYTE, node, 0, comm, sendrequest + sendproc);
          sendproc++;
        }
      MPI_Waitall(recvproc, recvrequest, MPI_STATUSES_IGNORE);
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(myid == ROOT)
        printf("time: %e\n", time);
      totalTime += time;
      totalData += 2 * (numnode - 1) * count * sizeof(Type) / 1.e9;
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- CUDA-Aware MPI\n", totalTime, totalData, totalData / totalTime * groupsize);
  }
#endif

#ifdef MPI_Staged
  { 
    if(myid == ROOT)
      printf("ENABLE CPU-Staged MPI\n");
    Type *sendbuf_h;
    Type *recvbuf_h;
#ifdef SCI_CUDA
    cudaMallocHost(&sendbuf_h, count * numnode * sizeof(Type));
    cudaMallocHost(&recvbuf_h, count * numnode * sizeof(Type));
    cudaStream_t recvstream[numnode];
    for(int node = 0; node < numnode; node++)
      cudaStreamCreate(recvstream + node);
#elif defined SCI_HIP
    hipHostMalloc(&sendbuf_h, count * numnode * sizeof(Type), 0);
    hipHostMalloc(&recvbuf_h, count * numnode * sizeof(Type), 0);
    hipStream_t recvstream[numnode];
    for(int node = 0; node < numnode; node++)
      hipStreamCreate(recvstream + node);
#endif
    {
      MPI_Request sendrequest[numnode];
      MPI_Request recvrequest[numnode];
      bool sendcomplete[numnode];
      bool recvcomplete[numnode];
      memset(sendcomplete, 0, numnode);
      memset(recvcomplete, 0, numnode);
      double time = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);
      for(int node = 0; node < numnode; node++)
        if(node != mynode) {
          MPI_Irecv(recvbuf_h + node * count, count * sizeof(Type), MPI_BYTE, node, MPI_ANY_TAG, comm, recvrequest + node);
#ifdef SCI_CUDA
          cudaMemcpyAsync(sendbuf_h + node * count, sendbuf_d, count * sizeof(Type), cudaMemcpyDeviceToHost, recvstream[node]);
#elif defined SCI_HIP
          hipMemcpyAsync(sendbuf_h + node * count, sendbuf_d, count * sizeof(Type), hipMemcpyDeviceToHost, recvstream[node]);
#endif
        }
      // SEND LOOP
      bool done_send = false;
      while(!done_send) {
        done_send = true;
        for(int node = 0; node < numnode; node++)
          if(node != mynode && !sendcomplete[node]) {
#ifdef SCI_CUDA
            if(cudaStreamQuery(recvstream[node]) == cudaSuccess) {
#elif defined SCI_HIP
            if(hipStreamQuery(recvstream[node]) == hipSuccess) {
#endif
              MPI_Isend(sendbuf_h + node * count, count * sizeof(Type), MPI_BYTE, node, 0, comm, sendrequest + node);
              sendcomplete[node] = true;
            }
            done_send = false;
          }
      }
      // MEMCPY LOOP
      bool done_recv = false;
      while (!done_recv) {
        done_recv = true;
        for (int node = 0; node < numnode; node++)
          if (node != mynode && !recvcomplete[node]) {
            int flag = 0;
            MPI_Test(recvrequest + node, &flag, MPI_STATUS_IGNORE);
            if (flag) {
#ifdef SCI_CUDA
              cudaMemcpyAsync(recvbuf_d + node * count, recvbuf_h + node * count, count * sizeof(Type), cudaMemcpyHostToDevice, recvstream[node]);
#elif defined SCI_HIP
              hipMemcpyAsync(recvbuf_d + node * count, recvbuf_h + node * count, count * sizeof(Type), hipMemcpyHostToDevice, recvstream[node]);
#endif
              recvcomplete[node] = true;
            }
            done_recv = false;
          }
      }
#ifdef SCI_CUDA
      cudaDeviceSynchronize();
#elif defined SCI_HIP
      hipDeviceSynchronize();
#endif
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(myid == ROOT)
        printf("warmup time: %e\n", time);
    }
    double totalTime = 0;
    double totalData = 0;
    for(int iter = 0; iter < numiter; iter++)
    { 
      MPI_Request sendrequest[numnode];
      MPI_Request recvrequest[numnode];
      bool sendcomplete[numnode];
      bool recvcomplete[numnode];
      memset(sendcomplete, 0, numnode);
      memset(recvcomplete, 0, numnode);
      double time = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);
      for(int node = 0; node < numnode; node++)
        if(node != mynode) {
          MPI_Irecv(recvbuf_h + node * count, count * sizeof(Type), MPI_BYTE, node, MPI_ANY_TAG, comm, recvrequest + node);
#ifdef SCI_CUDA
          cudaMemcpyAsync(sendbuf_h + node * count, sendbuf_d, count * sizeof(Type), cudaMemcpyDeviceToHost, recvstream[node]);
#elif defined SCI_HIP
          hipMemcpyAsync(sendbuf_h + node * count, sendbuf_d, count * sizeof(Type), hipMemcpyDeviceToHost, recvstream[node]);
#endif
        }
      // SEND LOOP
      bool done_send = false;
      while(!done_send) {
        done_send = true;
        for(int node = 0; node < numnode; node++)
          if(node != mynode && !sendcomplete[node]) {
#ifdef SCI_CUDA
            if(cudaStreamQuery(recvstream[node]) == cudaSuccess) {
#elif defined SCI_HIP
            if(hipStreamQuery(recvstream[node]) == hipSuccess) {
#endif
              MPI_Isend(sendbuf_h + node * count, count * sizeof(Type), MPI_BYTE, node, 0, comm, sendrequest + node);
              sendcomplete[node] = true;
            }
            done_send = false;
          }
      }
      // MEMCPY LOOP
      bool done_recv = false;
      while (!done_recv) {
        done_recv = true;
        for (int node = 0; node < numnode; node++)
          if (node != mynode && !recvcomplete[node]) {
            int flag = 0;
            MPI_Test(recvrequest + node, &flag, MPI_STATUS_IGNORE);
            if (flag) {
#ifdef SCI_CUDA
              cudaMemcpyAsync(recvbuf_d + node * count, recvbuf_h + node * count, count * sizeof(Type), cudaMemcpyHostToDevice, recvstream[node]);
#elif defined SCI_HIP
              hipMemcpyAsync(recvbuf_d + node * count, recvbuf_h + node * count, count * sizeof(Type), hipMemcpyHostToDevice, recvstream[node]);
#endif
              recvcomplete[node] = true;
            }
            done_recv = false;
          }
      }
#ifdef SCI_CUDA
      cudaDeviceSynchronize();
#elif defined SCI_HIP
      hipDeviceSynchronize();
#endif
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(myid == ROOT)
        printf("time: %e\n", time);
      totalTime += time;
      totalData += 2 * (numnode - 1) * count * sizeof(Type) / 1.e9;
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- CPU-Staged MPI\n", totalTime, totalData, totalData / totalTime * groupsize);

    for(int node = 0; node < numnode; node++)
#ifdef SCI_CUDA
      cudaStreamDestroy(recvstream[node]);
#elif defined SCI_HIP
      hipStreamDestroy(recvstream[node]);
#endif
  }
#endif

#ifdef IPC
  {
    cudaMemset(recvbuf_d, 0, numnode * count * sizeof(Type));
    if(myid == ROOT)
      printf("ENABLE CUDA IPC\n");

    Type *recvbuf_p[numnode];
    cudaStream_t stream_ipc[numnode];
    for(int node = 0; node < numnode; node++)
      cudaStreamCreate(stream_ipc + node);
    {
      cudaIpcMemHandle_t peerhandle[numnode];
      for(int node = 0; node < numnode; node++) {
        cudaIpcMemHandle_t myhandle;
        Type *temp = recvbuf_d + node * count;
        cudaIpcGetMemHandle(&myhandle, temp);
        MPI_Gather(&myhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, peerhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, node, comm);
      }
      for(int node = 0; node < numnode; node++)
        cudaIpcOpenMemHandle((void**)(recvbuf_p + node), peerhandle[node], cudaIpcMemLazyEnablePeerAccess);
    }
    {
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      for(int node = 0; node < numnode; node++)
        if(mynode != node)
          cudaMemcpyAsync(recvbuf_p[node] + mynode * count, sendbuf_d, count * sizeof(Type), cudaMemcpyDeviceToDevice, stream_ipc[node]);
      //cudaMemcpy(recvbuf_d + mynode * count, sendbuf_d, count * sizeof(Type), cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(myid == ROOT)
        printf("warmup time %e\n", time);
    }
    double totalData = 0;
    double totalTime = 0;
    for(int iter = 0; iter < numiter; iter++)
    {
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      for(int node = 0; node < numnode; node++)
        if(mynode != node)
          cudaMemcpyAsync(recvbuf_p[node] + mynode * count, sendbuf_d, count * sizeof(Type), cudaMemcpyDeviceToDevice, stream_ipc[node]);
      //cudaMemcpy(recvbuf_d + mynode * count, sendbuf_d, count * sizeof(Type), cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(myid == ROOT)
        printf("time %e\n", time);
      totalTime += time;
      totalData += 2 * (numnode - 1) * count * sizeof(Type) / 1.e9;
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- CUDA_IPC\n", totalTime, totalData, totalData / totalTime * groupsize);

    for(int node = 0; node < numnode; node++)
      cudaStreamDestroy(stream_ipc[node]);
  }
#endif

#ifdef NCCL
  {
    if(myid == ROOT)
      printf("ENABLE NCCL\n");
    ncclComm_t comm_nccl;
    ncclUniqueId id;
    if (myid / groupsize == 0)
      ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
    ncclCommInitRank(&comm_nccl, numnode, id, myid / groupsize);
    {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      ncclGroupStart();
      for(int node = 0; node < numnode; node++)
        if(myid / groupsize != node) {
            ncclSend(sendbuf_d,                count * sizeof(Type), ncclInt8, node, comm_nccl, 0);
            ncclRecv(recvbuf_d + node * count, count * sizeof(Type), ncclInt8, node, comm_nccl, 0);
        }
      ncclGroupEnd();
#ifdef SCI_CUDA
      cudaDeviceSynchronize();
#elif defined SCI_HIP
      hipDeviceSynchronize();
#endif
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(myid == ROOT)
        printf("warmup time %e\n", time);
    }
    double totalTime = 0;
    double totalData = 0;
    for(int iter = 0; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      ncclGroupStart();
      for(int node = 0; node < numnode; node++)
        if(myid / groupsize != node) {
            ncclSend(sendbuf_d,                count * sizeof(Type), ncclInt8, node, comm_nccl, 0);
            ncclRecv(recvbuf_d + node * count, count * sizeof(Type), ncclInt8, node, comm_nccl, 0);
        }
      ncclGroupEnd();
#ifdef SCI_CUDA
      cudaDeviceSynchronize();
#elif defined SCI_HIP
      hipDeviceSynchronize();
#endif
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(myid == ROOT)
        printf("time %e\n", time);
      totalTime += time;
      totalData += 2 * (numnode - 1) * count * sizeof(Type) / 1.e9;
    }
    if(myid == ROOT)
      printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- NCCL\n", totalTime, totalData, totalData / totalTime * groupsize);

    ncclCommDestroy(comm_nccl);
  }
#endif

// RELEASE GPU POINTERS
#ifdef SCI_CUDA
  cudaFree(sendbuf_d);
  cudaFree(recvbuf_d);
#elif defined SCI_HIP
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

