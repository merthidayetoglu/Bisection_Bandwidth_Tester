

namespace CommBench
{
  enum transport {MPI, MPI_staged, NCCL, IPC, memcpy};

  template <typename T>
  class Comm {

    const transport cap;
    const MPI_Comm &comm;
#ifdef CAP_MPI 
    MPI_Request *sendrequest;
    MPI_Request *recvrequest;
#endif
#ifdef CAP_MPI_staged
    T *sendbuf_h;
    T *recvbuf_h;
    MPI_Request *sendrequest_h;
    MPI_Request *recvrequest_h;
    bool *sendcomplete;
    bool *recvcomplete;
#ifdef PORT_CUDA
    cudaStream_t *sendstream;
    cudaStream_t *recvstream;
#elif defined PORT_HIP
    hipStream_t *sendstream;
    hipStream_t *recvstream;
#endif
    size_t *recvoffset_h;
    size_t *sendoffset_h;
#endif
#ifdef CAP_NCCL
    ncclComm_t comm_nccl;
#ifdef PORT_CUDA
    cudaStream_t stream_nccl;
#elif defined PORT_HIP
    hipStream_t stream_nccl;
#endif
#endif
#ifdef CAP_IPC
    T **recvbuf_ipc;
#ifdef PORT_CUDA
    cudaStream_t *stream_ipc;
#elif defined PORT_HIP
    hipStream_t *stream_ipc;
#endif
    size_t *recvoffset_ipc;
#endif

    T *&sendbuf;
    T *&recvbuf;
    int numsend;
    int numrecv;
    int *sendproc;
    int *recvproc;
    size_t *sendcount;
    size_t *recvcount;
    size_t *sendoffset;
    size_t *recvoffset;

    public:

    Comm(T* &sendbuf, size_t sendcount[], size_t sendoffset[],
         T* &recvbuf, size_t recvcount[], size_t recvoffset[],
         const MPI_Comm &comm, const transport cap);

    void start();
    void wait();
  };

  template <typename T>
  Comm<T>::Comm(T *&sendbuf, size_t sendcount[], size_t sendoffset[], T *&recvbuf, size_t recvcount[], size_t recvoffset[], const MPI_Comm &comm, const transport cap) : comm(comm), cap(cap), sendbuf(sendbuf), recvbuf(recvbuf) {
      int myid;
      int numproc;
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &numproc);
      // SETUP COMM INFO
      numsend = 0;
      numrecv = 0;
      for(int p = 0; p < numproc; p++) {
        if(sendcount[p]) numsend++;
        if(recvcount[p]) numrecv++;
      }
      sendproc = new int[numsend];
      recvproc = new int[numrecv];
      this->sendcount = new size_t[numsend];
      this->recvcount = new size_t[numrecv];
      this->sendoffset = new size_t[numsend];
      this->recvoffset = new size_t[numrecv];
      numsend = 0;
      numrecv = 0;
      for(int p = 0; p < numproc; p++) {
        if(sendcount[p]) {
          sendproc[numsend] = p;
          this->sendcount[numsend] = sendcount[p];
          this->sendoffset[numsend] = sendoffset[p];
          numsend++;
        }
        if(recvcount[p]) {
          recvproc[numrecv] = p;
          this->recvcount[numrecv] = recvcount[p];
          this->recvoffset[numrecv] = recvoffset[p];
          numrecv++;
        }
      }
      // SETUP CAPABILITY
      switch(cap) {
#ifdef CAP_MPI
        case MPI:
          if(myid == ROOT)
            printf("SETUP GPU-AWARE MPI\n");
          sendrequest = new MPI_Request[numsend];
          recvrequest = new MPI_Request[numrecv];
#ifdef PORT_CUDA
          if(myid == ROOT)
            printf("FOR CUDA\n");
#elif defined(PORT_HIP)
          if(myid == ROOT)
            printf("FOR HIP\n");
#elif
          if(myid == ROOT)
            printf("FOR CPU\n");
#endif
          break;
#endif
#ifdef CAP_MPI_staged
        case MPI_staged:
          if(myid == ROOT)
            printf("SETUP CPU-Staged MPI\n");
          sendrequest_h = new MPI_Request[numsend];
          recvrequest_h = new MPI_Request[numrecv];
          sendoffset_h = new size_t[numsend];
          recvoffset_h = new size_t[numrecv];
          {
            size_t sendcount_h = 0;
            for(int send = 0; send < numsend; send++)
              sendcount_h += sendcount[send];
            sendoffset_h[0] = 0;
            for(int send = 1; send < numsend; send++)
              sendoffset_h[send] = sendcount[send];
            size_t recvcount_h;
            for(int recv = 0; recv < numrecv; recv++)
              recvcount_h += recvcount[recv];
            recvoffset_h[0] = 0;
            for(int recv = 1; recv < numrecv; recv++)
              recvoffset_h[recv] = recvcount[recv];
#ifdef PORT_CUDA
            if(myid == ROOT)
              printf("FOR CUDA\n");
            sendstream = new cudaStream_t[numsend];
            for(int send = 0; send < numsend; send++)
              cudaStreamCreate(sendstream + send);
            recvstream = new cudaStream_t[numrecv];
            for(int recv = 0; recv < numrecv; recv++)
              cudaStreamCreate(recvstream + recv);
            cudaMallocHost(&sendbuf_h, sendcount_h * sizeof(T));
            cudaMallocHost(&recvbuf_h, recvcount_h * sizeof(T));
#elif defined PORT_HIP
            if(myid == ROOT)
              printf("FOR HIP\n");
            sendstream = new cudaStream_t[numsend];
            for(int send = 0; send < numsend; send++)
              hipStreamCreate(sendstream + send);
            recvstream = new cudaStream_t[numrecv];
            for(int recv = 0; recv < numrecv; recv++)
              hipStreamCreate(recvstream + recv);
            hipHostMalloc(&sendbuf_h, sendcount_h * sizeof(T));
            hipHostMalloc(&recvbuf_h, recvcount_h * sizeof(T));
#endif
          }
          break;
#endif
#ifdef CAP_NCCL
        case NCCL:
          if(myid == ROOT)
            printf("SETUP NCCL\n");
          ncclUniqueId id;
          if(myid == 0)
            ncclGetUniqueId(&id);
          MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
          ncclCommInitRank(&comm_nccl, numproc, id, myid);
#ifdef PORT_CUDA
          if(myid == ROOT)
            printf("FOR CUDA\n");
          cudaStreamCreate(&stream_nccl);
#elif defined PORT_HIP
          if(myid == ROOT)
            printf("FOR HIP\n");
          hipStreamCreate(&stream_nccl);
#endif
          break;
#endif
#ifdef CAP_IPC
        case IPC:
          if(myid == ROOT)
            printf("SETUP IPC\n");
          recvbuf_ipc = new T*[numsend];
          recvoffset_ipc = new size_t[numsend];
#ifdef PORT_CUDA
          if(myid == ROOT)
            printf("FOR CUDA\n");
          {
            cudaIpcMemHandle_t handle_ipc[numproc];
            cudaIpcMemHandle_t myhandle;
            cudaIpcGetMemHandle(&myhandle, recvbuf);
            MPI_Allgather(&myhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, handle_ipc, sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm);
            size_t recvoffset_ipc[numproc];
            MPI_Alltoall(recvoffset, 1, MPI_UNSIGNED_LONG, recvoffset_ipc, 1, MPI_UNSIGNED_LONG, comm);
            for(int send = 0; send < numsend; send++) {
              this->recvoffset_ipc[send] = recvoffset_ipc[sendproc[send]];
              cudaIpcOpenMemHandle((void**)(recvbuf_ipc + send), handle_ipc[sendproc[send]], cudaIpcMemLazyEnablePeerAccess);
            }
          }
          stream_ipc = new cudaStream_t[numsend];
          for(int send = 0; send < numsend; send++)
            cudaStreamCreate(stream_ipc + send);
#elif defined PORT_HIP
          if(myid == ROOT)
            printf("FOR HIP\n");
          {
            hipIpcMemHandle_t handle_ipc[numproc];
            hipIpcMemHandle_t myhandle;
            hipIpcGetMemHandle(&myhandle, recvbuf);
            MPI_Allgather(&myhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, handle_ipc, sizeof(hipIpcMemHandle_t), MPI_BYTE, comm);
            size_t recvoffset_ipc[numproc];
            MPI_Alltoall(recvoffset, 1, MPI_UNSIGNED_LONG, recvoffset_ipc, 1, MPI_UNSIGNED_LONG, comm);
            for(int send = 0; send < numsend; send++) {
              this->recvoffset_ipc[send] = recvoffset_ipc[sendproc[send]];
              hipIpcOpenMemHandle((void**)(recvbuf_ipc + send), handle_ipc[sendproc[send]], hipIpcMemLazyEnablePeerAccess);
            }
          }
          stream_ipc = new hipStream_t[numsend];
          for(int send = 0; send < numsend; send++)
            hipStreamCreate(stream_ipc + send);
#endif
          break;
#endif
        default:
          printf("Selected capability is not yet implemented for CommBench::Comm.\n");
      } // switch(cap)
  } // Comm::Comm

  template <typename T>
  void Comm<T>::start() {
    switch(cap) {
#ifdef CAP_MPI
      case MPI:
        for (int send = 0; send < numsend; send++)
          MPI_Isend(sendbuf + sendoffset[send], sendcount[send] * sizeof(T), MPI_BYTE, sendproc[send], 0, comm, sendrequest + send);         
        for (int recv = 0; recv < numrecv; recv++)
          MPI_Irecv(recvbuf + recvoffset[recv], recvcount[recv] * sizeof(T), MPI_BYTE, recvproc[recv], 0, comm, recvrequest + recv);         
        break;
#endif
#ifdef CAP_MPI_staged
      case MPI_staged:
        for (int recv = 0; recv < numrecv; recv++)
          MPI_Irecv(recvbuf_h + recvoffset_h[recv], recvcount[recv] * sizeof(T), MPI_BYTE, recvproc[recv], 0, comm, recvrequest_h + recv);
        for (int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaMemcpyAsync(sendbuf_h + sendoffset_h[send], sendbuf + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToHost, sendstream[send]);
#elif defined PORT_HIP
          hipMemcpyAsync(sendbuf_h + sendoffset_h[send], sendbuf + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToHost, sendstream[send]);
#endif
        }
        // SEND LOOP
        {
          bool done_send = false;
          memset(sendcomplete, 0, numsend);
          while(!done_send) {
            done_send = true;
            for(int send = 0; send < numsend; send++)
              if(!sendcomplete[send]) {
#ifdef PORT_CUDA
                if(cudaStreamQuery(sendstream[send]) == cudaSuccess) {
#elif defined PORT_HIP
                if(hipStreamQuery(sendstream[send]) == hipSuccess) {
#endif
                  MPI_Isend(sendbuf_h + sendoffset_h[send], sendcount[send] * sizeof(T), MPI_BYTE, sendproc[send], 0, comm, sendrequest_h + send);
                  sendcomplete[send] = true;
                }
                done_send = false;
              }
          }
        }
        break;
#endif
#ifdef CAP_NCCL
      case NCCL:
        ncclGroupStart(); 
        for(int send = 0; send < numsend; send++)
          ncclSend(sendbuf + sendoffset[send], sendcount[send] * sizeof(T), ncclInt8, sendproc[send], comm_nccl, stream_nccl);
        for(int recv = 0; recv < numrecv; recv++)
          ncclRecv(recvbuf + recvoffset[recv], recvcount[recv] * sizeof(T), ncclInt8, recvproc[recv], comm_nccl, stream_nccl);
        ncclGroupEnd();
        break;
#endif
#ifdef CAP_IPC
      case IPC: 
        for(int send = 0; send < numsend; send++)
#ifdef PORT_CUDA
          cudaMemcpyAsync(sendbuf + sendoffset[send], recvbuf_ipc[send] + recvoffset_ipc[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToDevice, stream_ipc[send]);
#elif PORT_HIP
          hipMemcpyAsync(sendbuf + sendoffset[send], recvbuf_ipc[send] + recvoffset_ipc[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToDevice, stream_ipc[send]);
#endif
        break;
#endif
      default:
        printf("Selected capability is not yet implemented for CommBench::init.\n");
    }
  }


  template <typename T>
  void Comm<T>::wait() { 
    switch(cap) {
#ifdef CAP_MPI
      case MPI:
        MPI_Waitall(numsend, sendrequest, MPI_STATUSES_IGNORE);
        MPI_Waitall(numrecv, recvrequest, MPI_STATUSES_IGNORE);
        break;
#endif
#ifdef CAP_MPI_staged
      case MPI_staged:
        // MEMCPY LOOP
        {
          bool done_recv = false;
          memset(recvcomplete, 0, numrecv);
          while(!done_recv) {
            done_recv = true;
            for(int recv = 0; recv < numrecv; recv++)
              if(!recvcomplete[recv]) {
                int flag = 0;
                MPI_Test(recvrequest_h + recv, &flag, MPI_STATUS_IGNORE);
                if(flag) {
#ifdef PORT_CUDA
                  cudaMemcpyAsync(recvbuf + recvoffset[recv], recvbuf_h + recvoffset_h[recv], recvcount[recv] * sizeof(T), cudaMemcpyHostToDevice, recvstream[recv]);
#elif defined PORT_HIP
                  hipMemcpyAsync(recvbuf + recvoffset[recv], recvbuf_h + recvoffset_h[recv], recvcount[recv] * sizeof(T), hipMemcpyHostToDevice, recvstream[recv]);
#endif
                  recvcomplete[recv] = true;
                }
                done_recv = false;
              }
          }
        }
        for(int recv = 0; recv < numrecv; recv++)
#ifdef PORT_CUDA
          cudaStreamSynchronize(recvstream[recv]);
#elif defined PORT_HIP
          hipStreamSynchronize(recvstream[recv]);
#endif
      break;
#endif
#ifdef CAP_NCCL
      case NCCL:
#ifdef PORT_CUDA
        cudaStreamSynchronize(stream_nccl);
#elif defined(PORT_HIP)
        hipStreamSynchronize(stream_nccl);
#endif
        break;
#endif
#ifdef CAP_IPC
      case IPC:
        for(int send = 0; send < numsend; send++)
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_ipc[send]);
#elif defined(PORT_HIP)
          hipStreamSynchronize(stream_ipc[send]);
#endif
        break;
#endif
      default:
        printf("NOT IMPLEMENTED\n");
    }
  }

  template <typename T>
  class Bench
  {
    int myid;
    int numproc;
    int mygroup;
    int numgroup;
    const int groupsize;
    int mysubgroup;
    int numsubgroup;
    const int subgroupsize;

    const MPI_Comm &comm_mpi;
    MPI_Comm comm;
    MPI_Comm comm_temp;
    MPI_Comm comm_local;

    size_t count;
    size_t count_local;

    T *sendbuf_d;
    T *recvbuf_d;
    T *recvbuf_d_local;

    T *sendbuf_h;
    T *recvbuf_h;
    T *recvbuf_h_local;

    public:

    void init(size_t count, transport cap_global, transport cap_local, double ratio);

    void measure(int numiter);

    Bench(const int groupsize, const int subgroupsize, const MPI_Comm &comm_mpi) : groupsize(groupsize), subgroupsize(subgroupsize), comm_mpi(comm_mpi) {

      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      mygroup = myid / groupsize;
      numgroup = numproc / groupsize;
      numsubgroup = groupsize / subgroupsize;
      mysubgroup = (myid % groupsize) / subgroupsize;

      int numthread;
      #pragma omp parallel
      if(omp_get_thread_num() == 0)
        numthread = omp_get_num_threads();

      // PRINT NUMBER OF PROCESSES AND THREADS
      if(myid == ROOT)
      {
        printf("\n");
        printf("Number of processes: %d\n", numproc);
        printf("Number of threads per proc: %d\n", numthread);
        printf("Number of proc. per group: %d\n", groupsize);
        printf("Number of proc. per subgroup: %d\n", subgroupsize);
        printf("Number of subgroups: %d\n", numsubgroup);
        printf("Number of groups: %d\n", numgroup);
        printf("Bytes per Type %lu\n", sizeof(T));
        printf("\n");
      }

      // PARTITION
      MPI_Comm_split(comm_mpi, myid % groupsize, mygroup, &comm);
      MPI_Comm_split(comm_mpi, mygroup, myid % groupsize, &comm_temp);
      MPI_Comm_split(comm_temp, (myid % groupsize) % subgroupsize, mysubgroup, &comm_local);

      size_t sendDim = 1;
      size_t recvDim_global = numgroup;
      size_t recvDim_local = numsubgroup;
      size_t sendDim_total = sendDim;
      size_t recvDim_total_global = recvDim_global;
      size_t recvDim_total_local = recvDim_local;
      MPI_Allreduce(MPI_IN_PLACE, &sendDim_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_mpi);
      MPI_Allreduce(MPI_IN_PLACE, &recvDim_total_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_mpi);
      MPI_Allreduce(MPI_IN_PLACE, &recvDim_total_local, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_mpi);
      if(myid == ROOT) {
        printf("sendDim %lu (%.2f) recvDim_global %lu (%.2f) recvDim_local %lu (%.2f)\n", sendDim_total, sendDim_total/(double)numproc, recvDim_total_global, recvDim_total_global/(double)numproc, recvDim_total_local, recvDim_total_local/(double)numproc);
        printf("Now initialize bench with comm.init\n");
        printf("\n");
      }
    }
  };

  template<typename T>
  void Bench<T>::init(size_t count, transport cap_global, transport cap_local, double ratio) {

    this->count = count;
    count_local = count * ratio;

    if(myid == ROOT)
      printf("count: %lu ratio: %f count_local: %lu\n", count, ratio, count_local);
#ifdef PORT_CUDA
    if(myid == ROOT)
      printf("CUDA VERSION\n");
    // SET DEVICE
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device = myid % deviceCount;
    cudaSetDevice(device);
    // MEMORY MANAGEMENT
    cudaMalloc(&sendbuf_d, count * sizeof(T));
    cudaMalloc(&recvbuf_d, count * numgroup * sizeof(T));
    cudaMalloc(&recvbuf_d_local, count_local * numsubgroup * sizeof(T));
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
    hipMalloc(&sendbuf_d, count * sizeof(T));
    hipMalloc(&recvbuf_d, count * numgroup * sizeof(T));
    hipMalloc(&recvbuf_d_local, count_local * numsubgroup * sizeof(T));
    // DONE
    // REPORT
    if(myid == ROOT)
      system("rocm-smi");
#else
    if(myid == ROOT)
      printf("CPU VERSION\n");
    // MEMORY MANAGEMENT
    sendbuf_d = new T[count];
    recvbuf_d = new T[count * numgroup];
    recvbuf_d_local = new T[count_local * numsubgroup];
    // DONE
#endif

 
    int sendoffset[numproc] = {0}; 
    //Comm alltoall(sendbuf_d, );

  }; // void init

  template <typename T>
  void Bench<T>::measure(int numiter){


#ifdef CAP_MPI
    {
      if(myid == ROOT)
        printf("ENABLE GPU-Aware MPI\n");
      double totalTime = 0;
      double totalData = 0;
      for(int iter = 0; iter < numiter; iter++)
      {
#if !defined PORT_CUDA && !defined PORT_HIP 
        memset(sendbuf_d, 0, count * sizeof(T));
        memset(recvbuf_d, 0, count * numgroup * sizeof(T));
        memset(recvbuf_d_local, 0, numsubgroup * count_local * sizeof(T));
#endif
        MPI_Request sendrequest[numgroup];
        MPI_Request recvrequest[numgroup];
        int sendproc = 0;
        int recvproc = 0;

        MPI_Request sendrequest_local[numsubgroup];
        MPI_Request recvrequest_local[numsubgroup];
        int sendproc_local = 0;
        int recvproc_local = 0;

        MPI_Barrier(comm_mpi);
        double time = MPI_Wtime();
        for(int group = 0; group < numgroup; group++)
          if(group != mygroup) {
            MPI_Irecv(recvbuf_d + group * count, count * sizeof(T), MPI_BYTE, group, MPI_ANY_TAG, comm, recvrequest + recvproc);
            MPI_Isend(sendbuf_d                , count * sizeof(T), MPI_BYTE, group, 0          , comm, sendrequest + sendproc);
            recvproc++;
            sendproc++;
          }
        for(int subgroup = 0; subgroup < numsubgroup; subgroup++)
          if(subgroup != mysubgroup) {
            MPI_Irecv(recvbuf_d_local + subgroup * count_local, count_local * sizeof(T), MPI_BYTE, subgroup, MPI_ANY_TAG, comm_local, recvrequest_local + recvproc_local);
            MPI_Isend(sendbuf_d                               , count_local * sizeof(T), MPI_BYTE, subgroup, 0          , comm_local, sendrequest_local + sendproc_local);
            sendproc_local++;
            recvproc_local++;
          }
        MPI_Waitall(recvproc_local, recvrequest_local, MPI_STATUSES_IGNORE);
        MPI_Waitall(recvproc, recvrequest, MPI_STATUSES_IGNORE);
        MPI_Barrier(comm_mpi);
        time = MPI_Wtime() - time;
        if(iter == 0)
        { 
          if(myid == ROOT)
            printf("warmup time: %e\n", time);
        }
        else
        { 
          if(myid == ROOT)
            printf("time: %e\n", time);
          totalTime += time;
          totalData += 2 * (numgroup - 1) * count * sizeof(T) / 1.e9;
        }
      }
      if(myid == ROOT)
        printf("totalTime: %e totalData: %.2e GB (%e GB/s) --- GPU-Aware MPI\n", totalTime, totalData, totalData / totalTime * groupsize);
    }
#endif

  };

} // namespace CommBench
