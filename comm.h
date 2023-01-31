
namespace CommBench
{
  enum capability {MPI, MPI_staged, NCCL, IPC, self};

  template <typename T>
  class Comm {

    const capability cap;
    const MPI_Comm &comm;

    // GPU-Aware MPI
    MPI_Request *sendrequest;
    MPI_Request *recvrequest;

    // CPU-Staged MPI
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

#ifdef CAP_NCCL
    ncclComm_t comm_nccl;
#endif
#ifdef PORT_CUDA
    cudaStream_t stream_nccl;
#elif defined PORT_HIP
    hipStream_t stream_nccl;
#endif

    // IPC
    T **recvbuf_ipc;
#ifdef PORT_CUDA
    cudaStream_t *stream_ipc;
#elif defined PORT_HIP
    hipStream_t *stream_ipc;
#endif
    size_t *recvoffset_ipc;

    // self
#ifdef PORT_CUDA
    cudaStream_t *stream_self;
#elif defined PORT_HIP
    hipStream_t *stream_self;
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
         const MPI_Comm &comm, const capability cap);

    void start();
    void wait();
  };

  template <typename T>
  Comm<T>::Comm(T *&sendbuf, size_t sendcount[], size_t sendoffset[], T *&recvbuf, size_t recvcount[], size_t recvoffset[], const MPI_Comm &comm, const capability cap) : comm(comm), cap(cap), sendbuf(sendbuf), recvbuf(recvbuf) {

      int myid_root;
      MPI_Comm_rank(MPI_COMM_WORLD, &myid_root);
      if(myid_root == ROOT)
        printf("Creating a Comm object requires global synchronization\n");

      int myid;
      int numproc;
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &numproc);
      if(myid_root == ROOT)
        for(int p = 0; p < numproc; p++)
          printf("myid %d send %d recv %d\n", myid, sendcount[p], recvcount[p]);
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
        case self:
          if(myid_root == ROOT)
            printf("SETUP self\n");
#ifdef PORT_CUDA
          if(myid_root == ROOT)
            printf("FOR CUDA\n");
          stream_self = new cudaStream_t[numsend];
          for(int send = 0; send < numsend; send++)
            cudaStreamCreate(stream_self + send);
#elif defined PORT_HIP
          if(myid_root == ROOT)
            printf("FOR HIP\n");
          stream_self = new hipStream_t[numsend];
          for(int send = 0; send < numsend; send++)
            hipStreamCreate(stream_self + send);
#else
          if(myid_root == ROOT)
            printf("FOR CPU\n");
#endif
          break;
        case MPI:
          if(myid_root == ROOT)
            printf("SETUP GPU-AWARE MPI\n");
          sendrequest = new MPI_Request[numsend];
          recvrequest = new MPI_Request[numrecv];
#ifdef PORT_CUDA
          if(myid_root == ROOT)
            printf("FOR CUDA\n");
#elif defined PORT_HIP
          if(myid_root == ROOT)
            printf("FOR HIP\n");
#else
          if(myid_root == ROOT)
            printf("FOR CPU\n");
#endif
          break;
        case MPI_staged:
          if(myid_root == ROOT)
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
            if(myid_root == ROOT)
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
            if(myid_root == ROOT)
              printf("FOR HIP\n");
            sendstream = new hipStream_t[numsend];
            for(int send = 0; send < numsend; send++)
              hipStreamCreate(sendstream + send);
            recvstream = new hipStream_t[numrecv];
            for(int recv = 0; recv < numrecv; recv++)
              hipStreamCreate(recvstream + recv);
            hipHostMalloc(&sendbuf_h, sendcount_h * sizeof(T));
            hipHostMalloc(&recvbuf_h, recvcount_h * sizeof(T));
#endif
          }
          break;
        case NCCL:
          if(myid_root == ROOT)
            printf("SETUP NCCL\n");
#ifdef CAP_NCCL
          ncclUniqueId id;
          if(myid == 0)
            ncclGetUniqueId(&id);
          MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
          ncclCommInitRank(&comm_nccl, numproc, id, myid);
#ifdef PORT_CUDA
          if(myid_root == ROOT)
            printf("FOR CUDA\n");
          cudaStreamCreate(&stream_nccl);
#elif defined PORT_HIP
          if(myid_root == ROOT)
            printf("FOR HIP\n");
          hipStreamCreate(&stream_nccl);
#endif
#endif
          break;
        case IPC:
          if(myid_root == ROOT)
            printf("SETUP IPC\n");
          recvbuf_ipc = new T*[numsend];
          recvoffset_ipc = new size_t[numsend];
#ifdef PORT_CUDA
          if(myid_root == ROOT)
            printf("FOR CUDA\n");
          {
            cudaIpcMemHandle_t handle_temp[numproc];
            for(int recv = 0; recv < numrecv; recv++) {
              cudaIpcMemHandle_t myhandle;
              cudaIpcGetMemHandle(&myhandle, recvbuf);
              handle_temp[recvproc[recv]] = myhandle;
            }
            cudaIpcMemHandle_t handle_ipc[numproc];
            MPI_Alltoall(handle_temp, sizeof(cudaIpcMemHandle_t), MPI_BYTE, handle_ipc, sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm);
            size_t recvoffset_temp[numproc];
            MPI_Alltoall(recvoffset, 1, MPI_UNSIGNED_LONG, recvoffset_temp, 1, MPI_UNSIGNED_LONG, comm);
            for(int send = 0; send < numsend; send++) {
              recvoffset_ipc[send] = recvoffset_temp[sendproc[send]];
              cudaIpcOpenMemHandle((void**)(recvbuf_ipc + send), handle_ipc[sendproc[send]], cudaIpcMemLazyEnablePeerAccess);
            }
          }
          stream_ipc = new cudaStream_t[numsend];
          for(int send = 0; send < numsend; send++)
            cudaStreamCreate(stream_ipc + send);
#elif defined PORT_HIP
          if(myid_root == ROOT)
            printf("FOR HIP\n");
          {
            hipIpcMemHandle_t handle_temp[numproc];
            for(int recv = 0; recv < numrecv; recv++) {
              hipIpcMemHandle_t myhandle;
              hipIpcGetMemHandle(&myhandle, recvbuf);
              handle_temp[recvproc[recv]] = myhandle;
            }
            hipIpcMemHandle_t handle_ipc[numproc];
            MPI_Alltoall(handle_temp, sizeof(hipIpcMemHandle_t), MPI_BYTE, handle_ipc, sizeof(hipIpcMemHandle_t), MPI_BYTE, comm);
            size_t recvoffset_temp[numproc];
            MPI_Alltoall(recvoffset, 1, MPI_UNSIGNED_LONG, recvoffset_temp, 1, MPI_UNSIGNED_LONG, comm);
            for(int send = 0; send < numsend; send++) {
              recvoffset_ipc[send] = recvoffset_temp[sendproc[send]];
              hipIpcOpenMemHandle((void**)(recvbuf_ipc + send), handle_ipc[sendproc[send]], hipIpcMemLazyEnablePeerAccess);
            }
          }
          stream_ipc = new hipStream_t[numsend];
          for(int send = 0; send < numsend; send++)
            hipStreamCreate(stream_ipc + send);
#endif
          break;
        default:
          printf("Selected capability is not yet implemented for CommBench::Comm.\n");
      } // switch(cap)
    if(myid_root == ROOT)
      printf("\n");
  } // Comm::Comm

  template <typename T>
  void Comm<T>::start() {
    switch(cap) {
      // SELF IMPLEMENTATION
      case self:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaMemcpyAsync(recvbuf + recvoffset[send], sendbuf + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToDevice, stream_self[send]);
#elif defined PORT_HIP
          hipMemcpyAsync(recvbuf + recvoffset[send], sendbuf + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToDevice, stream_self[send]);
#else
          memcpy(recvbuf + recvoffset[send], sendbuf + sendoffset[send], sendcount[send] * sizeof(T));
#endif
        }
        break;
      // MPI IMPLEMENTATION
      case MPI:
        for (int send = 0; send < numsend; send++)
          MPI_Isend(sendbuf + sendoffset[send], sendcount[send] * sizeof(T), MPI_BYTE, sendproc[send], 0, comm, sendrequest + send);         
        for (int recv = 0; recv < numrecv; recv++)
          MPI_Irecv(recvbuf + recvoffset[recv], recvcount[recv] * sizeof(T), MPI_BYTE, recvproc[recv], 0, comm, recvrequest + recv);         
        break;
      // CPU-Staged MPI IMPLEMENTATION
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
                if(cudaStreamQuery(sendstream[send]) == cudaSuccess)
#elif defined PORT_HIP
                if(hipStreamQuery(sendstream[send]) == hipSuccess)
#endif
                {
                  MPI_Isend(sendbuf_h + sendoffset_h[send], sendcount[send] * sizeof(T), MPI_BYTE, sendproc[send], 0, comm, sendrequest_h + send);
                  sendcomplete[send] = true;
                }
                done_send = false;
              }
          }
        }
        break;
      // NCCL IMPLEMENTATION
      case NCCL:
#ifdef CAP_NCCL
        ncclGroupStart(); 
        for(int send = 0; send < numsend; send++)
          ncclSend(sendbuf + sendoffset[send], sendcount[send] * sizeof(T), ncclInt8, sendproc[send], comm_nccl, stream_nccl);
        for(int recv = 0; recv < numrecv; recv++)
          ncclRecv(recvbuf + recvoffset[recv], recvcount[recv] * sizeof(T), ncclInt8, recvproc[recv], comm_nccl, stream_nccl);
        ncclGroupEnd();
        break;
#endif
      // IPC IMPLEMENTATION
      case IPC:
        int myid;
        int numproc;
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &numproc);

        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaMemcpyAsync(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToDevice, stream_ipc[send]);
#elif PORT_HIP
          hipMemcpyAsync(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToDevice, stream_ipc[send]);
#endif
        }
        break;
      default:
        printf("Selected capability is not yet implemented for CommBench::Comm.init.\n");
    }
  } // Comm::start


  template <typename T>
  void Comm<T>::wait() { 
    switch(cap) {
      case self:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_self[send]);
#elif defined PORT_HIP
          hipStreamSynchronize(stream_self[send]);
#endif
        }
        break;
      case MPI:
        MPI_Waitall(numrecv, recvrequest, MPI_STATUSES_IGNORE);
        MPI_Waitall(numsend, sendrequest, MPI_STATUSES_IGNORE);
        break;
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
        for(int recv = 0; recv < numrecv; recv++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(recvstream[recv]);
#elif defined PORT_HIP
          hipStreamSynchronize(recvstream[recv]);
#endif
        }
        MPI_Waitall(numsend, sendrequest, MPI_STATUSES_IGNORE);
        break;
      case NCCL:
#ifdef PORT_CUDA
        cudaStreamSynchronize(stream_nccl);
#elif defined(PORT_HIP)
        hipStreamSynchronize(stream_nccl);
#endif
        break;
      case IPC:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_ipc[send]);
#elif defined(PORT_HIP)
          hipStreamSynchronize(stream_ipc[send]);
#endif
        }
        MPI_Barrier(comm);
        break;
      default:
        printf("Selected capability is not yet implemented for CommBench::Comm.wait.\n");
    }
  } // Comm::wait()

} // namespace CommBench
