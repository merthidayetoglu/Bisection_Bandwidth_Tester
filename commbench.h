

namespace CommBench
{
  enum transport {MPI, MPI_staged, NCCL, IPC};

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
    MPI_Comm comm_group;
    MPI_Comm comm_temp;
    MPI_Comm comm_subgroup;

    size_t count;
    size_t count_local;

    public:
    T *sendbuf_d;
    T *recvbuf_d;
    T *recvbuf_d_local;

    T *sendbuf_h;
    T *recvbuf_h;
    T *recvbuf_h_local;


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
      MPI_Comm_split(comm_mpi, myid % groupsize, mygroup, &comm_group);
      MPI_Comm_split(comm_mpi, mygroup, myid % groupsize, &comm_temp);
      MPI_Comm_split(comm_temp, (myid % groupsize) % subgroupsize, mysubgroup, &comm_subgroup);

      {
        int myid_group_test;
        int myid_subgroup_test;
        int numproc_group_test;
        int numproc_subgroup_test = 5;
        MPI_Comm_rank(comm_group, &myid_group_test);
        MPI_Comm_rank(comm_subgroup, &myid_subgroup_test);
        MPI_Comm_size(comm_group, &numproc_group_test);
        MPI_Comm_size(comm_subgroup, &numproc_subgroup_test);
        //printf("myid %d mygroup %d/%d (%d/%d) mysubgroup %d/%d (%d/%d)\n", myid, mygroup, numgroup, myid_group_test, numproc_group_test, mysubgroup, numsubgroup, myid_subgroup_test, numproc_subgroup_test);
      }

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

    printf("count %d numgroup %d\n", count, numgroup);

    // MEMORY MANAGEMENT
    printf("malloc\n");
    sendbuf_d = new T[count];
    recvbuf_d = new T[count * numgroup];
    recvbuf_d_local = new T[count_local * numsubgroup];
    // DONE
#endif

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

        printf("memset\n");
        #pragma omp parallel for
        for(int i = 0; i < count; i++)
          sendbuf_d[i].data[0] = 5;
        memset(sendbuf_d, 0, count * sizeof(T));
        memset(recvbuf_d, 0, numgroup * count * sizeof(T));
        memset(recvbuf_d_local, 0, numsubgroup * count_local * sizeof(T));
#endif
        MPI_Request sendrequest[numgroup];
        MPI_Request recvrequest[numgroup];
        int sendproc = 0;
        int recvproc = 0;
        double time = MPI_Wtime();
        MPI_Barrier(comm_mpi);
        for(int group = 0; group < numgroup; group++)
          if(group != mygroup) {
            printf("group %d iter %d numiter %d\n", group, iter, numiter);
            MPI_Irecv(recvbuf_d + group * count, count * sizeof(T), MPI_BYTE, group, MPI_ANY_TAG, comm_group, recvrequest + recvproc);
            MPI_Isend(sendbuf_d, count * sizeof(T), MPI_BYTE, group, 0, comm_group, sendrequest + sendproc);
            recvproc++;
            sendproc++;
          }
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
