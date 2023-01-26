

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

    const size_t count;
    const double ratio;

    public:

    void init(T *&commbuf, transport cap_global, transport cap_local);

    Bench(const size_t count, const int groupsize, const int subgroupsize, const double ratio, const MPI_Comm &comm_mpi, size_t &buffsize) : count(count), groupsize(groupsize), subgroupsize(subgroupsize), ratio(ratio), comm_mpi(comm_mpi) {

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
      if(myid == ROOT)
        printf("sendDim %d recvDim_global %d recvDim_local %d\n", sendDim, recvDim_global, recvDim_local);
      size_t sendDim_total = sendDim;
      size_t recvDim_total_global = recvDim_global;
      size_t recvDim_total_local = recvDim_local;
      MPI_Allreduce(MPI_IN_PLACE, &sendDim_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_mpi);
      MPI_Allreduce(MPI_IN_PLACE, &recvDim_total_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_mpi);
      MPI_Allreduce(MPI_IN_PLACE, &recvDim_total_local, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_mpi);
      if(myid == ROOT) {
        printf("total sendDim %d total recvDim_global %d total recvDim_local %d\n", sendDim_total, recvDim_total_global, recvDim_total_local);
        printf("total per GPU %f\n", (sendDim_total + recvDim_total_global + recvDim_total_local) / (double)numproc);

        printf("Now initialize bench with comm.init\n");
        printf("\n");
      }

      buffsize = 55;
    }
  };

  template<typename T>
  void Bench<T>::init(T *&commbuf, transport cap_global, transport cap_local) {

    

    if(myid == ROOT) {
      printf("initialize capabilities\n");
      
    }
  };


}
