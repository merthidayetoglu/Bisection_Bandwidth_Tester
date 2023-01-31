
#include "comm.h"

namespace CommBench
{

  enum heuristic {across, within};

  template <typename T>
  class Bench
  {
    MPI_Comm comm_group;
    Comm<T> *comm;

    const heuristic mode;
    const size_t count;

    T *sendbuf;
    T *recvbuf;

    public:

    void start() { comm->start(); };
    void wait() { comm->wait(); };
    void test();

    ~Bench() {
#ifdef PORT_CUDA
      cudaFree(sendbuf);
      cudaFree(recvbuf);
#elif defined PORT_HIP
      hipFree(sendbuf);
      hipFree(recvbuf);
#else
      delete[] sendbuf;
      delete[] recvbuf;
#endif
    };

    Bench(const MPI_Comm &comm_world, const int groupsize, const heuristic mode, capability cap, const size_t count) : mode(mode), count(count) {

      int myid_root;
      MPI_Comm_rank(MPI_COMM_WORLD, &myid_root);
      if(myid_root == ROOT)
        printf("Creating a Bench object requires global synchronization\n");

      int myid;
      int numproc;
      MPI_Comm_rank(comm_world, &myid);
      MPI_Comm_size(comm_world, &numproc);

      if(myid_root == ROOT)
        printf("Create bench with %d processes\n", numproc);

      switch(mode) {
        case across:
          MPI_Comm_split(comm_world, myid % groupsize, myid / groupsize, &comm_group);
          if(myid_root == ROOT)
            printf("Split comm across groups of %d\n", groupsize);
          break;
        case within:
          MPI_Comm_split(comm_world, myid / groupsize, myid % groupsize, &comm_group);
          if(myid_root == ROOT)
            printf("Split comm within groups of %d\n", groupsize);
          break;
      }

      int mygroup;
      int numgroup;
      MPI_Comm_rank(comm_group, &mygroup);
      MPI_Comm_size(comm_group, &numgroup);

      if(myid_root == ROOT && mode == across)
        printf("There are %d groups to comm. across\n", numgroup);
      if(myid_root == ROOT && mode == within)
        printf("There are %d processes to comm within\n", numgroup);

      size_t sendcount[numgroup];
      size_t sendoffset[numgroup];
      size_t recvcount[numgroup];
      size_t recvoffset[numgroup];

      switch(mode) {
        case across:
          {
            printf("allocate %e GB comm buffer\n", count * numgroup * sizeof(T) / 1.e9);
#ifdef PORT_CUDA
            if(myid_root == ROOT) {
              printf("CUDA memory management\n");
            }
            cudaMalloc(&sendbuf, count * sizeof(T));
            cudaMalloc(&recvbuf, count * (numgroup - 1) * sizeof(T));
#elif defined PORT_HIP
            if(myid_root == ROOT)
              printf("HIP memory management\n");
            hipMalloc(&sendbuf, count * sizeof(T));
            hipMalloc(&recvbuf, count * (numgroup - 1) * sizeof(T));
#else
            if(myid_root == ROOT)
              printf("CPU memory management\n");
            sendbuf = new T[count];
            recvbuf = new T[count * (numgroup - 1)];
#endif
            int numrecv = 0;
            for(int group = 0; group < numgroup; group++)
              if(group != mygroup) {
                sendcount[group] = count;
                recvcount[group] = count;
                sendoffset[group] = 0;
                recvoffset[group] = numrecv * count;
                numrecv++;
              }
              else {
                sendcount[group] = 0;
                recvcount[group] = 0;
              }
          }
          break;
        case within:
          {
            printf("allocate %e GB comm buffer\n", count * (numgroup + 1) * sizeof(T) / 1.e9);
#ifdef PORT_CUDA
            if(myid_root == ROOT)
              printf("CUDA memory management\n");
            cudaMalloc(&sendbuf, count * sizeof(T));
            cudaMalloc(&recvbuf, count * numgroup * sizeof(T));
#elif defined PORT_HIP
            if(myid_root == ROOT)
              printf("HIP memory management\n");
            hipMalloc(&sendbuf, count * sizeof(T));
            hipMalloc(&recvbuf, count * numgroup * sizeof(T));
#else
            if(myid_root == ROOT)
              printf("CPU memory management\n");
            sendbuf = new T[count];
            recvbuf = new T[count * numgroup];
#endif
            for(int group = 0; group < numgroup; group++) {
              sendcount[group] = count;
              recvcount[group] = count;
              sendoffset[group] = 0;
              recvoffset[group] = group * count;
            }
          }
          break;
      }

      if(myid_root == ROOT)
        printf("\n");

      comm = new Comm<T>(sendbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, comm_group, cap);
    }

  }; // class Bench

  template <typename T>
  void Bench<T>::test() {

    int myid_root;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid_root);
#ifdef PORT_CUDA
    cudaMemset(sendbuf, myid_root, count * sizeof(T));
#elif defined PORT_HIP
    hipMemset(sendbuf, myid_root, count * sizeof(T));
#else
    memset(sendbuf, myid_root, count * sizeof(T));
#endif

  }

} // namespace CommBench*/
