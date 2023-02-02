
#include "comm.h"

namespace CommBench
{

  enum heuristic {across, within};

  template <typename T>
  class Bench
  {
    MPI_Comm commgroup;
    Comm<T> *transport;

    const heuristic mode;
    const size_t count;

    T *sendbuf;
    T *recvbuf;

    public:

    void start() { transport->start(); };
    void wait() { transport->wait(); };
    void test();

    ~Bench() {
      delete transport;
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
        printf("CommBench: Creating a Bench object requires global synchronization\n");


      int myid;
      int numproc;
      MPI_Comm_rank(comm_world, &myid);
      MPI_Comm_size(comm_world, &numproc);

      if(myid_root == ROOT)
        printf("CommBench: Create bench with %d processes\n", numproc);

      switch(mode) {
        case across:
          MPI_Comm_split(comm_world, myid % groupsize, myid / groupsize, &commgroup);
          if(myid_root == ROOT)
            printf("CommBench: Split comm across groups of %d\n", groupsize);
          break;
        case within:
          MPI_Comm_split(comm_world, myid / groupsize, myid % groupsize, &commgroup);
          if(myid_root == ROOT)
            printf("CommBench: Split comm within groups of %d\n", groupsize);
          break;
      }

      int mygroup;
      int numgroup;
      MPI_Comm_rank(commgroup, &mygroup);
      MPI_Comm_size(commgroup, &numgroup);

      size_t sendcount[numgroup];
      size_t sendoffset[numgroup];
      size_t recvcount[numgroup];
      size_t recvoffset[numgroup];

      switch(mode) {
        case across:
          {
	    if(myid_root == ROOT) {
              printf("CommBench: There are %d groups to comm. across\n", numgroup);
              printf("CommBench: allocate %e GB comm buffer\n", count * numgroup * sizeof(T) / 1.e9);
	    }
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
            if(myid_root == ROOT) {
              printf("allocate %e GB comm buffer\n", count * (numgroup + 1) * sizeof(T) / 1.e9);
              printf("There are %d processes to comm within\n", numgroup);
            }
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

      transport = new Comm<T>(sendbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, commgroup, cap);
    }

  }; // class Bench

  template <typename T>
  class Allgather {

    Comm<T> *transport;

    public:

    Allgather(T *sendbuf, size_t count, T *recvbuf, const MPI_Comm &comm, capability cap) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &numproc);

      if(myid == ROOT)
        printf("CommBench: Creating Allgather object\n");

      size_t sendcount[numproc];
      size_t recvcount[numproc];
      size_t sendoffset[numproc];
      size_t recvoffset[numproc];
      for(int p = 0; p < numproc; p++) {
        sendcount[p] = count;
        sendoffset[p] = 0;
        recvcount[p] = count;
        recvoffset[p] = p * count;
      }

      transport = new Comm<T>(sendbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, comm, cap);
    }

    ~Allgather() {delete transport;};
    void wait() {transport->start(); transport->wait();};
  };

} // namespace CommBench*/
