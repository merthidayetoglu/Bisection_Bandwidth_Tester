
#include "comm.h"

namespace CommBench
{

  class Arch {

    const int numlevel;

    MPI_Comm *comm_within = new MPI_Comm[numlevel + 1];
    MPI_Comm *comm_across = new MPI_Comm[numlevel + 1];
   
    public:

    Arch(const int numlevel, int groupsize[], const MPI_Comm &comm_world) : numlevel(numlevel) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_world, &myid);
      MPI_Comm_size(comm_world, &numproc);

      MPI_Comm_split(comm_world, myid / numproc, myid % numproc, comm_within);
      MPI_Comm_split(comm_world, myid % numproc, myid / numproc, comm_across);
      for(int level = 1; level < numlevel; level++) {
        int myid_within;
        MPI_Comm_rank(comm_within[level - 1], &myid_within);
        MPI_Comm_split(comm_within[level - 1], myid_within / groupsize[level - 1], myid_within % groupsize[level - 1], comm_within + level);
        MPI_Comm_split(comm_within[level - 1], myid_within % groupsize[level - 1], myid_within / groupsize[level - 1], comm_across + level);
      }
      int myid_within;
      MPI_Comm_rank(comm_within[numlevel - 1], &myid_within);
      MPI_Comm_split(comm_within[numlevel - 1], myid_within / 1, myid_within % 1, comm_within + numlevel);
      MPI_Comm_split(comm_within[numlevel - 1], myid_within % 1, myid_within / 1, comm_across + numlevel);

      for(int level = 0; level < numlevel + 1; level++) {
        int numproc_within;
        int numproc_across;
        MPI_Comm_size(comm_within[level], &numproc_within);
        MPI_Comm_size(comm_across[level], &numproc_across);
        if(myid == ROOT)
          printf("level %d numproc_within %d numproc_across %d\n", level, numproc_within, numproc_across);
      }
    } 

  };

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
  void Bench<T>::test() {

    int myid;
    int numproc;
    MPI_Comm_rank(commgroup, &myid);
    MPI_Comm_size(commgroup, &numproc);

    int recvproc;
    switch(mode) {case across: recvproc = numproc - 1; break; case within: recvproc = numproc; break;}

    T *sendbuf = new T[count];
    T *recvbuf = new T[count * recvproc];

    for(size_t i = 0; i < count; i++)
      sendbuf[i].data[0] = myid;
    memset(recvbuf, -1, count * recvproc * sizeof(T));

#ifdef PORT_CUDA
    cudaMemcpy(this->sendbuf, sendbuf, count * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(this->recvbuf, -1, count * recvproc * sizeof(T));
#elif defined PORT_HIP
    hipMemcpy(this->sendbuf, sendbuf, count * sizeof(T), hipMemcpyHostToDevice);
    hipMemset(this->recvbuf, -1, count * recvproc * sizeof(T));
#else
    memcpy(this->sendbuf, sendbuf, count * sizeof(T));
    memset(this->recvbuf, -1, count * recvproc * sizeof(T));
#endif

    this->start();
    this->wait();

#ifdef PORT_CUDA
    cudaMemcpy(recvbuf, this->recvbuf, count * recvproc * sizeof(T), cudaMemcpyDeviceToHost);
#elif defined PORT_HIP
    hipMemcpy(recvbuf, this->recvbuf, count * recvproc * sizeof(T), hipMemcpyDeviceToHost);
#else
    memcpy(recvbuf, this->recvbuf, count * recvproc * sizeof(T));
#endif

    bool pass = true;
    switch(mode) {
      case across:
        recvproc = 0;
        for(int p = 0; p < numproc; p++)
          if(p != myid) {
            for(size_t i = 0; i < count; i++)
              if(recvbuf[recvproc * count + i].data[0] != p)
                pass = false;
            recvproc++;
          }
        break;
      case within:
        for(int p = 0; p < numproc; p++)
          for(size_t i = 0; i < count; i++)
            if(recvbuf[p * count + i].data[0] != p)
              pass = false;
        break;
    }
    MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if(pass && myid == ROOT)
      printf("PASS!\n");
    else
      if(myid == ROOT)
        printf("ERROR!!!!\n");

    delete[] sendbuf;
    delete[] recvbuf;
  }

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
