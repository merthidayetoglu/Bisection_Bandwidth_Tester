
#include "comm.h"

namespace CommBench
{

  enum heuristic {across, within};

  template <typename T>
  class Bench
  {
    MPI_Comm comm;
    const heuristic type;
    const int groupsize;
    int numgroup;
    int mygroup;

    const size_t count;

    size_t *sendcount;
    size_t *sendoffset;
    size_t *recvcount;
    size_t *recvoffset;

    public:

    void measure(int numiter);
    void test();

    Bench(const MPI_Comm &comm_world, const int groupsize, const heuristic type, capability cap, const size_t count) : groupsize(groupsize), type(type), count(count) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_world, &myid);
      MPI_Comm_size(comm_world, &numproc);

      switch(type) {
        case across:
          MPI_Comm_split(MPI_COMM_WORLD, myid % groupsize, myid / groupsize, &comm);
          break;
        case within:
          MPI_Comm_split(comm_world, myid / groupsize, myid % groupsize, &comm);
          break;
      }

      MPI_Comm_rank(comm, &mygroup);
      MPI_Comm_size(comm, &numgroup);

      printf("myid %d numproc %d mygroup %d numgroup %d\n", myid, numproc, mygroup, numgroup);
    };

  }; // class Bench

} // namespace CommBench*/
