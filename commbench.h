
#define CommBench_ROOT = 0

namespace CommBench
{
  enum cap{MPI, MPI_staged, NCCL, IPC};

  struct Arch
  {
    const int printid;
    const int numlevel;
    cap *bridge = new cap[numlevel];

    // Communicators
    MPI_Comm *comm_mpi_across = new MPI_Comm[numlevel];
    MPI_Comm *comm_mpi_within;

    // Single-level constructor
    Arch(MPI_Comm &comm_mpi_data) : Arch(1, nullptr, comm_mpi_data) {};

    // Multilevel constructor
    Arch(const int numlevel, int *numprocs, const MPI_Comm &comm_mpi_data) : numlevel(numlevel) {
      for(int l = 0; l < numlevel; l++)
        bridge[l] = MPI;
      init(numprocs, comm_mpi_data);
    }
    // Multilevel constructor with capabilities
    Arch(const int numlevel, const int *const numprocs, cap *bridges, const MPI_Comm &MPI_COMM_DATA) : numlevel(numlevel) {
      for(int l = 0; l < numlevel; l++)
        bridge[l] = bridges[l];
      init(numprocs, comm_mpi_data);
    }
    // Deep copy constructor
    Arch(const Arch &arch) : numlevel{arch.numlevel}, printid{arch.printid}
    {
      for (int l = 0; l < numlevel + 1; l++)
      {
        MPI_Comm_dup(arch.comm_mpi_across[l], comm_mpi_across + l);
        MPI_Comm_dup(arch.comm_mpi_within[l], comm_mpi_within + l);
      }
      memcpy(bridge, arch.bridge, numlevel * sizeof(cap));
      if (printid == CommBenc_ROOT)
        printf("Construct a deep copy of %d-level architecture\n\n", numlevel);
    }

} // namespace CommBench
