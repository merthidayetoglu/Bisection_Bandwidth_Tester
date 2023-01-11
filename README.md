# Bisection Bandwidth Tester
This repository involves a unit test for measuring the bandwidth of communication of a group of processors (mainly GPUs). It is based on MPI but tests additional capabilites such as GPU-Aware MPI, CPU-Staged MPI, NCCL, and IPC.

Porting the capabilities are controlled by preprocessor directives. With no specification, it targets CPU by default. To port on Nvidia GPUs, one needs to ```#define SCI_CUDA```. To port on AMD GPUs, one needs to ```#define SCI_HIP```. Please refer to the table at the bottom to enable desired capabilities.

There are two parameters to describe the logical group topology. The first one is the number of processors and the second one is the group size. The benchmarking tool splits the global communicator ```MPI_COMM_WORLD``` into subcommunicators with ```MPI_Comm_split```. Eeach group talks to all other groups with a mapping between GPU as shown in the figure below. These partitioning scenarios can be applied to test communication bandwidth accross nodes, among GPUs within nodes, and between pairs of GPUs. In this scheme, each MPI rank runs a single GPU.

![Group Examples](https://github.com/merthidayetoglu/OLCF_BW_test/blob/main/results/group_examples.png)

Considering a hierarchical communication network, MPI ranks are assumed to be assigned in as an SMP style. For example, if there are six GPUs and three nodes, GPU 0 and GPU 1 are in the same node and so GPU 2 and GPU3, and so on. The first GPU of a group talks to the first GPUs on other groups, the second GPU of a group talks to the corresponding second GPUs on other groups, and so on.

This tool runs like
```cpp
mpirun -np #numproc Alltoall #count #numiter #groupsize
```
where count is the number of 32-byte words between two GPUs. The number of iterations is for averaging the bandwidth over many times. The program performs one warmup round and measures time over the number of iteration where each iteration are bulk-synchronized individually.

The figure below summarizes the Summit results. Note that all involved GPUs both sends and receives data and the measurement of the aggregate bidirectional bandwidth of a group is reported in GB/s.

![Summit Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/summit_measurement_bandwidth.png)

We use the default MPI implementation in the system. You can find more details in the dropdown menu on the bottom of the page. The table below summarizes the implemented capabilities.

| Porting Options   | Capability | Include |
| :---:               | ---: | :--- |
|Default is on CPU  | MPI | `#define MPI` |
|`#define SCI_CUDA` | CUDA-Aware MPI <br> CPU-Staged MPI <br> NCCL <br> CUDA IPC | `#define MPI` <br> `#define MPI_Staged` <br> `#define NCCL` <br> `#define IPC` |
|`#define SCI_HIP`  | GPU-Aware MPI <br> CPU-Staged MPI <br> (AMD port) NCCL <br> HIP IPC | `#define MPI` <br> `#define MPI_Staged` <br> `#define NCCL` <br> `#define IPC` |

Running on each system is like driving a different sports car, which has different handling and steering behaviour. This benchmarking tool helps understanding of the system characteristics. Our evaluation of various systems can be found below.

<details><summary>Summit Results</summary>
<p>

Summit has IBM Spectrum MPI, which uses a lower-level API called parallel active message interfece (PAMI). By default, PAMI variables are configured to have a lower latency [as reported here](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#spectrum-mpi-tunings-needed-for-maximum-bandwidth). Thanks [Chris Zimmer](https://www.olcf.ornl.gov/directory/staff-member/christopher-zimmer/) for pointing it out! To obtain full theoretical bandwidth, we set up the PAMI variables as:
```bash
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ADAPTER_AFFINITY=1
export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1"
export PAMI_IBV_DEVICE_NAME_1="mlx5_3:1,mlx5_0:1"
```

Results with default configuration is shown below (not to be confused with the full-bandwidth configuration that is shown above).

![Crusher Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/summit_measurement_latency.png)

NCCL performs irrespective of the PAMI configuration, because it uses UCX API across nodes. CPU-Staged MPI breaks down with large message sizes due to a known problem.
  
</p>
</details>

<details><summary>Crusher Results</summary>
<p>

Crusher is a testbed for Frontier&mdash;the first official exascale system. They have the same node architecture and software toolchain. It has Cray MPICH MPI implementation by default.
  
![Crusher Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/crusher_measurement.png)

</p>
</details>

<details><summary>Delta Results</summary>
<p>

Delta is a system composed of multi-GPU nodes with four A100 GPUs each. It has OpenMPI+UCX implementation by default.
  
![Delta Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/delta_measurement.png)

</p>
</details>

<details><summary>Spock Results</summary>
<p>

</p>
</details>

<details><summary>ThetaGPU Results</summary>
<p>


</p>
</details>

For reproducibility, we provide the preworked Makefiles and run scripts for various systems, including Summit, Crusher, Spock, Delta, and ThetaGPU in the scripts folder.

Please send and email to [merth@stanford.edu](merth@stanford.edu) for any questions or contributions. Especially, extension of this benchmarking tool with GASNet-EX and NVSHMEM capabilities would be great!
