# Bisection Bandwidth Tester
This repository involves a unit test for measuring the bandwidth of communication of a group of processors (mainly GPUs). It is based on MPI and tests various capabilites such as CPU-Only MPI, GPU-Aware MPI, CPU-Staged MPI, NCCL, and IPC.

Porting the capabilities are controlled by preprocessor directives. With no specification, it targets CPU by default. To port on Nvidia GPUs, one needs to ```#define PORT_CUDA```. To port on AMD GPUs, one needs to ```#define PORT_HIP```. Please refer to the table at the bottom to enable desired capabilities.

There are two parameters to describe the logical group topology. The first one is the number of processors (p) and the second one is the group size (g). The benchmarking tool splits the global communicator ```MPI_COMM_WORLD``` into subcommunicators with ```MPI_Comm_split```. Eeach group talks to all other groups with a mapping between GPU as shown in the figure below.

![Group Examples](https://github.com/merthidayetoglu/OLCF_BW_test/blob/main/results/group_examples.png)

These partitioning scenarios can be applied to test communication bandwidth accross nodes, among GPUs within nodes, and between pairs of GPUs.

Considering a hierarchical communication network, MPI ranks are assumed to be assigned in as an SMP style. For example, if there are six GPUs and three nodes, GPU 0 and GPU 1 are in the same node and so GPU 2 and GPU3. In this test, each MPI rank runs a single GPU and user is responsible to place the ranks correctly. The first GPU of a group talks to the first GPU on each group, the second GPU of a group talks to the second GPU on each group, and so on. This test excludes the self communication.

This tool runs like
```cpp
mpirun -np #numproc Alltoall #count #warmup #numiter #groupsize
```
where count is the number of 32-byte words between two GPUs. The number of iterations is for averaging the bandwidth over many times. The program performs one warmup round and measures time over the number of iteration where each iteration are bulk-synchronized individually.

The figure below summarizes the Summit results. Note that all involved GPUs both sends and receives data and the measurement of the aggregate bidirectional bandwidth of a group is reported in GB/s.

![Summit Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/summit_bandwidth.png)

We use the default MPI implementation in the system. You can find more details in the dropdown menu on the bottom of the page. The table below summarizes the testing capabilities (where available).

| Porting Options   | Capability | Include |
| :---:               | ---: | :--- |
|Default is on CPU  | MPI | `#define MPI` |
|`#define PORT_CUDA` | CUDA-Aware MPI <br> CPU-Staged MPI <br> NCCL <br> CUDA IPC | `#define MPI` <br> `#define MPI_staged` <br> `#define NCCL` <br> `#define IPC` |
|`#define PORT_HIP`  | GPU-Aware MPI <br> CPU-Staged MPI <br> (AMD port) NCCL <br> HIP IPC | `#define MPI` <br> `#define MPI_staged` <br> `#define NCCL` <br> `#define IPC` |

Running on each system is like driving a different sports car, which has different handling and steering behaviour. This benchmarking tool helps understanding of the system characteristics. Our evaluation of various systems can be found below.

<details><summary>Summit Results (Cont'd)</summary>
<p>

Summit has [IBM Spectrum MPI](https://www.ibm.com/docs/en/SSZTET_EOS/eos/guide_101.pdf), which uses a lower-level transport layer called parallel active message interfece (PAMI). By default, PAMI variables are configured to have a lower latency [as reported here](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#spectrum-mpi-tunings-needed-for-maximum-bandwidth). Thanks [Chris Zimmer](https://www.olcf.ornl.gov/directory/staff-member/christopher-zimmer/) for pointing it out! To obtain full theoretical bandwidth, we set up the PAMI variables as:
```bash
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ADAPTER_AFFINITY=1
export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1"
export PAMI_IBV_DEVICE_NAME_1="mlx5_3:1,mlx5_0:1"
```

Results with default configuration is shown below (not to be confused with the full-bandwidth configuration that is shown above). We include the equation for calculating the theoretical bandwidth of the CPU-Staged mode.

![Summit Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/summit_latency.png)

NCCL performs irrespective of the PAMI configuration, because it uses UCX API across nodes. CUDA-Aware MPI breaks down with large message sizes due to a known problem.
  
[Summit User Guide](https://docs.olcf.ornl.gov/systems/summit_user_guide.html)
  
</p>
</details>


<details><summary>Crusher Results</summary>
<p>

Crusher is a testbed for Frontier&mdash;the first official exascale system. They have the same node architecture and software toolchain. It has Cray MPICH MPI implementation by default.
  
![Crusher Across Nodes](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/crusher_across_nodes.png)

![Crusher Within Nodes](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/crusher_within_nodes.png)

[Crusher User Guide](https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html)
  
</p>
</details>


<details><summary>Perlmutter Results</summary>
<p>


![Perlmutter Bandwidth](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/perlmutter_bandwidth.png)

</p>
</details>

<details><summary>Delta Results</summary>
<p>

Delta is an NCSA system that is composed of multi-GPU nodes with four Nvidia A100 GPUs each. It has Slingshot 10 and runs OpenMPI+UCX by default.
  
![Delta Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/delta_measurement.png)

[Delta User Guide](https://wiki.ncsa.illinois.edu/display/DSC/Delta+User+Guide)
  
</p>
</details>


<details><summary>Spock Results</summary>
<p>
  
Spock is an experimental system at OLC that is composed of multi-GPU nodes with four AMD MI100 GPUs each. It has Slingshot 10 and runs Cray MPICH+OFI by default. We also tried Cray MPICH+UCX by loading modules `craype-network-ucx` and `cray-mpich-ucx`.

![Spock Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/spock_measurement.png)

The results below are taken within one node with the default MPI because Cray MPICH+UCX crahes with buffer size larger than 16 KB when GPUs are involved.

![Spock Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/spock_within_nodes.png)

[Spock User Guide](https://docs.olcf.ornl.gov/systems/spock_quick_start_guide.html)

<p>

</p>
</details>

<details><summary>ThetaGPU Results</summary>
<p>

ThetaGPU is an Nvidia DGX-A100 System with eight GPUs per node. The GPUs each GPU is connected to six NVSwitches via NVLinks, where each link has 100 GB/s bidirectional bandwidth. Considering the physical communication architecture, we can model the bisection bandwidth within a fully-connected topology, where each GPUs has a peak bandwidth of 600 GB/s. As a result, the bisection bandwidth of a group can be written as:
  
```math
\beta_{\textrm{group}}^{-1} = g\times600\textrm{ GB/s}
```
where g is the number of GPUs in each group. The figure below shows the bandwidth measurements with various configuration within the DGX-A100 node.

![ThetaGPU Across Nodes](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/thetaGPU_across_nodes.png)

![ThetaGPU Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/thetaGPU_within_nodes.png)


[ThetaGPU User Guide](https://maps.app.goo.gl/GLmdk82YJF3EWeiJ9)

</p>
</details>

For reproducibility, we provide the preworked Makefiles and run scripts for various systems, including Summit, Crusher, Spock, Delta, and ThetaGPU in the scripts folder.

As a summary of our measurements across nodes, the table below reports the measured peak bandwidth in GB/s. The bandwidth numbers represents the maximum attained bandwidth within package size interval of 4 Bytes to 1 GB. Measurement across nodes are performed with 48 GPUs from each system. As a result, the experiments are carried out on 16 nodes of Spock, Delta, and Perlmutter, 12 nodes of Summit, and 6 nodes of Frontier and ThetaGPU.

![Bandwidth Utilization](https://github.com/merthidayetoglu/OLCF_BW_test/blob/main/results/bandwidth_across_nodes.png)

Across nodes, $g$ is the number of GPUs per node, and $p$ is equal to the total number of GPUs. Within nodes, each pair of $p$, $g$ tests a different configuration.

![Bandwidth Within Nodes](https://github.com/merthidayetoglu/OLCF_BW_test/blob/main/results/bandwidth_within_nodes.png)


Please send and email to [merth@stanford.edu](merth@stanford.edu) for any questions or contributions. Especially, extension of this tool with GASNet implementation would be great!
