# OLCF_BW_test
Unit test for measuring bisectional bandwidth. It is based on MPI but tests additional capabilites such as GPU-Aware MPI, CPU-Staged MPI, NCCL, and IPC.

By default it works on CPU. To test Nvidia GPU, you need to ```#define SCI_CUDA```. To test AMD GPU, you need to ```#define SCI_HIP```.

There are two important parameters. The first one is the number of processors shown with $P$ and the second one is the group size shown with $G$. All groups talk to each other with a mapping between GPU as shown in the figure below.

![Group Examples](https://github.com/merthidayetoglu/OLCF_BW_test/blob/main/images/group_examples_corrected.png)

This tool runs like
```
mpirun -np #numproc Alltoall #numiter #groupsize
```

Number of iterations is for averaging the bandwidth over many times. The program performs one warmup round and measures time over the number of iteration.

There are preworked Makefiles and run scripts for Summit and Crusher.

![Summit Measurements](https://github.com/merthidayetoglu/OLCF_BW_test/blob/main/images/summit_measurement.png)

Please send and email to [merth@stanford.edu](merth@stanford.edu) for any questions or contributions.

Extension of this small tool with GASNet-EX would be great!
