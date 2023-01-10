# OLCF_BW_test
Unit test for measuring system bandwidth. It is based on MPI but tests additional capabilites such as GPU-Aware MPI, CPU-Staged MPI, NCCL, and IPC.

By default it works on CPU. To test Nvidia GPU, you need to ```#define SCI_CUDA```. To test AMD GPU, you need to ```#define SCI_HIP```.

There are two important parameters. The first one is the number of processors shown with $P$ and the second one is the group size shown with $G$. All groups talk to each other with a mapping between GPU as shown in the figure below.

![This is an image](https://github.com/merthidayetoglu/OLCF_BW_test/blob/main/images/group_examples.png)
