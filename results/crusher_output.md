Request Allocation
```
[merth@login2.crusher scripts]$ source submit_crusher.sh 
Resetting modules to system default. Reseting $MODULEPATH back to system default. All extra directories will be removed from $MODULEPATH.
salloc: Pending job allocation 253145
salloc: job 253145 queued and waiting for resources
salloc: job 253145 has been allocated resources
salloc: Granted job allocation 253145
salloc: Waiting for resource configuration
salloc: Nodes crusher[062-065] are ready for job
bashrc
merth@crusher062:~/Bisection_Bandwidth_Tester/scripts>
```
Compiling
```
merth@crusher062:~/Bisection_Bandwidth_Tester> make
CC -std=c++14 -fopenmp -I/opt/rocm-5.1.0/include -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 -x hip -O3  main.cpp -c -o main.o -craype-verbose
clang++ -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__ --offload-arch=gfx90a -D__HIP_ARCH_GFX90A__=1 -dynamic -D__CRAY_X86_TRENTO -D__CRAY_AMD_GFX90A -D__CRAYXT_COMPUTE_LINUX_TARGET --gcc-toolchain=/opt/cray/pe/gcc/10.3.0/snos -isystem /opt/cray/pe/cce/14.0.2/cce-clang/x86_64/lib/clang/14.0.6/include -isystem /opt/cray/pe/cce/14.0.2/cce/x86_64/include/craylibs -std=c++14 -fopenmp -I/opt/rocm-5.1.0/include -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 -x hip -O3 main.cpp -c -o main.o -I/opt/cray/pe/libsci/22.06.1.3/CRAY/9.0/x86_64/include -I/opt/cray/pe/mpich/8.1.17/ofi/cray/10.0/include -I/opt/cray/pe/dsmml/0.2.2/dsmml//include -I/opt/cray/pe/pmi/6.1.3/include -I/opt/cray/xpmem/2.4.4-2.3_11.2__gff0e1d9.shasta/include 
CC -o Alltoall main.o  -fopenmp -L/opt/rocm-5.1.0/lib -lamdhip64 -lrccl
merth@crusher062:~/Bisection_Bandwidth_Tester> 
```
Running
```
merth@crusher062:~/Bisection_Bandwidth_Tester> ./run.sh 
craype-x86-trento
libfabric/1.15.0.0
craype-network-ofi
perftools-base/22.06.0
xpmem/2.4.4-2.3_11.2__gff0e1d9.shasta
cray-pmi/6.1.3
cce/14.0.2
craype-accel-amd-gfx90a
craype/2.7.16
cray-dsmml/0.2.2
cray-mpich/8.1.17
cray-libsci/22.06.1.3
PrgEnv-cray/8.3.3
xalt/1.3.0
DefApps/default
rocm/5.1.0
Mon 23 Jan 2023 09:45:37 PM EST
PE 0: MPICH processor detected:
PE 0:   AMD Milan (25:48:1) (family:model:stepping)
PE 0: MPICH environment settings =====================================
PE 0:   MPICH_ENV_DISPLAY                              = 1
PE 0:   MPICH_VERSION_DISPLAY                          = 0
PE 0:   MPICH_ABORT_ON_ERROR                           = 0
PE 0:   MPICH_CPUMASK_DISPLAY                          = 0
PE 0:   MPICH_STATS_DISPLAY                            = 0
PE 0:   MPICH_RANK_REORDER_METHOD                      = 1
PE 0:   MPICH_RANK_REORDER_DISPLAY                     = 0
PE 0:   MPICH_MEMCPY_MEM_CHECK                         = 0
PE 0:   MPICH_USE_SYSTEM_MEMCPY                        = 0
PE 0:   MPICH_OPTIMIZED_MEMCPY                         = 1
PE 0:   MPICH_ALLOC_MEM_PG_SZ                          = 4096
PE 0:   MPICH_ALLOC_MEM_POLICY                         = PREFERRED
PE 0:   MPICH_ALLOC_MEM_AFFINITY                       = SYS_DEFAULT
PE 0:   MPICH_MALLOC_FALLBACK                          = 0
PE 0:   MPICH_MEM_DEBUG_FNAME                          = 
PE 0:   MPICH_INTERNAL_MEM_AFFINITY                    = SYS_DEFAULT
PE 0:   MPICH_NO_BUFFER_ALIAS_CHECK                    = 0
PE 0:   MPICH_COLL_SYNC                                = 0
PE 0:   MPICH_SINGLE_HOST_ENABLED                        = 1
PE 0: MPICH/RMA environment settings =================================
PE 0:   MPICH_RMA_MAX_PENDING                          = 128
PE 0:   MPICH_RMA_SHM_ACCUMULATE                       = 0
PE 0: MPICH/GPU environment settings =================================
PE 0:   MPICH_GPU_SUPPORT_ENABLED                      = 1
PE 0:   MPICH_GPU_IPC_ENABLED                          = 1
PE 0:   MPICH_GPU_EAGER_REGISTER_HOST_MEM              = 1
PE 0:   MPICH_GPU_IPC_THRESHOLD                        = 1024
PE 0:   MPICH_GPU_NO_ASYNC_COPY                        = 0
PE 0:   MPICH_GPU_COLL_STAGING_AREA_OPT                = 1
PE 0:   MPICH_GPU_EAGER_DEVICE_MEM                     = 0
PE 0:   MPICH_USE_GPU_STREAM_TRIGGERED                 = 0
PE 0:   MPICH_NUM_MAX_GPU_STREAMS                      = 27
PE 0:   MPICH_ENABLE_YAKSA                             = 0
PE 0: MPICH/Dynamic Process Management environment settings ==========
PE 0:   MPICH_DPM_DIR                                  = 3289072
PE 0:   MPICH_LOCAL_SPAWN_SERVER                       = -1
PE 0: MPICH/SMP environment settings =================================
PE 0:   MPICH_SMP_SINGLE_COPY_MODE                     = XPMEM
PE 0:   MPICH_SMP_SINGLE_COPY_SIZE                     = 8192
PE 0:   MPICH_SHM_PROGRESS_MAX_BATCH_SIZE              = 8
PE 0: MPICH/OFI environment settings =================================
PE 0:   MPICH_OFI_USE_PROVIDER                         = cxi
PE 0:   MPICH_OFI_VERBOSE                              = 0
PE 0:   MPICH_OFI_NIC_VERBOSE                          = 0
PE 0:   FI_CXI_RDZV_THRESHOLD                          = 16384
PE 0:   FI_CXI_RDZV_EAGER_SIZE                         = 2048
PE 0:   FI_CXI_DEFAULT_CQ_SIZE                         = 32768
PE 0:   FI_CXI_OFLOW_BUF_SIZE                          = 12582912
PE 0:   FI_CXI_OFLOW_BUF_COUNT                         = 3
PE 0:   FI_MR_CACHE_MAX_SIZE                           = -1
PE 0:   FI_MR_CACHE_MAX_COUNT                          = 524288
PE 0:   MPICH_OFI_DEFAULT_TCLASS                       = TC_BEST_EFFORT
PE 0:   MPICH_OFI_TCLASS_ERRORS                        = warn
PE 0:   MPICH_OFI_CXI_PID_BASE                         = 0
PE 0:   MPICH_OFI_USE_SCALABLE_STARTUP                 = 1
PE 0:   MPICH_OFI_NIC_POLICY                           = NUMA
PE 0:   MPICH_OFI_NIC_MAPPING                          = NULL
PE 0:   MPICH_OFI_NUM_NICS                             = NULL
PE 0:   MPICH_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS     = -1
PE 0:   MPICH_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS        = -1
PE 0:   MPICH_OFI_RMA_STARTUP_CONNECT                  = 0
PE 0:   MPICH_OFI_SKIP_NIC_SYMMETRY_TEST               = 0
PE 0:   MPICH_OFI_STARTUP_CONNECT                      = 0
PE 0: MPICH/COLLECTIVE environment settings ==========================
PE 0:   MPICH_COLL_OPT_OFF                             = 0
PE 0:   MPICH_BCAST_ONLY_TREE                          = 1
PE 0:   MPICH_BCAST_INTERNODE_RADIX                    = 4
PE 0:   MPICH_BCAST_INTRANODE_RADIX                    = 4
PE 0:   MPICH_ALLTOALL_SHORT_MSG                       = 64-512
PE 0:   MPICH_ALLTOALL_SYNC_FREQ                       = 1-24
PE 0:   MPICH_ALLTOALL_BLK_SIZE                        = 16384
PE 0:   MPICH_ALLTOALL_CHUNKING_MAX_NODES              = 90
PE 0:   MPICH_ALLTOALLV_THROTTLE                       = 8
PE 0:   MPICH_ALLGATHER_VSHORT_MSG                     = 8192-16384
PE 0:   MPICH_ALLGATHERV_VSHORT_MSG                    = 8192-16384
PE 0:   MPICH_GATHERV_SHORT_MSG                        = 131072
PE 0:   MPICH_GATHERV_MIN_COMM_SIZE                    = 64
PE 0:   MPICH_GATHERV_MAX_TMP_SIZE                     = 536870912
PE 0:   MPICH_GATHERV_SYNC_FREQ                        = 16
PE 0:   MPICH_IGATHERV_RAND_COMMSIZE                   = 2048
PE 0:   MPICH_IGATHERV_RAND_RECVLIST                   = 0
PE 0:   MPICH_SCATTERV_SHORT_MSG                       = 2048-8192
PE 0:   MPICH_SCATTERV_MIN_COMM_SIZE                   = 64
PE 0:   MPICH_SCATTERV_MAX_TMP_SIZE                    = 536870912
PE 0:   MPICH_SCATTERV_SYNC_FREQ                       = 16
PE 0:   MPICH_SCATTERV_SYNCHRONOUS                     = 0
PE 0:   MPICH_ALLREDUCE_MAX_SMP_SIZE                   = 262144
PE 0:   MPICH_ALLREDUCE_BLK_SIZE                       = 716800
PE 0:   MPICH_GPU_ALLREDUCE_USE_KERNEL                 = 0
PE 0:   MPICH_ALLREDUCE_NO_SMP                         = 0
PE 0:   MPICH_REDUCE_NO_SMP                            = 0
PE 0:   MPICH_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE = 524288
PE 0:   MPICH_REDUCE_SCATTER_MAX_COMMSIZE              = 1000
PE 0:   MPICH_SHARED_MEM_COLL_OPT                      = 1
PE 0:   MPICH_SHARED_MEM_COLL_NCELLS                   = 8
PE 0:   MPICH_SHARED_MEM_COLL_CELLSZ                   = 256
PE 0: MPICH MPIIO environment settings ===============================
PE 0:   MPICH_MPIIO_HINTS_DISPLAY                      = 0
PE 0:   MPICH_MPIIO_HINTS                              = NULL
PE 0:   MPICH_MPIIO_ABORT_ON_RW_ERROR                  = disable
PE 0:   MPICH_MPIIO_CB_ALIGN                           = 2
PE 0:   MPICH_MPIIO_DVS_MAXNODES                       = -1
PE 0:   MPICH_MPIIO_AGGREGATOR_PLACEMENT_DISPLAY       = 0
PE 0:   MPICH_MPIIO_AGGREGATOR_PLACEMENT_STRIDE        = -1
PE 0:   MPICH_MPIIO_MAX_NUM_IRECV                      = 50
PE 0:   MPICH_MPIIO_MAX_NUM_ISEND                      = 50
PE 0:   MPICH_MPIIO_MAX_SIZE_ISEND                     = 10485760
PE 0:   MPICH_MPIIO_OFI_STARTUP_CONNECT                = disable
PE 0:   MPICH_MPIIO_OFI_STARTUP_NODES_AGGREGATOR        = 2
PE 0: MPICH MPIIO statistics environment settings ====================
PE 0:   MPICH_MPIIO_STATS                              = 0
PE 0:   MPICH_MPIIO_TIMERS                             = 0
PE 0:   MPICH_MPIIO_WRITE_EXIT_BARRIER                 = 1
PE 0: MPICH Thread Safety settings ===================================
PE 0:   MPICH_ASYNC_PROGRESS                           = 0
PE 0:   MPICH_OPT_THREAD_SYNC                          = 1
PE 0:   rank 0 required = single, was provided = single


======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
0    39.0c  102.0W  800Mhz  1600Mhz  0%   auto  560.0W    0%   0%    
1    44.0c  N/A     800Mhz  1600Mhz  0%   auto  0.0W      0%   0%    
2    34.0c  93.0W   800Mhz  1600Mhz  0%   auto  560.0W    0%   0%    
3    32.0c  N/A     800Mhz  1600Mhz  0%   auto  0.0W      0%   0%    
4    36.0c  94.0W   800Mhz  1600Mhz  0%   auto  560.0W    0%   0%    
5    40.0c  N/A     800Mhz  1600Mhz  0%   auto  0.0W      0%   0%    
6    37.0c  93.0W   800Mhz  1600Mhz  0%   auto  560.0W    0%   0%    
7    36.0c  N/A     800Mhz  1600Mhz  0%   auto  0.0W      0%   0%    
================================================================================
============================= End of ROCm SMI Log ==============================

Number of processes: 32
Number of threads per proc: 7
Number of iterations 30
Number of proc. per group: 8
Number of groups: 4
Bytes per Type 4
Peer-to-peer count 10000000 ( 40000000 Bytes)
send buffer: 10000000 (0.04 GB) recv buffer: 40000000 (0.16 GB)

HIP VERSION
deviceCount: 1
ENABLE GPU-Aware MPI
warmup time: 2.065718e-02
time: 1.918853e-02
time: 1.863640e-02
time: 1.799292e-02
time: 1.943795e-02
time: 1.840366e-02
time: 1.856728e-02
time: 1.779785e-02
time: 1.793914e-02
time: 1.799937e-02
time: 1.844688e-02
time: 2.042253e-02
time: 1.942493e-02
time: 1.909298e-02
time: 1.884875e-02
time: 1.838981e-02
time: 1.979412e-02
time: 1.782838e-02
time: 1.768154e-02
time: 1.838983e-02
time: 1.806315e-02
time: 1.867967e-02
time: 1.767221e-02
time: 1.836455e-02
time: 1.861823e-02
time: 1.808089e-02
time: 1.878428e-02
time: 1.856667e-02
time: 1.970573e-02
time: 1.926228e-02
totalTime: 5.400805e-01 totalData: 6.96e+00 GB (1.030957e+02 GB/s) --- GPU-Aware MPI
Mon 23 Jan 2023 09:45:40 PM EST
merth@crusher062:~/Bisection_Bandwidth_Tester> 
```
