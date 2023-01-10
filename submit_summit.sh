
module load gcc
module load cuda
module load job-step-viewer

bsub -q debug -alloc_flags gpudefault -W 00:30 -nnodes 8 -P CHM137 -env "all,LSF_CPU_ISOLATION=on" -Is /bin/bash

