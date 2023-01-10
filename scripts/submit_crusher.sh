
module reset
module load rocm
module load craype-accel-amd-gfx90a

export MPICH_GPU_SUPPORT_ENABLED=1

export MPICH_OFI_NIC_VERBOSE=2
export MPICH_ENV_DISPLAY=1

salloc -A CHM137_crusher -t 01:00:00 -N 4 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest
