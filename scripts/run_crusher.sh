#!/bin/bash

#SBATCH -A CHM137_crusher
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=closest

module -t list

date

export OMP_NUM_THREADS=7

export MPICH_OFI_NIC_VERBOSE=2
export MPICH_ENV_DISPLAY=1

#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
for count in 10000000
do
srun -c7  Alltoall $count 10 20 8
done

date
