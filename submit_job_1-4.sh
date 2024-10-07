#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --job-name=oneto4kaponly
#SBATCH --output=logs/general_output-%j.txt
#SBATCH --error=logs/general_error-%j.txt
#SBATCH --account=mli:cmb-ml
#SBATCH --ntasks=1
#SBATCH --time=0-23:59:00
#SBATCH --gpus=a100:4
#SBATCH --mem=960G

# Ensuring the Singularity image is available
ls -l $SCRATCH/tensorflow-linux_v4.sif

# Executing command on GPU 0
singularity exec --nv --bind /sdf/group/kipac/users/mati:/mnt -e $SCRATCH/tensorflow-linux_v4.sif bash -c '
export CUDA_VISIBLE_DEVICES=0
echo "$(date "+%Y-%m-%d %H:%M:%S") - Starting process on GPU 0"
./commands/commands1.sh
' > logs/output_lv1.txt 2> logs/error_lv1.txt &

# Executing command on GPU 1
singularity exec --nv --bind /sdf/group/kipac/users/mati:/mnt -e $SCRATCH/tensorflow-linux_v4.sif bash -c '
export CUDA_VISIBLE_DEVICES=1
echo "$(date "+%Y-%m-%d %H:%M:%S") - Starting process on GPU 1"
./commands/commands2.sh
' > logs/output_lv2.txt 2> logs/error_lv2.txt &

# Executing command on GPU 2
singularity exec --nv --bind /sdf/group/kipac/users/mati:/mnt -e $SCRATCH/tensorflow-linux_v4.sif bash -c '
export CUDA_VISIBLE_DEVICES=2
echo "$(date "+%Y-%m-%d %H:%M:%S") - Starting process on GPU 2"
./commands/commands3.sh
' > logs/output_lv3.txt 2> logs/error_lv3.txt &

# Executing command on GPU 3
singularity exec --nv --bind /sdf/group/kipac/users/mati:/mnt -e $SCRATCH/tensorflow-linux_v4.sif bash -c '
export CUDA_VISIBLE_DEVICES=3
echo "$(date "+%Y-%m-%d %H:%M:%S") - Starting process on GPU 3"
./commands/commands4.sh
' > logs/output_lv4.txt 2> logs/error_lv4.txt &


# Wait for all background jobs to finish
wait