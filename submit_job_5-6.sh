#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --job-name=test2
#SBATCH --output=logs/general_output-%j.txt
#SBATCH --error=logs/general_error-%j.txt
#SBATCH --account=mli:cmb-ml
#SBATCH --ntasks=1
#SBATCH --time=0-00:10:00
#SBATCH --gpus=a100:1

# Ensuring the Singularity image is available
ls -l $SCRATCH/tensorflow-linux_v4.sif

# Executing command on GPU 0
singularity exec --nv --bind $SCRATCH:/mnt -e $SCRATCH/tensorflow-linux_v4.sif bash -c '
export CUDA_VISIBLE_DEVICES=0
echo "$(date "+%Y-%m-%d %H:%M:%S") - Starting process on GPU 4"
./commands/commands5.sh
' > logs/output_lv5.txt 2> logs/error_lv5.txt &

# # Executing command on GPU 1
# singularity exec --nv --bind $SCRATCH:/mnt -e $SCRATCH/tensorflow-linux_v4.sif bash -c '
# export CUDA_VISIBLE_DEVICES=1
# echo "$(date "+%Y-%m-%d %H:%M:%S") - Starting process on GPU 5"
# ./commands/commands6.sh
# ' > logs/output_lv6.txt 2> logs/error_lv6.txt &