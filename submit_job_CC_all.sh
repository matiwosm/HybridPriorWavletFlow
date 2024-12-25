#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --job-name=HCC_prior_best_model_256x256_all_levels%a
#SBATCH --output=logs/HCC_prior_best_model_256x256_all_levels%a_log.txt
#SBATCH --error=logs/HCC_prior_best_model_256x256_all_levels%a_error.txt
#SBATCH --account=mli:cmb-ml
#SBATCH --ntasks=1
#SBATCH --time=3-23:59:59
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --array=7-8

# The array will spawn four separate jobs with SLURM_ARRAY_TASK_ID = 1,2,3,4
# Each job will pick a different level based on $SLURM_ARRAY_TASK_ID.

source /sdf/group/kipac/sw/conda/bin/activate 
conda activate /sdf/group/kipac/users/mati/torchcfm
cd HybridPriorWavletFlow
pwd
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
echo "Training level: $SLURM_ARRAY_TASK_ID"

# Run the training command, using the array task ID as the level
python train.py --level $SLURM_ARRAY_TASK_ID --config configs/HCC_prior_best_model_256x256_all_levels.py

wait