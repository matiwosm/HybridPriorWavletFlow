#!/bin/bash

#SBATCH --partition=ada
#SBATCH --job-name=nyq_kappa_noise%a
#SBATCH --output=logs/nyquest_noise_kappa_cib%a_log.txt
#SBATCH --error=logs/nyquest_noise_kappa_cib%a_error.txt
#SBATCH --account=kipac
#SBATCH --ntasks=1
#SBATCH --time=4-23:59:59
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --array=1-4

# The array will spawn four separate jobs with SLURM_ARRAY_TASK_ID = 1,2,3,4
# Each job will pick a different level based on $SLURM_ARRAY_TASK_ID.

source /sdf/group/kipac/sw/conda/bin/activate 
conda activate /sdf/group/kipac/users/mati/torchcfm
cd HybridPriorWavletFlow_works_well
pwd
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
echo "Training level: $SLURM_ARRAY_TASK_ID"

# Run the training command, using the array task ID as the level
python train.py --level $SLURM_ARRAY_TASK_ID --config configs/kappa_cib_hcc_prior_noised_kappa_nyquest_noise.py

wait