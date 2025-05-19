#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --job-name=CC_0.01_noised_kappa_256_cc_prior
#SBATCH --output=logs/noised_kappa_0.01_256_cc_prior%a_log.txt
#SBATCH --error=logs/noised_kappa_0.01_256_cc_prior%a_error.txt
#SBATCH --account=mli:cmb-ml
#SBATCH --time=5-23:59:59
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --array=[8]


# The array will spawn four separate jobs with SLURM_ARRAY_TASK_ID = 1,2,3,4
# Each job will pick a different level based on $SLURM_ARRAY_TASK_ID.

source /sdf/group/kipac/sw/conda/bin/activate 
conda activate /sdf/group/kipac/users/mati/torchcfm
cd HybridPriorWavletFlow_works_well
pwd
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
echo "Training level: $SLURM_ARRAY_TASK_ID"

# Run the training command, using the array task ID as the level
python train.py --level $SLURM_ARRAY_TASK_ID --config configs/CC_prior_best_model_256x256_all_levels_kap_noise_0.01.py

wait