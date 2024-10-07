#!/bin/bash
# commands.sh

source /opt/conda/bin/activate
cd $HOME/WaveletFlowPytorch_corr
conda activate pytorch
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
python train.py --level 4
