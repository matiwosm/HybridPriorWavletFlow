#!/bin/bash
# commands.sh

source /opt/conda/bin/activate
cd $HOME/WaveletFlowPytorch_corr
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
conda activate pytorch
python train.py --level 2
