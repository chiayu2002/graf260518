#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o output260525_twostage_lowrecon_lazyregSRskip_drift_interp_feature_changegru_gloss.txt				
#PBS -e error20260525_twostage_lowrecon_lazyregSRskip_drift_interp_feature_changegru_gloss.txt				
#PBS -N nerf
cd ~/graf260518_im64										

source ~/.bashrc											
conda activate graf_gpu	

module load cuda-12.4										
python train_twostage.py --config /Data/home/vicky/graf260518_im64/configs/twostage.yaml