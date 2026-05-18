#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=1
#PBS -l place=shared
#PBS -o output260516_twostage_v2_revttur_lowrecon_lazyregSRskip_drift.txt				
#PBS -e error20260516_twostage_v2_revttur_lowrecon_lazyregSRskip_drift.txt				
#PBS -N nerf
cd ~/graf260108_im64										

source ~/.bashrc											
conda activate graf_gpu	

module load cuda-12.4										
python train_twostage.py --config /Data/home/vicky/graf260108_im64/configs/twostage.yaml