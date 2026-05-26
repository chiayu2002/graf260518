#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o output260510_recon_per1.0_rec0.5_scale_anneal_label0.5_mat_TTUR_nsample128.txt				
#PBS -e error260510_recon_per1.0_rec0.5_scale_anneal_label0.5_mat_TTUR_nsample128.txt				
#PBS -N nerf
cd ~/graf260108_im64										

source ~/.bashrc											
conda activate graf_gpu	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python train.py --config /Data/home/vicky/graf260108_im64/configs/recon.yaml 