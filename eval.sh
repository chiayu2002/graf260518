#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=4
#PBS -l place=shared
#PBS -o output260515_twostage.txt				
#PBS -e error260515_twostage.txt				
#PBS -N eval
cd ~/graf260108_im64										

source ~/.bashrc											
conda activate graf_gpu	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python eval_twostage.py --create_sample --config /Data/home/vicky/graf260108_im64/results/column20260514_twostage_v2_revttur_lowrecon_lazyregSRskip/config.yaml \
                        --checkpoint /Data/home/vicky/graf260108_im64/results/column20260514_twostage_v2_revttur_lowrecon_lazyregSRskip/chkpts/model_00139999.pt --gpu 4 \
                        --all --sr