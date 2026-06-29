#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=1
#PBS -l place=shared
#PBS -o output260627_twostage.txt				
#PBS -e error260627_twostage.txt				
#PBS -N eval
cd ~/graf260518_im64										

source ~/.bashrc											
conda activate graf_gpu	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python eval_twostage.py --config /Data/home/vicky/graf260518_im64/results/column20260608_twostage_film_damage_damage_proxy/config.yaml \
                        --checkpoint /Data/home/vicky/graf260518_im64/results/column20260608_twostage_film_damage_damage_proxy/chkpts/model_00159999.pt --gpu 1 \
                        --create_sample --sr