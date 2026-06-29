#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=1
#PBS -l place=shared
#PBS -o output20260627_rot3.txt				
#PBS -e error20260627_rot3.txt				
#PBS -N eval
cd ~/graf260518_im64										

source ~/.bashrc											
conda activate graf_gpu	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python eval_rotation_twostage.py \
  --config /Data/home/vicky/graf260518_im64/results/column20260608_twostage_film_damage_damage_proxy/config.yaml \
  --checkpoint /Data/home/vicky/graf260518_im64/results/column20260608_twostage_film_damage_damage_proxy/chkpts/model_00159999.pt\
  --rotation \
  --N_frames 72 \
  --gpu 1 --sr