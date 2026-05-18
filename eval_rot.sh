#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=3
#PBS -l place=shared
#PBS -o output2604122_column20260417_gru_disc_Dbn_lrg3_aux_sampling_speed_rot_2.txt				
#PBS -e error260422_column20260417_gru_disc_Dbn_lrg3_aux_sampling_speed_rot_2.txt				
#PBS -N eval
cd ~/graf260108_im64										

source ~/.bashrc											
conda activate graf_gpu	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python eval_rotation.py \
  --config /Data/home/vicky/graf260108_im64/results/column20260417_gru_disc_Dbn_lrg3_aux_sampling_speed/config.yaml \
  --checkpoint /Data/home/vicky/graf260108_im64/results/column20260417_gru_disc_Dbn_lrg3_aux_sampling_speed/chkpts/model_00209999.pt \
  --interpolate_hs --interp_steps 11 --gpu 3