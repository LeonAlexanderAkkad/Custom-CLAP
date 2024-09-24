#!/bin/bash
#
#  usage: sbatch ./gpu_test.scrpt
#
#SBATCH -J clasp
# #SBATCH -N 1                           #use -N only if you use both GPUs on the nodes, otherwise leave this line out
#SBATCH --partition zen2_0256_a40x2
#SBATCH --qos zen2_0256_a40x2            
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8

cd $HOME/Custom-CLAP/examples

CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

conda activate custom-clap

python -m stage1_training.py $HOME/Custom-CLAP/clap/configs/clap_cnn14_distilroberta-base_vTestDistillation.yml $HOME/Custom-CLAP/clap/checkpoints/clap_cnn14_distilroberta-bas_vStage1_Clotho_Test.ckpt Clotho --dataset_paths $HOME/Custom-CLAP/clap/datasets/clotho --use_wandb --project CLAP-Training --name "Stage 1 new evaluation"