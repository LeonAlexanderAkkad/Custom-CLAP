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

python -m stage2_training $HOME/Custom-CLAP/clap/configs/clap_htsat-tiny_roberta-large_vTestDistillation.yml $HOME/Custom-CLAP/clap/checkpoints/clap_htsat-tiny_roberta-large_vStage2_Clotho.ckpt Clotho --dataset_paths $DATA/clotho --use_wandb --project CLAP-Training --name "Stage 2 htsat-tiny_roberta-large" --distill_weight 1.0 --distill_ckpt_paths $HOME/Custom-CLAP/clap/checkpoints/clap_htsat-tiny_roberta-large_vStage1_Clotho_epoch20.ckpt $HOME/Custom-CLAP/clap/checkpoints/clap_htsat-tiny_roberta-large_vStage1_Clotho.ckpt $HOME/Custom-CLAP/clap/checkpoints/clap_htsat-base_roberta-large_vStage1_Clotho_epoch20.ckpt --distill_config_paths $HOME/Custom-CLAP/clap/configs/clap_htsat-tiny_roberta-large_vTestDistillation.yml $HOME/Custom-CLAP/clap/configs/clap_htsat-tiny_roberta-large_vTestDistillation.yml $HOME/Custom-CLAP/clap/configs/clap_htsat-base_roberta-large_vTestDistillation.yml --parameters $HOME/Custom-CLAP/clap/checkpoints/clap_htsat-tiny_roberta-large_vStage1_Clotho_epoch20.ckpt 