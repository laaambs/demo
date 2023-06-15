#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.error
#SBATCH --gres=gpu:1
#SBATCH --job-name=demo

module load anaconda/3.7

source activate /home/kong_04/project/envs/cad

cd $SLURM_SUBMIT_DIR


python train.py
