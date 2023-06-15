#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.error
#SBATCH --gres=gpu:2
#SBATCH --job-name=cad_cifar

module load anaconda/3.7

source activate /home/kong_04/project/envs/cad

cd $SLURM_SUBMIT_DIR


MODEL=ResNet12
METHOD=cad # lr=0.003 
#
# N=1 300 epochs since we use 200 episodes per epoch (200 x 300 tasks in total), (see datamgr.py)
# N=5 200 epochs since we use '''''''' (200 x 200 tasks in total)
#

for DATA in  cifarfs #tieredImagenet CUB cifarfs cross
do
    NSHOT=1
    python train.py --dataset $DATA --model $MODEL --method $METHOD --n_shot $NSHOT --train_aug --stop_epoch 300 --optim adam --lr 0.003 --milestones 300
    # #save features
    python save_features.py --dataset $DATA --model $MODEL --method $METHOD --n_shot $NSHOT --split novel --train_aug --save_iter -1
    # #test model
    python test.py --dataset $DATA --model $MODEL --method $METHOD --n_shot $NSHOT --split novel --train_aug --save_iter -1

    NSHOT=5
    python train.py --dataset $DATA --model $MODEL --method $METHOD --n_shot $NSHOT --train_aug --stop_epoch 200 --optim adam --lr 0.003 --milestones 200
    # #save features
    python save_features.py --dataset $DATA --model $MODEL --method $METHOD --n_shot $NSHOT --split novel --train_aug --save_iter -1
    # #test model
    python test.py --dataset $DATA --model $MODEL --method $METHOD --n_shot $NSHOT --split novel --train_aug --save_iter -1 
done
