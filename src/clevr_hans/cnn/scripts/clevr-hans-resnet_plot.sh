#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="resnet-clevr-hans-$NUM"
DATASET=clevr-hans-state

#-------------------------------------------------------------------------------#
# Train on CLEVR_Hans with resnet model

# without obj coordinates
CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL \
--lr 0.0001 --batch-size 500 --seed 0 --mode plot \
--fp-ckpt runs/conf_3/resnet-clevr-hans-17-conf_3_seed0/model_epoch56_bestvalloss_0.0132.pth
