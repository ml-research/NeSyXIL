#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="resnet-clevr-hans-$NUM"
DATASET=clevr-hans-state

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL \
--lr 0.0001 --batch-size 64 --seed 0 --mode test \
--fp-ckpt runs/conf_3/resnet-clevr-hans-17-conf_3_seed0/model_epoch56_bestvalloss_0.0132.pth
