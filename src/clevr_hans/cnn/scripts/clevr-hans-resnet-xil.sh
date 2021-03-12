#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="resnet-clevr-hans-lexi-$NUM"
DATASET=clevr-hans-state

#-------------------------------------------------------------------------------#
# Train on CLEVR_Hans with resnet model

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 \
--name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 0 --mode train
