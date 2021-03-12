#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="resnet-clevr-hans-lexi-$NUM"
DATASET=clevr-hans-state
OUTPATH="out/clevr-state/$MODEL"

#-------------------------------------------------------------------------------#
# Train on CLEVR_Hans with resnet model

# without obj coordinates
CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 500 --l2_grads 10 --seed 0 --mode plot --fp-ckpt runs/conf_3/resnet-clevr-hans-lexi-17-conf_3_seed0/model_epoch86_bestvalloss_0.0122.pth
