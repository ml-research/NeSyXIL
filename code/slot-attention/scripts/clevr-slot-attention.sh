#!/bin/bash

# to be called as: python clevr-clot-attention.sh 0 0 /pathtoclevrv1/
# (for cuda device 0 and run 0)

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="slot-attention-clevr-state-$NUM"
DATASET=clevr-state
#-------------------------------------------------------------------------------#
# Train on CLEVR_v1
CUDA_VISIBLE_DEVICES=$DEVICE python train.py --data-dir $DATA --dataset $DATASET --epochs 2000 --name $MODEL --lr 0.0004 --batch-size 512 --n-slots 10 --n-iters-slot-att 3 --n-attr 18
