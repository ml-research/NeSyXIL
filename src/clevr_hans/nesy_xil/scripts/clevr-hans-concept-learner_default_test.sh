#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="concept-learner-$NUM"
DATASET=clevr-hans-state

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner.py --data-dir $DATA --dataset $DATASET \
--epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
--mode test \
--fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-17-conf_3_seed0/model_epoch52_bestvalloss_0.0167.pth
