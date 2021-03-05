#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="concept-learner-xil-$NUM"
DATASET=clevr-hans-state
OUTPATH="out/clevr-state/$MODEL"

#-------------------------------------------------------------------------------#

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.001 --l2_grads 20 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 1 --mode plot --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-rrr-17-conf_3_seed1/model_epoch37_bestvalloss_0.1620.pth