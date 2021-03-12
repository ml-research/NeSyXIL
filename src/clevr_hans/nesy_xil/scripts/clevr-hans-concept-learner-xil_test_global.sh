#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="concept-learner-xil-$NUM"
DATASET=clevr-hans-state

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

python train_clevr_hans_concept_learner_xil_globalrule.py --data-dir $DATA --dataset $DATASET --epochs 50 \
--name $MODEL --lr 0.001 --l2_grads 20 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
--mode test \
--fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-rrr-17-conf_3_seed0/model_epoch48_bestvalloss_0.3832.pth
