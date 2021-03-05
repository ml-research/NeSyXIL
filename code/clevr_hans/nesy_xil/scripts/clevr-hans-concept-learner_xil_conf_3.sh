#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="concept-learner-xil-$NUM"
DATASET=clevr-hans-state
OUTPATH="out/clevr-state/$MODEL"

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 --mode train

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 1 --mode train

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 2 --mode train

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 3 --mode train

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 4 --mode train
