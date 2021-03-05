#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="concept-learner-$NUM"
DATASET=clevr-hans-state
OUTPATH="out/clevr-state/$MODEL-$ITER"

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-17-conf_3_seed0/model_epoch52_bestvalloss_0.0167.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 1 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-17-conf_3_seed1/model_epoch29_bestvalloss_0.0251.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 2 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-17-conf_3_seed2/model_epoch48_bestvalloss_0.0388.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 3 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-17-conf_3_seed3/model_epoch31_bestvalloss_0.0377.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_concept_learner.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 4 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-17-conf_3_seed4/model_epoch36_bestvalloss_0.0352.pth
