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

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-xil-17-conf_3_seed0/model_epoch47_bestvalloss_0.7580.pth

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 1 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-xil-17-conf_3_seed1/model_epoch32_bestvalloss_0.7523.pth

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 2 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-xil-17-conf_3_seed2/model_epoch45_bestvalloss_0.7417.pth

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 3 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-xil-17-conf_3_seed3/model_epoch23_bestvalloss_0.7400.pth

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 4 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-xil-17-conf_3_seed4/model_epoch42_bestvalloss_0.7654.pth
