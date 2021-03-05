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

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 20 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-rrr-17-conf_3_seed0/model_epoch48_bestvalloss_0.3832.pth

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 20 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 1 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-rrr-17-conf_3_seed1/model_epoch37_bestvalloss_0.1620.pth

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 20 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 2 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-rrr-17-conf_3_seed2/model_epoch43_bestvalloss_0.0845.pth

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 20 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 3 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-rrr-17-conf_3_seed3/model_epoch48_bestvalloss_0.2247.pth

python train_clevr_hans_concept_learner_xil.py --data-dir $DATA --dataset $DATASET --epochs 50 --name $MODEL --lr 0.001 --l2_grads 20 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 4 --mode test --fp-ckpt runs/conf_3/slot-attention-clevr-state-set-transformer-rrr-17-conf_3_seed4/model_epoch41_bestvalloss_0.0755.pth
