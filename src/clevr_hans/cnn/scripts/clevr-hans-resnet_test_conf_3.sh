#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="resnet-clevr-hans-$NUM"
DATASET=clevr-hans-state
OUTPATH="out/clevr-state/$MODEL"

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 0 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-17-conf_3_seed0/model_epoch56_bestvalloss_0.0132.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 1 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-17-conf_3_seed1/model_epoch99_bestvalloss_0.0120.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 2 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-17-conf_3_seed2/model_epoch93_bestvalloss_0.0189.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 3 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-17-conf_3_seed3/model_epoch99_bestvalloss_0.0107.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 4 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-17-conf_3_seed4/model_epoch99_bestvalloss_0.0110.pth


#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 23 --mode test --fp-ckpt runs/resnet-clevr-hans-17-conf_3_seed23/model_epoch92_besttrainloss_0.0000.pth

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 42 --mode test --fp-ckpt runs/resnet-clevr-hans-17-conf_3_seed42/model_epoch99_besttrainloss_0.0000.pth

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 112 --mode test --fp-ckpt runs/resnet-clevr-hans-17-conf_3_seed112/model_epoch92_besttrainloss_0.0000.pth

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 540 --mode test --fp-ckpt runs/resnet-clevr-hans-17-conf_3_seed540/model_epoch98_besttrainloss_0.0000.pth

