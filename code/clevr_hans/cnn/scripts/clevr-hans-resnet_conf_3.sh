#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="resnet-clevr-hans-$NUM"
DATASET=clevr-hans-state
OUTPATH="out/clevr-state/$MODEL"

#-------------------------------------------------------------------------------#
# Train on CLEVR_Hans with resnet model

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 0 --mode train

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 1 --mode train

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 2 --mode train

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 3 --mode train

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 4 --mode train


#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 42 --mode train

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 23 --mode train

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 112 --mode train

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --seed 540 --mode train

