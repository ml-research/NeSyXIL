#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="resnet-clevr-hans-lexi-$NUM"
DATASET=clevr-hans-state
OUTPATH="out/clevr-state/$MODEL"

#-------------------------------------------------------------------------------#
# CLEVR_Hans3

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 0 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-lexi-17-conf_3_seed0/model_epoch86_bestvalloss_0.0122.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 1 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-lexi-17-conf_3_seed1/model_epoch91_bestvalloss_0.0142.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 2 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-lexi-17-conf_3_seed2/model_epoch83_bestvalloss_0.0101.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 3 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-lexi-17-conf_3_seed3/model_epoch92_bestvalloss_0.0119.pth

CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 4 --mode test --fp-ckpt runs/conf_3/resnet-clevr-hans-lexi-17-conf_3_seed4/model_epoch74_bestvalloss_0.0181.pth


#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 23 --mode test --fp-ckpt runs/resnet-clevr-hans-lexi-17-conf_3_seed23/model_epoch90_besttrainloss_0.0020.pth

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 42 --mode test --fp-ckpt runs/resnet-clevr-hans-lexi-17-conf_3_seed42/model_epoch87_besttrainloss_0.0019.pth

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 112 --mode test --fp-ckpt runs/resnet-clevr-hans-lexi-17-conf_3_seed112/model_epoch86_besttrainloss_0.0018.pth

#CUDA_VISIBLE_DEVICES=$DEVICE python train_resnet34_lexi.py --data-dir $DATA --dataset $DATASET --epochs 100 --name $MODEL --lr 0.0001 --batch-size 64 --l2_grads 10 --seed 540 --mode test --fp-ckpt runs/resnet-clevr-hans-lexi-17-conf_3_seed540/model_epoch96_besttrainloss_0.0018.pth

