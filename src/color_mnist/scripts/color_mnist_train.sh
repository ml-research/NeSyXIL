#!/bin/bash

# to be run as: python color_mnist_train.sh 0
# (for cuda device 0)

# CUDA DEVICE ID
DEVICE=$1

#-------------------------------------------------------------------------------#
# Train on ColorMNIST
CUDA_VISIBLE_DEVICES=$DEVICE python train.py --mode train