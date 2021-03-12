#!/bin/bash

# to be run as: python color_mnist_plot.sh 0
# (for cuda device 0)

# CUDA DEVICE ID
DEVICE=$1

#-------------------------------------------------------------------------------#
# Create plot results from trained model
CUDA_VISIBLE_DEVICES=$DEVICE python train_default.py --mode plot --fp-ckpt runs/<name of run>/mnist_cnn.pt --batch-size 128