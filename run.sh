#!/bin/bash

# run.sh

conda create -n reptile_env python=3.5 pip -y
source activate reptile_env
conda install -c anaconda tensorflow-gpu -y
pythons setup.py install

CUDA_VISIBLE_DEVICES=0 python -u run_omniglot.py \
    --shots 1 \
    --inner-batch 25 \
    --inner-iters 3 \
    --meta-step 1 \
    --meta-batch 10 \
    --meta-iters 100000 \
    --eval-batch 25 \
    --eval-iters 5 \
    --learning-rate 0.001 \
    --meta-step-final 0 \
    --train-shots 15 \
    --checkpoint ckpt_o15t \
    --transductive
