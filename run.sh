#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 \
python3 train.py \
--log_file ./checkpoint/train.logs \
--tensorboard ./checkpoint/tensorboard \
--model_path ./checkpoint/model.pkl \
--model shufflenet_corr \
--PDB_mode \