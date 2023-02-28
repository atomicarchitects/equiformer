#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate th102_cu113_tgconda

python main_md17.py \
    --output-dir 'models/md17/equiformer/se_l3/target@benzene/lr@1e-4_bs@5_wd@1e-6_epochs@2000_w-f2e@80_dropout@0.0_exp@32_l2mae-loss' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l3_md17' \
    --input-irreps '64x0e' \
    --target 'benzene' \
    --data-path 'datasets/md17' \
    --epochs 2000 \
    --lr 1e-4 \
    --batch-size 5 \
    --eval-batch-size 16 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 80
