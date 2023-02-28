#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate th102_cu113_tgconda

python main_qm9.py \
    --output-dir 'models/qm9/equiformer/se_l2/target@6/' \
    --model-name 'graph_attention_transformer_nonlinear_bessel_l2' \
    --input-irreps '5x0e' \
    --target 6 \
    --data-path 'datasets/qm9' \
    --feature-type 'one_hot' \
    --batch-size 128 \
    --radius 5.0 \
    --num-basis 8 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp
