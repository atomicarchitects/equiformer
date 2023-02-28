#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate th102_cu113_tgconda

python main_qm9.py \
    --output-dir 'models/qm9/compute_stats_radius@5.0' \
    --model-name 'graph_attention_transformer_l2' \
    --input-irreps '5x0e' \
    --target 0 \
    --data-path 'datasets/qm9' \
    --compute-stats \
    --feature-type 'one_hot' \
    --batch-size 256 \
    --radius 5.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp
