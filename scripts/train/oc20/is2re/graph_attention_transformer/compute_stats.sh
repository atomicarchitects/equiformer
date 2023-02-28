#!/bin/bash

# Loading the required module
#source /etc/profile
#module load anaconda/2021a

#export PYTHONNOUSERSITE=True    # prevent using packages from base
#source activate th102_cu113_tgconda

python main_oc20.py \
    --mode train \
    --config-yml 'oc20/configs/is2re/100k/graph_attention_transformer/compute_stats.yml' \
    --run-dir 'models/oc20/is2re/graph_attention_transformer/l1_100k/compute_stats_max-radius@5.0_max-neighbors@50' \
    --print-every 200 \
    --cpu