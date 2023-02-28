#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate th102_cu113_tgconda

python main_oc20.py \
    --mode train \
    --config-yml 'oc20/configs/is2re/10k/graph_attention_transformer/l1_256.yml' \
    --run-dir 'models/oc20/is2re/graph_attention_transformer/l1_10k/wd@2e-3_adamw' \
    --print-every 50