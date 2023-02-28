#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate th90_cu111

python -u -m torch.distributed.launch --nproc_per_node=4 main_oc20.py \
    --distributed \
    --num-gpus 4 \
    --mode train \
    --config-yml 'oc20/configs/is2re/all/graph_attention_transformer/l1_256_g@4_local.yml' \
    --run-dir 'models/oc20/is2re/graph_attention_transformer/l1_all/bs@64_lr@4e-4_warmup-epochs@2_wd@1e-2_alpha-drop@0.2_max-radius@5.0_max-neighbors@500_gaussian-rbf@128_embed-rewieght_edge-deg-embed-exp-dw-proj-scaled-scatter_attn-ffn-no-concat_head-scaled-scatterg@2' \
    --print-every 200
