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
    --config-yml 'oc20/configs/is2re/all/graph_attention_transformer/l1_256_blocks@18_nonlinear_aux_g@4_local.yml' \
    --run-dir 'models/oc20/is2re/graph_attention_transformer/l1_blocks@18_all/bs@32_lr@5e-4_warmup-epochs@2_wd@1e-3_max-radus@5.0_num-layers@18_alpha-drop@0.2_drop-path@0.05_gaussian-rbf@128_embed-rewieght_edge-deg-embed-exp-dw-proj-scaled-scatter_attn-ffn-no-concat_head-scaled-scatter_nonlinear_aux@15.0_g@4' \
    --print-every 200
