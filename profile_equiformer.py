import torch
from torch_cluster import radius_graph
import numpy as np
from torch_geometric.loader import DataLoader
from functools import partial
import time

import os
from logger import FileLogger
from pathlib import Path

from datasets.pyg.qm9 import QM9

from nets.graph_attention_transformer import TransBlock, NodeEmbeddingNetwork, GaussianRadialBasisLayer, EdgeDegreeEmbeddingNetwork
from e3nn import o3
import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="qm9", help="qm9 or oc20")
args = parser.parse_args()


if args.data == "qm9":
    irreps_node_input = o3.Irreps('128x0e+64x1e+32x2e')
    irreps_node_attr = o3.Irreps('1x0e')
    irreps_edge_attr= o3.Irreps('1x0e+1x1e+1x2e')
    irreps_node_output = '128x0e+64x1e+32x2e'
    number_of_basis = 128
    fc_neurons = [number_of_basis] + [64, 64]
    irreps_head= o3.Irreps('32x0e+16x1e+8x2e')
    num_heads=4
    irreps_mlp_mid = o3.Irreps('384x0e+192x1e+96x2e')
    max_radius = 5.0
    _MAX_ATOM_TYPE = 5
    _AVG_DEGREE = 15.57930850982666
    
    
elif args.data == "md17":
    irreps_node_input = o3.Irreps('128x0e+64x1e+32x2e')
    irreps_node_attr = o3.Irreps('1x0e')
    irreps_edge_attr= o3.Irreps('1x0e+1x1e+1x2e')
    irreps_node_output = o3.Irreps('128x0e+64x1e+32x2e')
    number_of_basis = 128
    fc_neurons = [number_of_basis] + [64, 64]
    irreps_head= o3.Irreps('32x0e+16x1o+8x2e')
    num_heads=8
    irreps_mlp_mid = o3.Irreps('128x0e+64x1e+32x2e')
    max_radius = 5.0
    _MAX_ATOM_TYPE = 64
    _AVG_DEGREE = 15.57930850982666
    
elif args.data == "oc20":
    irreps_node_input = o3.Irreps('256x0e+128x1e')
    irreps_node_attr = o3.Irreps('1x0e')
    irreps_edge_attr= o3.Irreps('1x0e+1x1e')
    irreps_node_output = o3.Irreps('256x0e+128x1e')
    number_of_basis = 128
    fc_neurons = [number_of_basis] + [64, 64]
    irreps_head= o3.Irreps('32x0e+16x1e')
    num_heads=8
    irreps_mlp_mid = o3.Irreps('768x0e+384x1e')
    max_radius = 6.0
    _MAX_ATOM_TYPE = 84
    _AVG_DEGREE = 23.395238876342773
    


model = TransBlock(
    irreps_node_input=irreps_node_input,
    irreps_node_attr=irreps_node_attr,
    irreps_edge_attr=irreps_edge_attr,
    irreps_node_output=irreps_node_output,
    fc_neurons=fc_neurons,
    irreps_head=irreps_head,
    num_heads=num_heads,
    irreps_pre_attn=None,
    rescale_degree=False,
    nonlinear_message=True,
    alpha_drop=0.0,
    proj_drop=0.0,
    drop_path_rate=0.0,
    irreps_mlp_mid=irreps_mlp_mid,
    norm_layer='layer'
)
        

## Load QM9 using DataLoader

dataset = QM9(f'data/{args.data}', 'valid', feature_type='one_hot')
data_loader = DataLoader(dataset, batch_size=128)
data = None
for dt in data_loader:
    data = dt
    break
f_in, pos, batch, node_atom, edge_d_index, edge_d_attr = data.x, data.pos, data.batch, data.z, data.edge_d_index, data.edge_d_attr

# Create Inference Features

edge_src, edge_dst = radius_graph(pos, r=max_radius, batch=batch, max_num_neighbors=1000)
edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
edge_sh = o3.spherical_harmonics(l=irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
node_atom = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
atom_embedding, atom_attr, atom_onehot = NodeEmbeddingNetwork(irreps_node_input, _MAX_ATOM_TYPE)(node_atom)
edge_length = edge_vec.norm(dim=1)
edge_length_embedding = GaussianRadialBasisLayer(number_of_basis, cutoff=max_radius)(edge_length)
edge_degree_embedding = EdgeDegreeEmbeddingNetwork(irreps_node_input, irreps_edge_attr, fc_neurons, _AVG_DEGREE)(
    atom_embedding, edge_sh, 
    edge_length_embedding, edge_src, edge_dst, batch)
node_features = atom_embedding + edge_degree_embedding
node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

model = torch.compile(model, fullgraph=True)

model.forward(node_input=node_features, node_attr=node_attr,
        edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh,
        edge_scalars=edge_length_embedding,
        batch=batch
        )