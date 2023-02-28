'''
    This file modifies `graph_attention_transfomer.py` based on 
    some properties of data in OC20.
    
    1. Handling periodic boundary conditions (PBC)
    2. [TODO] Predicting forces
    3. Using tag (0: sub-surface, 1: surface, 2: adsorbate)
        for extra input information.
    4. Using OC20 registry to register models
    5. Not using one-hot encoded atom type as node attributes since there are much more
        atom types than QM9.
'''

import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric
import math

from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, DropPath
from .graph_attention_transformer import (get_norm_layer, 
    FullyConnectedTensorProductRescaleNorm, 
    FullyConnectedTensorProductRescaleNormSwishGate, 
    FullyConnectedTensorProductRescaleSwishGate,
    DepthwiseTensorProduct, SeparableFCTP,
    Vec2AttnHeads, AttnHeads2Vec,
    GraphAttention, FeedForwardNetwork, 
    TransBlock, 
    NodeEmbeddingNetwork, EdgeDegreeEmbeddingNetwork, ScaledScatter
)
from .gaussian_rbf import GaussianRadialBasisLayer

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)


_RESCALE = True
_USE_BIAS = True

# OC20
_MAX_ATOM_TYPE = 84
_NUM_TAGS = 3   # 0: sub-surface, 1: surface, 2: adsorbate

# Statistics of IS2RE 100K 
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 36.60622024536133

# IS2RE: 100k, max_radius = 5, max_neighbors = 100
_AVG_DEGREE = 23.395238876342773

# Statistics of IS2RE all 
#_AVG_NUM_NODES = 77.74773422429224
#_AVG_DEGREE = 36.5836296081543

       
@registry.register_model("graph_attention_transformer")
class GraphAttentionTransformerOC20(torch.nn.Module):
    '''
        Differences from GraphAttentionTransformer:
            1. Use `otf_graph` and `use_pbc`. `otf_graph` corresponds to whether to 
                build edges on the fly for each inputs. `use_pbc` corresponds to whether
                to consider periodic boundary condition.
            2. Use OC20 registry.
            3. Use `max_neighbors` following models in OC20.
            4. The first two input arguments (e.g., num_atoms and bond_feat_dim) are 
                not used. They are there because of trainer takes extra arguments.
    ''' 
    def __init__(self,
        num_atoms,
        bond_feat_dim,
        num_targets, 
        irreps_node_embedding='256x0e+128x1e', num_layers=6,
        irreps_node_attr='1x0e', use_node_attr=False,
        irreps_sh='1x0e+1x1e',
        max_radius=6.0,
        number_of_basis=128, fc_neurons=[64, 64], 
        use_atom_edge_attr=False, irreps_atom_edge_attr='8x0e',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e', num_heads=8, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='768x0e+384x1e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        use_auxiliary_task=False, auxiliary_head_dropout=True,
        use_attention_head=False, 
        otf_graph=False, use_pbc=True, max_neighbors=50):
        
        
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        
        # for OC20
        self.otf_graph= otf_graph
        self.use_pbc = use_pbc
        self.max_neighbors = max_neighbors
        
        self.use_node_attr = use_node_attr
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        #if not self.use_node_attr:
        #    assert self.irreps_node_attr == o3.Irreps('1x0e')
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.use_atom_edge_attr = use_atom_edge_attr
        self.irreps_atom_edge_attr = o3.Irreps(irreps_atom_edge_attr)
        temp = 0
        if self.use_atom_edge_attr:
            for _, ir in self.irreps_atom_edge_attr:
                assert ir.is_scalar()
            temp = 2 * self.irreps_atom_edge_attr.dim
        self.fc_neurons = [temp + self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.tag_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _NUM_TAGS)
        
        self.attr_embed = None
        if self.use_node_attr:
            self.attr_embed = NodeEmbeddingNetwork(self.irreps_node_attr, _MAX_ATOM_TYPE)
        self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.edge_src_embed = None
        self.edge_dst_embed = None
        if self.use_atom_edge_attr:
            self.edge_src_embed = NodeEmbeddingNetwork(self.irreps_atom_edge_attr, _MAX_ATOM_TYPE)
            self.edge_dst_embed = NodeEmbeddingNetwork(self.irreps_atom_edge_attr, _MAX_ATOM_TYPE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantScalarsDropout(self.irreps_feature, self.out_drop)
        
        self.irreps_feature_scalars = []
        for mul, ir in self.irreps_feature:
            if ir.l == 0 and ir.p == 1:
                self.irreps_feature_scalars.append((mul, (ir.l, ir.p)))
        self.irreps_feature_scalars = o3.Irreps(self.irreps_feature_scalars)
        
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature_scalars, rescale=_RESCALE), 
            Activation(self.irreps_feature_scalars, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature_scalars, o3.Irreps('1x0e')))
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
        self.use_auxiliary_task = use_auxiliary_task
        self.use_attention_head = use_attention_head
        if self.use_auxiliary_task and not self.use_attention_head:
            irreps_out_auxiliary = o3.Irreps('1x1o')
            if o3.Irrep('1o') not in self.irreps_feature:
                irreps_out_auxiliary = o3.Irreps('1x1e')
            self.auxiliary_head = GraphAttention(self.irreps_feature, 
                self.irreps_node_attr, self.irreps_edge_attr, irreps_out_auxiliary,
                self.fc_neurons, 
                self.irreps_head, self.num_heads, self.irreps_pre_attn, 
                self.rescale_degree, self.nonlinear_message,
                alpha_drop=self.alpha_drop if auxiliary_head_dropout else 0.0, 
                proj_drop=0.0)
                
        # Use `GraphAttention` for both energy and force prediction
        if self.use_attention_head:
            irreps_out = o3.Irreps('1x0e')
            if self.use_auxiliary_task:
                irreps_out = irreps_out + irreps_out_auxiliary
            
            self.head = GraphAttention(self.irreps_feature, 
                self.irreps_node_attr, self.irreps_edge_attr, irreps_out,
                self.fc_neurons, 
                self.irreps_head, self.num_heads, self.irreps_pre_attn, 
                self.rescale_degree, self.nonlinear_message,
                alpha_drop=self.alpha_drop if auxiliary_head_dropout else 0.0, 
                proj_drop=0.0)
            self.head_skip_connect = LinearRS(self.irreps_feature, irreps_out)
            
        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer)
            self.blocks.append(blk)
            
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
                          
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        
        return set(no_wd_list)
        

    def _forward_otf_graph(self, data):
        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.max_radius, self.max_neighbors
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors
            return data
        else:
            return data
    
    
    def _forward_use_pbc(self, data):
        pos = data.pos
        batch = data.batch
        if self.use_pbc:
            out = get_pbc_distances(pos,
                data.edge_index,
                data.cell, data.cell_offsets,
                data.neighbors,
                return_offsets=True)
            edge_index = out["edge_index"]
            #dist = out["distances"]
            offsets = out["offsets"]
            edge_src, edge_dst = edge_index
            edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst) + offsets
            dist = edge_vec.norm(dim=1)
        else:
            edge_index = radius_graph(pos, r=self.max_radius, 
                batch=batch, max_num_neighbors=self.max_neighbors)
            edge_src, edge_dst = edge_index
            edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
            dist = edge_vec.norm(dim=1)
            offsets = None
        return edge_index, edge_vec, dist, offsets
        

    def forward(self, data):
        # Following OC20 models
        data = self._forward_otf_graph(data)
        edge_index, edge_vec, edge_length, offsets = self._forward_use_pbc(data)
        batch = data.batch 
        
        edge_src, edge_dst = edge_index[0], edge_index[1]
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        
        # Following Graphoformer, which encodes both atom type and tag
        atomic_numbers = data.atomic_numbers.long()
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(atomic_numbers)
        tags = data.tags.long()
        tag_embedding, _, _ = self.tag_embed(tags)
        
        edge_length_embedding = self.rbf(edge_length, atomic_numbers, 
            edge_src, edge_dst)
        if self.use_atom_edge_attr:
            src_attr, _, _ = self.edge_src_embed(atomic_numbers)
            dst_attr, _, _ = self.edge_dst_embed(atomic_numbers)
            edge_length_embedding = torch.cat((src_attr[edge_src], 
                dst_attr[edge_dst], edge_length_embedding), dim=1)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + tag_embedding + edge_degree_embedding
        
        if self.attr_embed is not None:
            node_attr, _, _ = self.attr_embed(atomic_numbers)
        else:
            node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
            
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
        node_features = self.norm(node_features, batch=batch)
        
        if self.out_dropout is not None:
            outputs = self.out_dropout(node_features)
        else:
            outputs = node_features
            
        # When `self.use_attention_head` is True, 
        # use GraphAttention for energy and force prediction
        if self.use_attention_head:
            outputs_skip = self.head_skip_connect(outputs)
            outputs = self.head(node_input=outputs, 
                node_attr=node_attr, edge_src=edge_src, edge_dst=edge_dst, 
                edge_attr=edge_sh, edge_scalars=edge_length_embedding, 
                batch=batch)
            outputs = outputs + outputs_skip
            if self.use_auxiliary_task:
                outputs_aux = outputs.narrow(1, 1, 3)   # force
                outputs = outputs.narrow(1, 0, 1)       # energy
            outputs = self.scale_scatter(outputs, batch, dim=0)
            if self.use_auxiliary_task:
                return outputs, outputs_aux
            else: 
                return outputs
            
        # FFN for energy prediction
        outputs = self.head(outputs)
        outputs = self.scale_scatter(outputs, batch, dim=0)
        
        # auxiliary IS2RS
        if self.use_auxiliary_task:
            outputs_aux = self.auxiliary_head(node_input=node_features, 
                node_attr=node_attr, edge_src=edge_src, edge_dst=edge_dst, 
                edge_attr=edge_sh, edge_scalars=edge_length_embedding, 
                batch=batch)
                
            return outputs, outputs_aux
        
        return outputs

    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
