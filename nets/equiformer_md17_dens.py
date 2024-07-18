from locale import normalize
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric
import math

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .gaussian_rbf import GaussianRadialBasisLayer

# for bessel radial basis
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis

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
from .graph_attention_transformer_md17 import (
    CosineCutoff, 
    ExpNormalSmearing
)


_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 64 # Set to some large value

# Statistics of QM9 with cutoff radius = 5
# For simplicity, use the same statistics for MD17
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666
      

class Equiformer_MD17_DeNS(torch.nn.Module):
    def __init__(self,
        irreps_in='64x0e',
        irreps_equivariant_inputs='1x0e+1x1e+1x2e',     # for encoding forces during denoising positions
        irreps_node_embedding='128x0e+64x1e+32x2e', 
        num_layers=6,
        irreps_node_attr='1x0e', 
        irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=32, 
        basis_type='exp', 
        fc_neurons=[64, 64], 
        irreps_feature='512x0e+256x1e+128x2e',          # increase numbers of channels by 4 times
        irreps_head='32x0e+16x1o+8x2e', 
        num_heads=4, 
        irreps_pre_attn='128x0e+64x1e+32x2e',
        rescale_degree=False, 
        nonlinear_message=True,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.0, 
        proj_drop=0.0, 
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, 
        std=None, 
        scale=None, 
        atomref=None,
        use_force_encoding=True,                        # for ablation study
    ):
        
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)
        self.use_force_encoding = use_force_encoding
        
        self.irreps_node_attr   = o3.Irreps(irreps_node_attr)
        self.irreps_node_input  = o3.Irreps(irreps_in)
        self.irreps_node_equivariant_inputs = o3.Irreps(irreps_equivariant_inputs)  # for encoding forces during denoising positions
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers     = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head    = o3.Irreps(irreps_head)
        self.num_heads      = num_heads
        self.irreps_pre_attn    = irreps_pre_attn
        self.rescale_degree     = rescale_degree
        self.nonlinear_message  = nonlinear_message
        self.irreps_mlp_mid     = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, 
                rbf={'name': 'spherical_bessel'})
        elif self.basis_type == 'exp':
            self.rbf = ExpNormalSmearing(cutoff_lower=0.0, cutoff_upper=self.max_radius, 
                num_rbf=self.number_of_basis, trainable=False)
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        # for denoising positions
        self.force_embed = LinearRS(self.irreps_node_equivariant_inputs, self.irreps_node_embedding, rescale=_RESCALE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        
        # energy and forces prediction
        irreps_feature_scalars = []
        for mul, ir in self.irreps_feature:
            if (ir.l == 0) and (ir.p == 1):
                irreps_feature_scalars.append((mul, ir))
        irreps_feature_scalars = o3.Irreps(irreps_feature_scalars)
        self.energy_head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, irreps_feature_scalars, rescale=_RESCALE), 
            Activation(irreps_feature_scalars, acts=[torch.nn.SiLU()]),
            LinearRS(irreps_feature_scalars, o3.Irreps('1x0e'), rescale=_RESCALE)
        ) 
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
        # denoising position
        #   check for parity
        irreps_denoising_pos_outputs = o3.Irreps('1x1e') if o3.Irrep('1e') in self.irreps_node_equivariant_inputs else o3.Irreps('1x1o')
        self.denoising_pos_head = GraphAttention(
            irreps_node_input=self.irreps_feature, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr, 
            irreps_node_output=irreps_denoising_pos_outputs,
            fc_neurons=self.fc_neurons, 
            irreps_head=self.irreps_head, 
            num_heads=self.num_heads, 
            irreps_pre_attn=self.irreps_pre_attn, 
            rescale_degree=self.rescale_degree,
            nonlinear_message=self.nonlinear_message,
            alpha_drop=self.alpha_drop, 
            proj_drop=self.proj_drop
        )
        
        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding, 
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
                norm_layer=self.norm_layer
            )
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
                or isinstance(module, GaussianRadialBasisLayer) 
                or isinstance(module, RadialBasis)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
        

    # the gradient of energy is following the implementation here:
    # https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py#L186
    @torch.enable_grad()
    def forward(self, data):

        node_atom   = data.z
        pos         = data.pos
        batch       = data.batch 

        pos = pos.requires_grad_(True)

        edge_src, edge_dst = radius_graph(
            pos, 
            r=self.max_radius, 
            batch=batch,
            max_num_neighbors=1000
        )
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec, 
            normalize=True, 
            normalization='component'
        )
        
        atom_embedding, _, _ = self.atom_embed(node_atom)
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding, 
            edge_sh, 
            edge_length_embedding, 
            edge_src, 
            edge_dst, 
            batch
        )
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        # encoding forces during denoising positions
        if hasattr(data, 'force') and self.use_force_encoding:
            force_data = data.force
            force_sh = o3.spherical_harmonics(
                l=self.irreps_node_equivariant_inputs,
                x=force_data,
                normalize=True,
                normalization='component'
            )
            force_sh[(~data.noise_mask)] *= 0
            force_norm = force_data.norm(dim=1, keepdim=True)
            force_norm = force_norm / math.sqrt(3.0)
            force_sh = force_sh * force_norm
        else:
            force_sh = torch.zeros(
                (node_features.shape[0], self.irreps_node_equivariant_inputs.dim), 
                device=node_features.device, 
                dtype=node_features.dtype
            )
        force_embedding = self.force_embed(force_sh)
        node_features = node_features + force_embedding
        
        for blk in self.blocks:
            node_features = blk(
                node_input=node_features, 
                node_attr=node_attr, 
                edge_src=edge_src, 
                edge_dst=edge_dst, 
                edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch
            )
        
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)

        # energy prediction
        energy_outputs = self.energy_head(node_features)
        #   not predict denoising energy if `self.use_force_encoding` is False
        if hasattr(data, 'denoising_mask') and not self.use_force_encoding:
            energy_outputs[data.denoising_mask] *= 0
        energy_outputs = self.scale_scatter(energy_outputs, batch, dim=0)
        if self.scale is not None:
            energy_outputs = self.scale * energy_outputs

        # force prediction
        # https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py#L321-L328
        forces_outputs = -1 * (
            torch.autograd.grad(
                energy_outputs,
                pos,
                grad_outputs=torch.ones_like(energy_outputs),
                create_graph=True,
            )[0]
        )
        
        if hasattr(data, 'noise_mask'):
            # denoising positions
            denoising_pos_outputs = self.denoising_pos_head(
                node_input=node_features, 
                node_attr=node_attr, 
                edge_src=edge_src, 
                edge_dst=edge_dst, 
                edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch
            )

            outputs_dy = torch.zeros_like(forces_outputs)
            if hasattr(data, 'noise_mask'):
                outputs_dy[(~data.noise_mask)] = forces_outputs[(~data.noise_mask)]
                outputs_dy[data.noise_mask] = denoising_pos_outputs[data.noise_mask]

                # not predict denoising energy if `self.use_force_encoding` is False
                if not self.use_force_encoding:
                    outputs_dy[data.denoising_pos_mask] *= 0

            return energy_outputs, outputs_dy
    
        else:
            return energy_outputs, forces_outputs
        

@register_model
def equiformer_md17_dens(**kwargs):
    return Equiformer_MD17_DeNS(**kwargs)