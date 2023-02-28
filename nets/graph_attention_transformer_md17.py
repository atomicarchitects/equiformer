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
from .fast_layer_norm import EquivariantLayerNormFast
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


_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 64 # Set to some large value

# Statistics of QM9 with cutoff radius = 5
# For simplicity, use the same statistics for MD17
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666
      

class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


# https://github.com/torchmd/torchmd-net/blob/main/torchmdnet/models/utils.py#L111
class ExpNormalSmearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class GraphAttentionTransformerMD17(torch.nn.Module):
    def __init__(self,
        irreps_in='64x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        use_attn_head=False, 
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.use_attn_head = use_attn_head
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
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
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        if self.use_attn_head:
            self.head = GraphAttention(irreps_node_input=self.irreps_feature, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=o3.Irreps('1x0e'),
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop)
        else:
            self.head = torch.nn.Sequential(
                LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
                Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
                LinearRS(self.irreps_feature, o3.Irreps('1x0e'), rescale=_RESCALE)) 
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
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
    def forward(self, node_atom, pos, batch):

        pos = pos.requires_grad_(True)

        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch,
            max_num_neighbors=1000)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        if self.use_attn_head:
            outputs = self.head(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        else:
            outputs = self.head(node_features)
        outputs = self.scale_scatter(outputs, batch, dim=0)
        
        if self.scale is not None:
            outputs = self.scale * outputs

        energy = outputs
        # https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py#L321-L328
        forces = -1 * (
                    torch.autograd.grad(
                        energy,
                        pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True,
                    )[0]
                )

        return energy, forces


@register_model
def graph_attention_transformer_l2_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_l2_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_l2_e3_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+32x0o+32x1e+32x1o+16x2e+16x2o', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+8x0o+8x1e+8x1o+4x2e+4x2o', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+96x0o+96x1e+96x1o+48x2e+48x2o',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_bessel_l2_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model



@register_model
def graph_attention_transformer_nonlinear_exp_l2_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='exp',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_exp_l3_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+64x2e+32x3e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e+1x3e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='exp',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+16x2e+8x3e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+192x2e+96x3e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_attn_exp_l3_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+64x2e+32x3e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e+1x3e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='exp',
        irreps_feature='128x0e+64x1e+64x2e+32x3e',
        irreps_head='32x0e+16x1e+16x2e+8x3e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+192x2e+96x3e',
        use_attn_head=True, 
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_exp_l3_e3_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x0o+32x1e+32x1o+32x2e+32x2o+16x3e+16x3o', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e+1x3o',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='exp',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x0o+8x1e+8x1o+8x2e+8x2o+4x3e+4x3o', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x0o+96x1e+96x1o+96x2e+96x2o+48x3e+48x3o',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_bessel_l3_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+64x2e+32x3e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e+1x3e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+16x2e+8x3e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+192x2e+96x3e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_bessel_l3_e3_md17(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x0o+32x1e+32x1o+32x2e+32x2o+16x3e+16x3o', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e+1x3o',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x0o+8x1e+8x1o+8x2e+8x2o+4x3e+4x3o', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x0o+96x1e+96x1o+96x2e+96x2o+48x3e+48x3o',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model