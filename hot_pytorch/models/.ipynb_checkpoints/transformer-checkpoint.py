from typing import Union, Optional, Tuple

import torch.nn as nn
import torch
from torch.autograd import grad
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import contains_self_loops, add_self_loops
from torch_geometric.nn.models.schnet import qm9_target_dict


from .dense.attn import SelfAttn as D_SelfAttn
from .dense.kernelattn import KernelSelfAttn as D_KernelSelfAttn
from .dense.f import Apply as D_Apply, Add as D_Add
from .dense.linear import Linear as D_Linear

from .sparse.attn import SelfAttn as S_SelfAttn
from .sparse.kernelattn import KernelSelfAttn as S_KernelSelfAttn
from .sparse.f import Apply as S_Apply, Add as S_Add
from .sparse.linear import Linear as S_Linear

from ..batch.dense import Batch as D
from ..batch.sparse import Batch as S
from .common.kernel import KernelFeatureMap

from .encoder import TorchMD_ET
from .utils import EquivariantScalar, EquivariantDipoleMoment, EquivariantElectronicSpatialExtent

from hot_pytorch.batch.sparse import make_batch
from .utils import CosineCutoff


#from torch_geometric.nn import global_add_pool



class EquiTransLayer(nn.Module):
    def __init__(self, ord_in, ord_out, dim_in, dim_qk, dim_v, dim_ff, n_heads, cfg='default', att_cfg='default',
                 dropout=0., drop_mu=0., feature_map=None, sparse=True):
        super(EquiTransLayer, self).__init__()
        assert cfg in ('default', 'local')
        assert att_cfg in ('default', 'kernel', 'generalized_kernel')
        SelfAttn = S_SelfAttn if sparse else D_SelfAttn
        KernelSelfAttn = S_KernelSelfAttn if sparse else D_KernelSelfAttn
        Linear = S_Linear if sparse else D_Linear
        Apply = S_Apply if sparse else D_Apply
        self.add = S_Add() if sparse else D_Add()
        self.sparse = sparse
        self.n_heads = n_heads
        

        self.ln = Apply(nn.LayerNorm(dim_in))
        if att_cfg == 'default' or ord_out == 0:
            self.attn = SelfAttn(ord_in, ord_out, dim_in, dim_qk, dim_v, n_heads, cfg, dropout, drop_mu)
        else:
            self.attn = KernelSelfAttn(ord_in, ord_out, dim_in, dim_qk, dim_v, n_heads, cfg, dropout, drop_mu, feature_map)

        self.ffn = nn.Sequential(
            Apply(nn.LayerNorm(dim_in)),
            Linear(ord_out, ord_out, dim_in, dim_ff, cfg='light'),
            Apply(nn.GELU(), skip_masking=True),
            Linear(ord_out, ord_out, dim_ff, dim_in, cfg='light'),
            Apply(nn.Dropout(dropout, inplace=True), skip_masking=True)
        )
        
        
    def forward(self, input):
        G = input[0]
        edge_weight = input[1]
        edge_vec = input[2]
        vec = input[3]
        h = self.ln(G)
        h, delta_vec = self.attn(h, edge_weight, edge_vec, vec)
        vec = vec + delta_vec
                #vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        #vec = vec.reshape(-1, 3, self.n_heads, 32)
        #vec_dot = (vec1 * vec2).sum(dim=1)
        
        G = h
        h = self.ffn(G) 
        return G, edge_weight, edge_vec, vec


class EquivariantTransformer(nn.Module):
    def __init__(self, ord_in, ord_out, ord_hidden: list, dim_in, pdb_out, dim_hidden, dim_qk, dim_v, dim_ff, n_heads,
                 readout_dim_qk, readout_dim_v, readout_n_heads, chem10_out, chem50_out, chem100_out, enc_cfg='default', att_cfg='default',
                 drop_input=0., dropout=0., drop_mu=0., sparse=True):
        super(EquivariantTransformer, self).__init__()
        Linear = S_Linear if sparse else D_Linear
        Apply = S_Apply if sparse else D_Apply
        self.sparse = sparse
        self.dim_hidden = dim_hidden

        self.input = nn.Sequential(
            Linear(ord_in, ord_in, dim_in, dim_hidden, cfg='light'),
            Apply(nn.Dropout(drop_input, inplace=True), skip_masking=True)
        )

        self.feature_map = None
        self.skip_redraw_projections = True
        if att_cfg in ('kernel', 'generalized_kernel'):
            feat_dim = dim_qk // n_heads if dim_qk >= n_heads else 1
            self.feature_map = KernelFeatureMap(feat_dim, generalized_attention=(att_cfg == 'generalized_kernel'))
            self.skip_redraw_projections = False

        layers = []
        for ord1, ord2 in zip([ord_in] + ord_hidden, ord_hidden + [ord_out]):
            dim_qk_, dim_v_, n_heads_ = (dim_qk, dim_v, n_heads) if ord2 > 0 else (readout_dim_qk, readout_dim_v, readout_n_heads)
            layers.append(EquiTransLayer(ord1, ord2, dim_hidden, dim_qk_, dim_v_, dim_ff, n_heads_, enc_cfg, att_cfg, dropout, drop_mu, self.feature_map, sparse))
        self.layers = nn.Sequential(*layers)

        

        self.pdb_output = Apply(nn.Sequential(nn.LayerNorm(dim_hidden), nn.Linear(dim_hidden, pdb_out), nn.LayerNorm(pdb_out)))
        self.chemb10_output = Apply(nn.Sequential(nn.LayerNorm(dim_hidden), nn.Linear(dim_hidden, chem10_out), nn.LayerNorm(chem10_out)))
        self.chemb50_output = Apply(nn.Sequential(nn.LayerNorm(dim_hidden), nn.Linear(dim_hidden, chem50_out), nn.LayerNorm(chem50_out)))
        self.chemb100_output = Apply(nn.Sequential(nn.LayerNorm(dim_hidden), nn.Linear(dim_hidden, chem100_out), nn.LayerNorm(chem100_out)))

        self.pdb_ln = nn.LayerNorm(pdb_out)
        self.chemb10_ln = nn.LayerNorm(chem10_out)
        self.chemb50_ln = nn.LayerNorm(chem50_out)
        self.chemb100_ln = nn.LayerNorm(chem100_out)
        self.reset_parameters()
        #self.output_model = EquivariantScalar(hidden_channels=dim_hidden, out_channels=1)

    def reset_parameters(self):
        self.chemb10_ln.reset_parameters()
        self.chemb50_ln.reset_parameters()
        self.chemb100_ln.reset_parameters()
        self.pdb_ln.reset_parameters()
        
        
    def aggregate(self, vec, index, dim, dim_size):
        vec = scatter(vec, index, dim=dim, dim_size=dim_size)
        return vec

    def forward(self, G, edge_weight, edge_vec, nnode, subname=None):
        assert isinstance(G, S if self.sparse else D)
        if (self.feature_map is not None) and (not self.skip_redraw_projections):
            self.feature_map.redraw_projections()
        G = self.input(G)
        #vec = node_vec
        vec = torch.zeros(nnode, 3, self.dim_hidden, device=G.device)
        G, edge_weight, edge_vec, vec = self.layers((G, edge_weight, edge_vec, vec))
        if subname in {'chemb10'}:
            G = self.chemb10_output(G)
            out = self.aggregate(G.values.squeeze(), G.indices.squeeze().T[1], dim=0, dim_size=nnode)
            #out = self.chemb10_ln(out)
            out = torch.sum(out, dim=0)
            out = self.chemb10_ln(out)
        elif subname in {'chemb50'}:
            G = self.chemb50_output(G)
            out = self.aggregate(G.values.squeeze(), G.indices.squeeze().T[1], dim=0, dim_size=nnode)
            #out = self.chemb50_ln(out)
            out = torch.sum(out, dim=0)
            out = self.chemb50_ln(out)
        elif subname in {'chemb100'}:
            G = self.chemb100_output(G)
            out = self.aggregate(G.values.squeeze(), G.indices.squeeze().T[1], dim=0, dim_size=nnode)
            #out = self.chemb100_ln(out)
            out = torch.sum(out, dim=0)
            out = self.chemb100_ln(out)
        elif subname is None:
            G = self.pdb_output(G)
            out = self.aggregate(G.values.squeeze(), G.indices.squeeze().T[1], dim=0, dim_size=nnode)
            out = self.pdb_ln(out)
            out = out.sum()#torch.sum(out, dim=0, keepdim=True)
        return out
            
            
    
    
class GraphEmbedding(MessagePassing):
    def __init__(self, dim_in, dim_out, cutoff_lower=0.0, cutoff_upper=5.0):
        super(GraphEmbedding, self).__init__()
        
        self.proj_low = nn.Linear(dim_in, dim_out, bias=False)
        self.proj_high = nn.Linear(dim_in, dim_out, bias=False)
        self.proj_identity = nn.Linear(dim_in, dim_out, bias=False)
        self.att_vec_low = nn.Linear(dim_out, 1, bias=False)
        self.att_vec_high = nn.Linear(dim_out, 1, bias=False)
        self.att_vec_mlp = nn.Linear(dim_out, 1, bias=False)
        self.att_vec = nn.Linear(3, 3, bias=False)
        
        self.proj_dist_high = nn.Linear(1, dim_out, bias=False)
        self.proj_dist_low = nn.Linear(1, dim_out, bias=False)
        
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_low.weight)
        #self.proj_low.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.proj_high.weight)
        #self.proj_high.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.proj_identity.weight)
        #self.proj_identity.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.att_vec_low.weight)
        nn.init.xavier_uniform_(self.att_vec_high.weight)
        nn.init.xavier_uniform_(self.att_vec_mlp.weight)
        nn.init.xavier_uniform_(self.att_vec.weight)
        
        nn.init.xavier_uniform_(self.proj_dist_high.weight)
        nn.init.xavier_uniform_(self.proj_dist_low.weight)
        
    def attention3(self, output_low, output_high, output_mlp): #
        T = 3
        att = torch.softmax(self.att_vec(torch.sigmoid(torch.cat([self.att_vec_low((output_low)), self.att_vec_high((output_high)), self.att_vec_mlp((output_mlp))],1)))/T,1) #
        
        return att[:,0][:,None],att[:,1][:,None],att[:,2][:,None]
        
    def forward(self, n_fea, e_id, e_fea, e_we, nnode):                
        XW = self.proj_high(n_fea)
        if not contains_self_loops(e_id):
            e_id, e_fea = add_self_loops(e_id, edge_attr=e_fea, num_nodes=nnode)
            e_we = torch.cat([e_we, torch.zeros(nnode)], dim=-1)
            
        C = self.cutoff(e_we)
        dist_low = self.proj_dist_low(e_fea.float()) * C.view(-1, 1)
        dist_high = self.proj_dist_high(e_fea.float()) * C.view(-1, 1)
        
        AXW = self.propagate(e_id, x=XW, W=dist_high)
        high = F.relu(XW-AXW)
        
        low = F.relu(self.propagate(e_id, x=self.proj_low(n_fea), W=dist_low))
        mlp = F.relu(self.proj_identity(n_fea))
        
        self.att_low, self.att_high, self.att_mlp = self.attention3((low), (high), (mlp)) # 
        return 3*(self.att_low*low + self.att_high*high + self.att_mlp*mlp)
    
    def message(self, x_j, W):
        return x_j * W
    
    
            
class GlobalEncoder(nn.Module):
    def __init__(self, layers, dim_in, dim_hidden, dim_out, rbf, cut_dist, heads):
        super(GlobalEncoder, self).__init__()
        self.representation_model = TorchMD_ET(hidden_channels=dim_hidden, num_layers=layers, num_heads=heads, num_rbf=rbf, cutoff_upper=cut_dist, max_z=100)
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()

    def forward(self, z, e_id, e_fea, e_we, e_vec, pos, task):

        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z)

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, e_id, e_fea, e_we, e_vec, pos, task, batch)
        return x, v, z, pos, batch

        
        
class MD17Encoder(nn.Module):
    def __init__(
        self, dim_hidden, stats_dic):
        super(MD17Encoder, self).__init__()
        self.asp = EquivariantScalar(hidden_channels=dim_hidden)
        self.eth = EquivariantScalar(hidden_channels=dim_hidden)
        self.mal = EquivariantScalar(hidden_channels=dim_hidden)
        self.nap = EquivariantScalar(hidden_channels=dim_hidden)
        self.sal = EquivariantScalar(hidden_channels=dim_hidden)
        self.tol = EquivariantScalar(hidden_channels=dim_hidden)
        self.ura = EquivariantScalar(hidden_channels=dim_hidden)
        
        self.asp_mu, self.asp_sd = stats_dic['asp']
        self.eth_mu, self.eth_sd = stats_dic['eth']
        self.mal_mu, self.mal_sd = stats_dic['mal']
        self.nap_mu, self.nap_sd = stats_dic['nap']
        self.sal_mu, self.sal_sd = stats_dic['sal']
        self.tol_mu, self.tol_sd = stats_dic['tol']
        self.ura_mu, self.ura_sd = stats_dic['ura']
        
        self.reset_parameters()

    def reset_parameters(self):
        self.asp.reset_parameters()
        self.eth.reset_parameters()
        self.mal.reset_parameters()
        self.nap.reset_parameters()
        self.sal.reset_parameters()
        self.tol.reset_parameters()
        self.ura.reset_parameters()

    def forward(self, x, v, z, pos, batch, subname):
        if subname in {'aspirin'}:
            x = self.asp.pre_reduce(x, v, z, pos, batch)
            x = x * self.asp_sd
            out = torch.sum(x, dim=0, keepdim=True)#scatter(x, batch, dim=0, reduce='add')
            out = out + self.asp_mu
            out = self.asp.post_reduce(out)
        elif subname in {'ethanol'}:
            x = self.eth.pre_reduce(x, v, z, pos, batch)
            x = x * self.eth_sd
            out = torch.sum(x, dim=0, keepdim=True)#scatter(x, batch, dim=0, reduce='add')
            out = out + self.eth_mu
            out = self.eth.post_reduce(out)
        elif subname in {'malonaldehyde'}:
            x = self.mal.pre_reduce(x, v, z, pos, batch)
            x = x * self.mal_sd
            out = torch.sum(x, dim=0, keepdim=True)#scatter(x, batch, dim=0, reduce='add')
            out = out + self.mal_mu
            out = self.mal.post_reduce(out)
        elif subname in {'naphthalene'}:
            x = self.nap.pre_reduce(x, v, z, pos, batch)
            x = x * self.nap_sd
            out = torch.sum(x, dim=0, keepdim=True)#scatter(x, batch, dim=0, reduce='add')
            out = out + self.nap_mu
            out = self.nap.post_reduce(out)
        elif subname in {'salicylic_acid'}:
            x = self.sal.pre_reduce(x, v, z, pos, batch)
            x = x * self.sal_sd
            out = torch.sum(x, dim=0, keepdim=True)#scatter(x, batch, dim=0, reduce='add')
            out = out + self.sal_mu
            out = self.sal.post_reduce(out)
        elif subname in {'toluene'}:
            x = self.tol.pre_reduce(x, v, z, pos, batch)
            x = x * self.tol_sd
            out = torch.sum(x, dim=0, keepdim=True)#scatter(x, batch, dim=0, reduce='add')
            out = out + self.tol_mu
            out = self.tol.post_reduce(out)
        elif subname in {'uracil'}:
            x = self.ura.pre_reduce(x, v, z, pos, batch)
            x = x * self.ura_sd
            out = torch.sum(x, dim=0, keepdim=True)#scatter(x, batch, dim=0, reduce='add')
            out = out + self.ura_mu
            out = self.ura.post_reduce(out)


        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
        dy = grad(
            [out],
            [pos],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        return out, -dy
        

            
            
            
class QM9Encoder(nn.Module):
    def __init__(
        self, dim_hidden, prior):
        super(QM9Encoder, self).__init__()
        self.dip = EquivariantDipoleMoment(hidden_channels=dim_hidden)
        self.pol = EquivariantScalar(hidden_channels=dim_hidden)
        self.hom = EquivariantScalar(hidden_channels=dim_hidden)
        self.lum = EquivariantScalar(hidden_channels=dim_hidden)
        self.dlt = EquivariantScalar(hidden_channels=dim_hidden)
        self.ele = EquivariantElectronicSpatialExtent(hidden_channels=dim_hidden)
        self.zpv = EquivariantScalar(hidden_channels=dim_hidden)
        self.eu0 = EquivariantScalar(hidden_channels=dim_hidden)
        self.eu1 = EquivariantScalar(hidden_channels=dim_hidden)
        self.ent = EquivariantScalar(hidden_channels=dim_hidden)
        self.efr = EquivariantScalar(hidden_channels=dim_hidden)
        self.hea = EquivariantScalar(hidden_channels=dim_hidden)
        
        self.label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        
        self.prior_model = prior
        
        self.reset_parameters()

    def reset_parameters(self):
        self.dip.reset_parameters()
        self.pol.reset_parameters()
        self.hom.reset_parameters()
        self.lum.reset_parameters()
        self.dlt.reset_parameters()
        self.ele.reset_parameters()
        self.zpv.reset_parameters()
        self.eu0.reset_parameters()
        self.eu1.reset_parameters()
        self.ent.reset_parameters()
        self.efr.reset_parameters()
        self.hea.reset_parameters()
        
        for model in self.prior_model:
            model.reset_parameters()
        
    def dipole(self, x, v, z, pos, batch):
        x = self.dip.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['dipole_moment']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.dip.post_reduce(out)
        return out
    
    def polar(self, x, v, z, pos, batch):
        x = self.pol.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['isotropic_polarizability']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.pol.post_reduce(out)
        return out
    
    def homo(self, x, v, z, pos, batch):
        x = self.hom.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['homo']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.hom.post_reduce(out)
        return out
    
    def lumo(self, x, v, z, pos, batch):
        x = self.lum.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['lumo']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.lum.post_reduce(out)
        return out
    
    def delta(self, x, v, z, pos, batch):
        x = self.dlt.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['gap']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.dlt.post_reduce(out)
        return out
    
    def electronic(self, x, v, z, pos, batch):
        x = self.ele.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['electronic_spatial_extent']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.ele.post_reduce(out)
        return out
    
    def zpve(self, x, v, z, pos, batch):
        x = self.zpv.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['zpve']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.zpv.post_reduce(out)
        return out
    
    def energyU0(self, x, v, z, pos, batch):
        x = self.eu0.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['energy_U0']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.eu0.post_reduce(out)
        return out
    
    def energyU(self, x, v, z, pos, batch):
        x = self.eu1.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['energy_U']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.eu1.post_reduce(out)
        return out
    
    def enthalpy(self, x, v, z, pos, batch):
        x = self.ent.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['enthalpy_H']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.ent.post_reduce(out)
        return out
    
    def energyFree(self, x, v, z, pos, batch):
        x = self.efr.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['free_energy']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.efr.post_reduce(out)
        return out
    
    def heat(self, x, v, z, pos, batch):
        x = self.hea.pre_reduce(x, v, z, pos, batch)
        x = self.prior_model[self.label2idx['heat_capacity']](x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce='add')
        out = self.hea.post_reduce(out)
        return out

    def forward(self, x, v, z, pos, batch):
        dip = self.dipole(x, v, z, pos, batch)
        pol = self.polar(x, v, z, pos, batch)
        hom = self.homo(x, v, z, pos, batch)
        lum = self.lumo(x, v, z, pos, batch)
        gap = self.delta(x, v, z, pos, batch)
        ele = self.electronic(x, v, z, pos, batch)
        zpv = self.zpve(x, v, z, pos, batch)
        eu0 = self.energyU0(x, v, z, pos, batch)
        eu1 = self.energyU(x, v, z, pos, batch)
        ent = self.enthalpy(x, v, z, pos, batch)
        efr = self.energyFree(x, v, z, pos, batch)
        hea = self.heat(x, v, z, pos, batch)

        return dip, pol, hom, lum, gap, ele, zpv, eu0, eu1, ent, efr, hea
    
    
    
        
class ModelI(nn.Module):
    def __init__(
        self,
        layers, dim_in, dim_hidden, dim_out, rbf, cut_dist, heads,
        mean,
        std,
        ):
        super(ModelI, self).__init__()
        self.representation_model = TorchMD_ET(hidden_channels=dim_hidden, num_layers=layers, num_heads=heads, num_rbf=rbf, cutoff_upper=cut_dist, max_z=dim_in)
        self.output_model = EquivariantScalar(hidden_channels=dim_hidden)
        self.proj_network = nn.Linear(dim_hidden, dim_out)
        self.mean = mean
        self.std = std
        
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        nn.init.xavier_uniform_(self.proj_network.weight)
        self.proj_network.bias.data.fill_(0)

    def forward(self, z, e_id, e_fea, e_we, e_vec, pos, mtl=False):

        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z)

        if mtl:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, e_id, e_fea, e_we, e_vec, pos, mtl, batch)
        
        if not mtl:
            out = self.proj_network(x)
            return out, v
        
#         elif:
#             x = self.output_model.pre_reduce(x, v, z, pos, batch)
#             x = self.prior_model(x, z, pos, batch)
#             # aggregate atoms
#             out = torch.sum(x, dim=0, keepdim=True)#scatter(x, batch, dim=0, reduce='add')
#             # apply output model after reduction
#             out = self.output_model.post_reduce(out)
#             return out
        
        else:
            # apply the output network
            x = self.output_model.pre_reduce(x, v, z, pos, batch)

            x = x * self.std
            
            # aggregate atoms
            out = torch.sum(x, dim=0, keepdim=True)#scatter(x, batch, dim=0, reduce='add')
            # shift by data mean
            out = out + self.mean


            # apply output model after reduction
            out = self.output_model.post_reduce(out)

            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]

            return out, -dy
            
            
        
class MTLModel(nn.Module):
    def __init__(self, stats_dic, glob_layer=6, glob_heads=8, glob_in=64, glob_hidden=64, glob_out=64, n_rbf=50, cut_dist=5., pdb_in=None, chemb_in=None, pre_complex_out=64, complex_layer=4, complex_in=32, complex_out=32, complex_hidden=32, complex_qk=32, complex_v=32, complex_ff=32, complex_heads=4, complex_readout_qk=32, complex_readout_v=32, complex_readout_heads=4, chem10_out=406, chem50_out=263, chem100_out=129, enc_cfg='default', att_cfg='default', complex_dropout_in=0., complex_dropout=0., complex_dropout_mu=0., prior=None, sparse=True):
        super(MTLModel, self).__init__()
        
        self.glob = GlobalEncoder(glob_layer, glob_in, glob_hidden, glob_out, n_rbf, cut_dist, glob_heads)
        self.md17 = MD17Encoder(glob_hidden, stats_dic)
        self.qm9 = QM9Encoder(glob_hidden, prior=prior)
        #self.modelI = ModelI(n_layersI, dim_atom_inI, dim_atom_hiddenI, dim_atom_outI, n_rbf, cut_dist, n_headsI, mol_mean, mol_std)
        #self.embedding = GraphEmbedding(dim_emb_in, dim_emb_out, cutoff_upper=cut_dist)
        self.complex = EquivariantTransformer(2, 2, [2] * complex_layer, complex_in, complex_out, complex_hidden, complex_qk, complex_v, complex_ff, complex_heads, complex_readout_qk, complex_readout_v, complex_readout_heads, chem10_out, chem50_out, chem100_out, 'default', 'generalized_kernel', complex_dropout_in, complex_dropout, complex_dropout_mu, True)
        
        #self.proj_combine = nn.Linear(dim_atom_outI+dim_emb_out, dim_atom_outI+dim_emb_out, bias=False)
        self.proj_combine_pdb = nn.Linear(glob_hidden+pdb_in, glob_out+pre_complex_out, bias=False)
        self.proj_combine_chemb = nn.Linear(glob_hidden+chemb_in, glob_out+pre_complex_out, bias=False)
        self.transform = nn.Linear(glob_out+pre_complex_out, glob_out+pre_complex_out, bias=False)
        
        #self.cutoff = CosineCutoff(cutoff_lower=0., cutoff_upper=cut_dist)
        self.complex_ln = nn.LayerNorm(glob_out+pre_complex_out)

        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_combine_pdb.weight)
        nn.init.xavier_uniform_(self.proj_combine_chemb.weight)
        nn.init.xavier_uniform_(self.transform.weight)
        self.complex_ln.reset_parameters()
        #nn.init.xavier_uniform_(self.vec_proj_network.weight)

    def forward(self, pos, atoms, nnode, task, n_fea=None, e_id=None, e_fea=None, e_we=None, e_vec=None, subname=None):
        #return self.test_model(atoms, e_id, e_fea, e_we, e_vec, pos, mtl=True)
        if task in {'md17'}:
            pos.requires_grad_(True)
            x, v, z, pos, batch = self.glob(atoms, e_id, e_fea, e_we, e_vec, pos, task)
            energy, dy = self.md17(x, v, z, pos, batch, subname)
            #energy, dy = self.modelI(atoms, e_id, e_fea, e_we, e_vec, pos, mtl)
            #energy, dy = self.pred_network(atom_features.float(), processed_pos, batch)
            return energy, dy
        
        
        elif task in {'qm9'}:
            x, v, z, pos, batch = self.glob(atoms, e_id, e_fea, e_we, e_vec, pos, task)
            pred = self.qm9(x, v, z, pos, batch)
            #energy, dy = self.modelI(atoms, e_id, e_fea, e_we, e_vec, pos, mtl)
            #energy, dy = self.pred_network(atom_features.float(), processed_pos, batch)
            return pred
        
        
        elif task in {'chemb'}:
            atom_features, _, _, _, _ = self.glob(atoms, e_id, e_fea, e_we, e_vec, pos, task)
            #atom_features, node_vec = self.modelI(atoms, e_id, e_fea, e_we, e_vec, pos)
            #node_vec = self.vec_proj_network(node_vec)
            #node_features = self.embedding(n_fea.float(), e_id, e_fea, e_we, nnode)
            #features = self.proj_combine(torch.cat([node_features, atom_features], -1))
            features = self.proj_combine_chemb(torch.cat([n_fea.float(), atom_features], -1))
            features = self.complex_ln(self.transform(features))
            
            G = make_batch([features], [e_id], [e_fea])
            edge_index = G.indices.T
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
            edge_weight = torch.norm(edge_vec, dim=-1)
            mask = edge_index[0] != edge_index[1]
            edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
            
            out = self.complex(G, edge_weight, edge_vec, nnode, subname)
            return out
        
        
        elif task in {'pdb'}:
            atom_features, _, _, _, _ = self.glob(atoms, e_id, e_fea, e_we, e_vec, pos, task)
            #atom_features, node_vec = self.modelI(atoms, e_id, e_fea, e_we, e_vec, pos)
            #node_vec = self.vec_proj_network(node_vec)
            #node_features = self.embedding(n_fea.float(), e_id, e_fea, e_we, nnode)
            #features = self.proj_combine(torch.cat([node_features, atom_features], -1))
            features = self.proj_combine_pdb(torch.cat([n_fea.float(), atom_features], -1))
            features = self.complex_ln(self.transform(features))
            
            G = make_batch([features], [e_id], [e_fea])
            edge_index = G.indices.T
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
            edge_weight = torch.norm(edge_vec, dim=-1)
            mask = edge_index[0] != edge_index[1]
            edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
            
            out = self.complex(G, edge_weight, edge_vec, nnode)
            return out