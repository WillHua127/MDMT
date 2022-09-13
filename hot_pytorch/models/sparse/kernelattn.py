import torch
import torch.nn as nn

from ...batch.sparse import Batch as B, batch_like, t, v2d, d, add_batch
from ..common.mudrop import MuDropout
from .linear import Linear
from .kernelattncoef import KernelFeatureMapWrapper, KernelAttnCoef
from ..equivariant import MPNN


class KernelSelfAttn(nn.Module):
    def __init__(self, ord_in, ord_out, dim_in, dim_v, dim_qk, n_heads, cfg='default', dropout=0., drop_mu=0., feature_map=None):
        super().__init__()
        assert cfg in ('default', 'local')
        self.is_local = cfg == 'local'
        self.ord_in = ord_in
        self.ord_out = ord_out
        self.dim_in = dim_in
        self.dim_v = dim_v
        self.dim_qk = dim_qk
        self.n_heads = n_heads
        self.feature_map = KernelFeatureMapWrapper(feature_map, dim_qk, n_heads)
        self.feat_dim = feature_map.num_features
        self.head_dim = dim_in // n_heads
                
        if (ord_in, ord_out) == (1, 0):
            raise ValueError('Kernel gives no asymptotic improvement. Use softmax instead')
        elif (ord_in, ord_out) == (1, 1):
            n_qk1, n_v = 2, 1
            self.fc_1 = Linear(1, 1, dim_in, dim_qk * n_qk1 + dim_in, cfg='light')
            self.att_1_1 = KernelAttnCoef(1, 1, self.feat_dim, dim_v, n_heads)
        elif (ord_in, ord_out) == (1, 2):
            raise NotImplementedError('Sparse set-to-graph is inefficient; use a dense layer')
        elif (ord_in, ord_out) == (2, 0):
            raise ValueError('Kernel gives no asymptotic improvement. Use softmax instead')
        elif (ord_in, ord_out) == (2, 1):
            n_qk1, n_qk2, n_v = 7, 1, 4
            self.fc_1 = Linear(2, 1, dim_in, dim_qk * n_qk1 + dim_in, cfg='light')
            self.fc_2 = Linear(2, 2, dim_in, dim_qk * n_qk2, cfg='light')
            self.att_1_1 = KernelAttnCoef(1, 1, self.feat_dim, dim_v, n_heads)
            self.att_1_2 = KernelAttnCoef(1, 2, self.feat_dim, dim_v, n_heads)
        elif (ord_in, ord_out) == (2, 2):
            n_qk1, n_qk2, n_v = 12, 8, 10
            self.fc_1 = Linear(2, 1, dim_in, dim_qk * n_qk1, cfg='light')
            self.fc_2 = Linear(2, 2, dim_in, dim_qk * n_qk2 + dim_in, cfg='light')
            self.att_1_1 = KernelAttnCoef(1, 1, self.feat_dim, dim_v, n_heads)
            self.att_2_1 = KernelAttnCoef(2, 1, self.feat_dim, dim_v, n_heads)
            self.att_1_2 = KernelAttnCoef(1, 2, self.feat_dim, dim_v, n_heads)
            self.att_2_2 = KernelAttnCoef(2, 2, self.feat_dim, dim_v, n_heads)
        else:
            raise NotImplementedError
            
        self.n_v = n_v
        #self.fc_v = nn.Linear(dim_in, dim_v * n_v)
        self.fc_v = nn.Linear(dim_in, dim_v * n_v * 3)
        self.fc_o = nn.Linear(dim_v * n_v, dim_in)
        self.mpnn = MPNN(dim_in, n_v)
        #self.bn = nn.BatchNorm1d(dim_in)
        #self.layernorm = nn.LayerNorm(dim_in * n_v)
        
        
        self.equiv_proj = nn.Linear(dim_v * n_v, dim_in, bias=False)
        #self.value_proj = nn.Linear(dim_in, dim_v * n_v * 2, bias=False)
        self.out_proj = nn.Linear(dim_v * n_v, dim_v * 3 * n_v, bias=False)
        self.vec_proj = nn.Linear(dim_in, dim_in * 3 * n_v, bias=False)
        self.ln = nn.LayerNorm(dim_in * n_v)
        
        self.reset_parameters()
        
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.mu_dropout = MuDropout(p=drop_mu)

    def reset_parameters(self):
        self.ln.reset_parameters()
        nn.init.xavier_normal_(self.fc_v.weight)
        nn.init.xavier_normal_(self.fc_o.weight)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.xavier_normal_(self.vec_proj.weight)
        nn.init.xavier_normal_(self.equiv_proj.weight)
        #nn.init.xavier_normal_(self.value_proj.weight)
        
        nn.init.constant_(self.fc_v.bias, 0.)
        nn.init.constant_(self.fc_o.bias, 0.)

    def get_qk_list(self, G: B):
        v_list = G.values.split(self.feat_dim * self.n_heads, -1)
        return [batch_like(G, v, skip_masking=True) for v in v_list]

    def get_v_list(self, G: B):
        proj_v = self.fc_v(G.values)
        #vec = self.value_proj(G.values)
        proj_v, vec1, vec2 = torch.split(proj_v, self.dim_v * self.n_v, dim=2)
        
        v = batch_like(G, proj_v, skip_masking=False)
        v_list = v.values.split(self.dim_v, -1)
        vec1_list = vec1.split(self.dim_v, -1)
        vec2_list = vec2.split(self.dim_v, -1)
        return [batch_like(G, v, skip_masking=True) for v in v_list], vec1_list, vec2_list

    def combine_att(self, G: B, att_list, delta_x_list, edge_weight, last_layer=False):
        att = self.fc_o(self.dropout(torch.cat(self.mu_dropout([att_list[idx].values + delta_x_list[idx].unsqueeze(0) for idx in range(self.n_v)]), -1)))
        if last_layer:
            return att
        else:
            att = att * edge_weight
            return batch_like(G, att, skip_masking=False)
    
    
    def combine_equiv(self, delta_vec_list):
        equiv_vec = self.equiv_proj(delta_vec_list)
        return equiv_vec


    def _2_to_2_v2(self, G: B, vec, vec_dot_list, vec3_list, edge_weight, edge_vec, nnode):
        # compute query, key and value
        h_1 = self.fc_1(G)  # [B, N, 12D]
        q_1 = batch_like(h_1, h_1.values[..., :self.dim_qk * 4], skip_masking=True)  # [B, N, 4D]
        k_1 = batch_like(h_1, h_1.values[..., self.dim_qk * 4:], skip_masking=True)  # [B, N, 8D]
        h_2 = self.fc_2(G)  # [B, |E|, (8+1)D]
        non_att = batch_like(h_2, h_2.values[..., -self.dim_in:], skip_masking=True)  # [B, N, D]
        q_2 = batch_like(h_2, h_2.values[..., :self.dim_qk * 6], skip_masking=True)  # [B, |E|, 6D]
        k_2 = batch_like(h_2, h_2.values[..., self.dim_qk * 6:self.dim_qk * 8], skip_masking=True)  # [B, |E|, 2D]
        edge = G.indices
        v_2_list, vec1_list, vec2_list = self.get_v_list(G, nnode, edge)  # [B, |E|, 10D]
        # kernel feature map
        q_1 = self.feature_map(q_1, is_query=True)  # [B, N, 4D']
        q_2 = self.feature_map(q_2, is_query=True)  # [B, |E|, 8D']
        k_1 = self.feature_map(k_1, is_query=False)  # [B, N, 6D']
        k_2 = self.feature_map(k_2, is_query=False)  # [B, |E|, 2D']
        q_1_list = self.get_qk_list(q_1)  # List([B, N, D'])
        q_2_list = self.get_qk_list(q_2)  # List([B, |E|, D'])
        k_1_list = self.get_qk_list(k_1)  # List([B, N, D'])
        k_2_list = self.get_qk_list(k_2)  # List([B, |E|, D'])
        
        #print(self.layernorm(self.mpnn.equiv_aggregate(G.indices, vec1_list[0].squeeze(), nnode)))
        
        vec1_list = self.mpnn.all_expand(edge, vec1_list)
        vec2_list = self.mpnn.all_expand(edge, vec2_list)
        vec_list = self.mpnn.propagate(edge, vec, vec1_list, vec2_list, edge_vec, last_layer=True)

        # graph -> set
        att_1, att_2 = self.att_1_1(q_1_list[0:2], k_1_list[0:2], [v_2_list[0], t(v_2_list[1])], diagonal=(1, 2))  # [B, N, D] -> [B, |E|, D]
        att_1 = v2d(G, att_1)
        att_2 = v2d(G, att_2)
        # graph -> graph
        att_3, att_5 = self.att_2_1(q_2_list[0:2], k_1_list[2:4], [v_2_list[2], t(v_2_list[3])], diagonal=(2, 3))
        att_4, att_6 = self.att_2_1(q_2_list[2:4], k_1_list[4:6], [t(v_2_list[4]), v_2_list[5]], diagonal=(1, 3))
        att_list = [att_1, att_2, att_3, att_4, att_5, att_6]
        if not self.is_local:
            # set -> set
            att_7 = v2d(G, self.att_1_1(q_1_list[2], k_1_list[6], d(v_2_list[6])))  # [B, N, D] -> [B, |E|, D]
            # graph -> set
            att_8 = v2d(G, self.att_1_2(q_1_list[3], k_2_list[0], v_2_list[7]))  # [B, N, D] -> [B, |E|, D]
            # set -> graph
            att_9 = self.att_2_1(q_2_list[4], k_1_list[7], d(v_2_list[8]))  # [B, N, D] -> [B, |E|, D]
            # graph -> graph
            att_10 = self.att_2_2(q_2_list[5], k_2_list[1], v_2_list[9])
            att_list += [att_7, att_8, att_9, att_10]
            
        
        o1_list, o2_list, o3_list = torch.split(self.out_proj(torch.cat([self.mpnn.equiv_aggregate(edge, att.values.squeeze(), nnode) for att in att_list], dim=-1)), self.dim_v * self.n_v, dim=-1)
        o1_list = torch.split(o1_list, self.dim_v, dim=-1)
        o2_list = torch.split(o2_list, self.dim_v, dim=-1)
        o3_list = torch.split(o3_list, self.dim_v, dim=-1)
        
        o1_list = self.mpnn.all_expand(edge, o1_list)
        o2_list = self.mpnn.all_expand(edge, o2_list)
        o3_list = self.mpnn.all_expand(edge, o3_list)        
                
        delta_x_list = [vec_dot_list[idx] * o2_list[idx] + o3_list[idx] for idx in range(self.n_v)]
        delta_vec_list = [vec3_list[idx] * o1_list[idx].unsqueeze(1) + vec_list[idx]  for idx in range(self.n_v)]
                
        # combine
        att = self.combine_att(G, att_list, delta_x_list, edge_weight)
        delta_vec = self.mpnn.equiv_aggregate(edge, torch.cat(delta_vec_list, -1), nnode)
        delta_vec = self.ln(delta_vec)
        delta_vec = self.combine_equiv(delta_vec)
        
        return add_batch(non_att, att), delta_vec

    

    def _2_to_2(self, G: B, vec, vec_dot_list, vec3_list, edge_weight, edge_vec, nnode):
        # compute query, key and value
        h_1 = self.fc_1(G)  # [B, N, 12D]
        q_1 = batch_like(h_1, h_1.values[..., :self.dim_qk * 4], skip_masking=True)  # [B, N, 4D]
        k_1 = batch_like(h_1, h_1.values[..., self.dim_qk * 4:], skip_masking=True)  # [B, N, 8D]
        h_2 = self.fc_2(G)  # [B, |E|, (8+1)D]
        non_att = batch_like(h_2, h_2.values[..., -self.dim_in:], skip_masking=True)  # [B, N, D]
        q_2 = batch_like(h_2, h_2.values[..., :self.dim_qk * 6], skip_masking=True)  # [B, |E|, 6D]
        k_2 = batch_like(h_2, h_2.values[..., self.dim_qk * 6:self.dim_qk * 8], skip_masking=True)  # [B, |E|, 2D]
        v_2_list, vec1_list, vec2_list = self.get_v_list(G)  # [B, |E|, 10D]
        # kernel feature map
        q_1 = self.feature_map(q_1, is_query=True)  # [B, N, 4D']
        q_2 = self.feature_map(q_2, is_query=True)  # [B, |E|, 8D']
        k_1 = self.feature_map(k_1, is_query=False)  # [B, N, 6D']
        k_2 = self.feature_map(k_2, is_query=False)  # [B, |E|, 2D']
        q_1_list = self.get_qk_list(q_1)  # List([B, N, D'])
        q_2_list = self.get_qk_list(q_2)  # List([B, |E|, D'])
        k_1_list = self.get_qk_list(k_1)  # List([B, N, D'])
        k_2_list = self.get_qk_list(k_2)  # List([B, |E|, D'])
        
        #print(self.layernorm(self.mpnn.equiv_aggregate(G.indices, vec1_list[0].squeeze(), nnode)))
        
        vec_list = self.mpnn.propagate(G.indices, vec, vec1_list, vec2_list, edge_vec, last_layer=True)

        # graph -> set
        att_1, att_2 = self.att_1_1(q_1_list[0:2], k_1_list[0:2], [v_2_list[0], t(v_2_list[1])], diagonal=(1, 2))  # [B, N, D] -> [B, |E|, D]
        att_1 = v2d(G, att_1)
        att_2 = v2d(G, att_2)
        # graph -> graph
        att_3, att_5 = self.att_2_1(q_2_list[0:2], k_1_list[2:4], [v_2_list[2], t(v_2_list[3])], diagonal=(2, 3))
        att_4, att_6 = self.att_2_1(q_2_list[2:4], k_1_list[4:6], [t(v_2_list[4]), v_2_list[5]], diagonal=(1, 3))
        att_list = [att_1, att_2, att_3, att_4, att_5, att_6]
        if not self.is_local:
            # set -> set
            att_7 = v2d(G, self.att_1_1(q_1_list[2], k_1_list[6], d(v_2_list[6])))  # [B, N, D] -> [B, |E|, D]
            # graph -> set
            att_8 = v2d(G, self.att_1_2(q_1_list[3], k_2_list[0], v_2_list[7]))  # [B, N, D] -> [B, |E|, D]
            # set -> graph
            att_9 = self.att_2_1(q_2_list[4], k_1_list[7], d(v_2_list[8]))  # [B, N, D] -> [B, |E|, D]
            # graph -> graph
            att_10 = self.att_2_2(q_2_list[5], k_2_list[1], v_2_list[9])
            att_list += [att_7, att_8, att_9, att_10]
            
        
        o1_list, o2_list, o3_list = torch.split(self.out_proj(torch.cat([atten.values.squeeze() for atten in att_list], dim=-1)), self.dim_v * self.n_v, dim=-1)
        o1_list = torch.split(o1_list, self.dim_v, dim=-1)
        o2_list = torch.split(o2_list, self.dim_v, dim=-1)
        o3_list = torch.split(o3_list, self.dim_v, dim=-1)
        
                
        delta_x_list = [vec_dot_list[idx] * o2_list[idx] + o3_list[idx] for idx in range(self.n_v)]
        delta_vec_list = [vec3_list[idx] * o1_list[idx].unsqueeze(1) + vec_list[idx]  for idx in range(self.n_v)]
                
        # combine
        att = self.combine_att(G, att_list, delta_x_list, edge_weight)
        delta_vec = self.mpnn.equiv_aggregate(G.indices, torch.cat(delta_vec_list, -1), nnode)
        delta_vec = self.ln(delta_vec)
        delta_vec = self.combine_equiv(delta_vec)
        
        return add_batch(non_att, att), delta_vec

    def forward(self, G: B, edge_weight, edge_vec, vec):
        assert G.order == self.ord_in
                
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.dim_in * self.n_v, dim=-1)
        #vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.dim_in, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot_list = torch.split(vec_dot, self.dim_in, dim=-1)
        vec3_list = torch.split(vec3, self.dim_in, dim=-1)
        nnode = vec.shape[0]
        
                
        #G.values = self.layernorm(G.values)                
        #vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        if (self.ord_in, self.ord_out) == (1, 1):
            G_att = self._1_to_1(G)
        elif (self.ord_in, self.ord_out) == (2, 1):
            G_att = self._2_to_1(G, vec, vec_dot_list, vec3_list, edge_weight, edge_vec, nnode)
        elif (self.ord_in, self.ord_out) == (2, 2):
            vec = self.mpnn.all_expand(G.indices, vec, single=True, reshape=True)
            vec_dot_list = self.mpnn.all_expand(G.indices, vec_dot_list)        
            vec3_list = self.mpnn.all_expand(G.indices, vec3_list, reshape=True)
            G_att, delta_vec = self._2_to_2(G, vec, vec_dot_list, vec3_list, edge_weight, edge_vec, nnode)
        else:
            raise NotImplementedError('Currently supports up to second-order invariance only')
            
        #vec = vec + delta_vec

        if self.ord_out > 0:
            assert G_att.order == self.ord_out
        else:
            assert isinstance(G_att, torch.Tensor)
            
        #dx = vec_dot * o2 + o3
        #dvec = vec3 * o1.unsqueeze(1) + vec
        
        return G_att, delta_vec
