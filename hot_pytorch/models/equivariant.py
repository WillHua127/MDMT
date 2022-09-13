from typing import Optional, Tuple, List
import torch
from torch import Tensor, nn
from torch_scatter import scatter


from torch_geometric.typing import Adj, Size
from torch_scatter.utils import broadcast
from torch_geometric.nn.conv.utils.inspector import Inspector
    
    
    
class MPNN(torch.nn.Module):
    def __init__(self, dim_in, n_v, aggr = "add",
                 flow = "source_to_target", node_dim = 0):

        super(MPNN, self).__init__()

        self.dim_in = dim_in
        self.n_v = n_v

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']
        
        self.ln = nn.LayerNorm(dim_in)

        self.node_dim = node_dim

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)

    def reset_parameters(self):
        self.ln.reset_parameters()

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype == torch.long
            assert edge_index.dim() == 2
            assert edge_index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.index_select(self.node_dim, index)


    def __collect__(self, edge_index, size, x):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        dim = 0
        data = x
        self.__set_size__(size, dim, data)
        data = self.__lift__(data, edge_index, j)


        return data
    
    
    def expand(self, edge_index, vec_dot_list, vec3_list, size: Size = None):
        edge_index = edge_index.squeeze().T
        size = self.__check_input__(edge_index, size)
        vec_dot_list = [self.__collect__(edge_index, size, vec_dot_list[idx]) for idx in range(self.n_v)]
        vec3_list = [self.__collect__(edge_index, size, vec3_list[idx].reshape(-1, 3 * self.dim_in)).reshape(-1, 3, self.dim_in) for idx in range(self.n_v)]
        
        return vec_dot_list, vec3_list
    
    def all_expand(self, edge_index, content, reshape=False, single= False, size: Size = None):
        edge_index = edge_index.squeeze().T
        size = self.__check_input__(edge_index, size)
        if not single:
            if not reshape:
                content = [self.__collect__(edge_index, size, content[idx]) for idx in range(self.n_v)]
            else:
                content = [self.__collect__(edge_index, size, content[idx].reshape(-1, 3 * self.dim_in)).reshape(-1, 3, self.dim_in) for idx in range(self.n_v)]
        else:
            if not reshape:
                content = self.__collect__(edge_index, size, content)
            else:
                content = self.__collect__(edge_index, size, content.reshape(-1, 3 * self.dim_in)).reshape(-1, 3, self.dim_in)
        return content
    
    def equiv_aggregate(self, edge_index, content, nnode):
        edge_index = edge_index.squeeze().T
        #agg_list = []
        #for idx in range(self.n_v):
        #    equiv_vec = equiv_vec_list[idx].reshape(-1, 3 * self.dim_in)
        #    agg_list.append(self.aggregate(equiv_vec, edge_index[1], ptr=None, dim_size=nnode).reshape(-1, 3, self.dim_in))
        
        content = self.aggregate(content, edge_index[1], dim_size=nnode)

            
        return content#agg_list
    
    
    def propagate_v2(self, edge_index, vec, pos_vec1_list, pos_vec2_list, edge_vec, size: Size = None, edge_attr = None, last_layer=False):
        edge_index = edge_index.squeeze().T
        #vec = vec.reshape(-1, 3 * self.dim_in)
        size = self.__check_input__(edge_index, size)
        nnode = vec.shape[0]
        #equiv_list = []
        aggs = []
        #if last_layer:
        #    vec = vec.reshape(-1, 3 * self.dim_in)
        #    vec = self.__collect__(edge_index, size, vec)
        #    vec = vec.reshape(-1, 3, self.dim_in)
        
        for idx in range(self.n_v):
            msg = self.message(vec, pos_vec1_list[0], pos_vec2_list[0], edge_vec)
            msg = msg.reshape(-1, 3 * self.dim_in)
            agg = self.aggregate(msg, edge_index[1], dim_size=nnode)
            #aggs.append(agg)

            if not last_layer:
                agg = self.__collect__(edge_index, size, agg)
                
            aggs.append(self.ln(agg.reshape(-1, 3, self.dim_in)))

        return aggs
    
        
    def propagate(self, edge_index, vec, pos_vec1_list, pos_vec2_list, edge_vec, size: Size = None, edge_attr = None, last_layer=False):
        edge_index = edge_index.squeeze().T
        #vec = vec.reshape(-1, 3 * self.dim_in)
        size = self.__check_input__(edge_index, size)
        nnode = vec.shape[0]
        #equiv_list = []
        aggs = []

        if last_layer:
            vec = vec.reshape(-1, 3 * self.dim_in)
            vec = self.__collect__(edge_index, size, vec)
            vec = vec.reshape(-1, 3, self.dim_in)
        
        for idx in range(self.n_v):
            msg = self.message(vec, pos_vec1_list[idx].squeeze(), pos_vec2_list[idx].squeeze(), edge_vec)
            msg = msg.reshape(-1, 3 * self.dim_in)
            agg = self.aggregate(msg, edge_index[1], dim_size=nnode)
            #aggs.append(agg)

            if not last_layer:
                agg = self.__collect__(edge_index, size, agg)
                
            aggs.append(agg.reshape(-1, 3, self.dim_in))

        return aggs

    def message(self, vec, pos_vec1, pos_vec2, edge_vec):
        # update vector features
        vec_pos = vec * pos_vec1.unsqueeze(1) + pos_vec2.unsqueeze(1) * edge_vec.unsqueeze(3).squeeze(1)
                
        return vec_pos

    def aggregate(
        self,
        vec: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        dim_size: Optional[int],
    ) -> torch.Tensor :
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor :
        return inputs
    
    
    