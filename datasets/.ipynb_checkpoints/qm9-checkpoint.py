import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

class Transform(BaseTransform):
    def __init__(self, cut_dist=5.):
        self.cut_dist = cut_dist
        
    def __call__(self, data):
        coords = data.pos
        edges = data.edge_index

        dist_mat = torch.cdist(coords, coords)[edges[0], edges[1]].reshape(-1, 1)
        edge_attr = torch.cat([data.edge_attr,dist_mat],dim=-1)
        
        edge_vec = coords[edges[0]] - coords[edges[1]]
        edge_weight = torch.norm(edge_vec, dim=-1)
        mask = edges[0] != edges[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        
        data = Data(edge_index=edges, edge_attr=edge_attr, edge_weight=edge_weight, edge_vec=edge_vec, y=data.y, pos=coords, atom=data.z, dataset='qm9')
        return data
    

class QM9(QM9_geometric):
    def __init__(self, root, cut_dist=5., transform=None):
        transform = Transform(cut_dist)
        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM9, self).__init__(root, transform=transform)

    def get_atomref(self, max_z=100, idx=None):
        atomref = self.atomref(idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, :max(qm9_target_dict.keys())+1].T
        return batch

    def download(self):
        super(QM9, self).download()

    def process(self):
        super(QM9, self).process()