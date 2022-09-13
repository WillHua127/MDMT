import os
import numpy as np
import pickle
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from tqdm import tqdm
from torch_geometric.data import Data
import numpy as np
import torch
                
                
class PDB():
    def __init__(self, data_path, dataset, cut_dist, save_file=True):
        self.data_path = data_path
        self.dataset = dataset
        self.cut_dist = cut_dist
        self.save_file = save_file
    
        self.graphs = []

        self.load_data()
        
        
    def __len__(self):
        """ Return the number of graphs. """
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

    def has_cache(self):
        """ Check cache file."""
        self.graph_path = f'{self.data_path}/{self.dataset}_{int(self.cut_dist)}_graph.pkl'
        return os.path.exists(self.graph_path)
    
    def save(self):
        """ Save the generated graphs. """
        print('Saving processed complex data...')
        with open(self.graph_path, 'wb') as f:
            pickle.dump((self.graphs), f)
        #graph_path = f'{self.data_path}/{self.dataset}_{int(self.cut_dist)}_pyg.pt'

    def load(self):
        """ Load the generated graphs. """
        print('Loading processed complex data...')
        with open(self.graph_path, 'rb') as f:
            graphs = pickle.load(f)
        return graphs

    
    def build_graph(self, mol, y):
        _, coords, features, atoms, inter_feats = mol

        ##################################################
        # prepare distance matrix and interaction matrix #
        ##################################################
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)

        ############################
        # build atom to atom graph #
        ############################
        dist_graph_base = dist_mat.copy()
        dist_feat = dist_graph_base[dist_graph_base < self.cut_dist].reshape(-1,1)
        dist_graph_base[dist_graph_base >= self.cut_dist] = 0.
        atom_graph = coo_matrix(dist_graph_base)
        edges = torch.tensor([atom_graph.row, atom_graph.col], dtype=torch.long)
        #print([atom_graph.row, atom_graph.col])
        features = torch.tensor(features, dtype=torch.long)
        dist_feat = torch.tensor(dist_feat, dtype=torch.long)
        coords = torch.tensor(coords)
        atoms = torch.tensor(atoms, dtype=torch.long)
        y = torch.tensor(y)
        inter_feats = torch.tensor(inter_feats, dtype=torch.long)
        inter_feats = inter_feats / inter_feats.sum()
        
        edge_vec = coords[edges[0]] - coords[edges[1]]
        edge_weight = torch.norm(edge_vec, dim=-1)
        mask = edges[0] != edges[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        
        if edges.shape[1]!=dist_feat.shape[0]:
            return None
        
        graph = Data(x=features, edge_index=edges, edge_attr=dist_feat, edge_weight=edge_weight, edge_vec=edge_vec, pos=coords, y=y, atom=atoms, inter_feat=inter_feats, dataset='pdb')
        
        return graph

    def load_data(self):
        """ Generate complex interaction graphs. """
        if self.has_cache():
            graphs = self.load()
            self.graphs = graphs
        else:
            print('Processing raw protein-ligand complex data...')
            file_name = os.path.join(self.data_path, "{0}.pkl".format(self.dataset))
            with open(file_name, 'rb') as f:
                data_mols, data_Y = pickle.load(f)

            for mol, y in tqdm(zip(data_mols, data_Y)):
                graph = self.build_graph(mol, y)
                if graph is None:
                    continue
                self.graphs.append(graph)

            # self.labels = np.array(data_Y).reshape(-1, 1)
            if self.save_file:
                self.save()