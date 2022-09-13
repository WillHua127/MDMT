import os
import torch
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform



possible_atomic_num_list = list(range(1, 119))
chirality = {Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
             Chem.rdchem.ChiralType.CHI_OTHER: 0}
bonds = {Chem.rdchem.BondType.SINGLE: 0, Chem.rdchem.BondType.DOUBLE: 1, Chem.rdchem.BondType.TRIPLE: 2, Chem.rdchem.BondType.AROMATIC: 3}

def embed_func(mol, numConfs):
    AllChem.EmbedMolecule(mol)
    if len(mol.GetConformers()) == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mol = Chem.RemoveHs(mol, sanitize=True)
    if len(mol.GetConformers()) == 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol, confId=0)
    return mol
    
    
def one_k_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    #pos 
    mol_ = embed_func(mol, 1)
    if mol_ is None:
        return None
    pos = torch.tensor(mol_.GetConformer().GetPositions(), dtype=torch.float)

    ## atoms
    N = mol.GetNumAtoms()
    atom_type_idx = []
    atomic_number = []
    atom_features = []
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        atom_type_idx.append(possible_atomic_num_list.index(atom.GetAtomicNum()))
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend([atom.GetAtomicNum(),
                              1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
                                ]))
        atom_features.extend(one_k_encoding(chirality[atom.GetChiralTag()], [-1, 0, 1]))
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        #print(atom.GetAtomicNum(), atom.GetDegree(), atom.GetHybridization(), atom.GetImplicitValence(), (atom.GetFormalCharge()))
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3, 4, 5]))

    z = torch.tensor(atomic_number, dtype=torch.long)

    ## bonds 
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    #edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
    dist_feat = torch.cdist(pos[row], pos[col])[row, col].reshape(-1, 1)
    dist_feat = torch.tensor(dist_feat, dtype=torch.long)
    
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_weight = torch.norm(edge_vec, dim=-1)
    mask = edge_index[0] != edge_index[1]
    edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)

    x1 = torch.tensor(atom_type_idx).unsqueeze(1)
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)
    
    data = Data(x=x, atom=z, edge_index=edge_index, edge_attr=dist_feat, edge_weight=edge_weight, edge_vec=edge_vec, pos=pos)

    return data


class Transform(BaseTransform):
    def __init__(self, data_name):
        self.data_name = data_name
        
    def __call__(self, data):      
        data.dataset = 'chemb'
        if self.data_name in {'chembl_dense_10'}:
            data.subdata = 'chemb10'
        elif self.data_name in {'chembl_dense_50'}:
            data.subdata = 'chemb50'
        elif self.data_name in {'chembl_dense_100'}:
            data.subdata = 'chemb100'
        return data
    
class CHEMBL(InMemoryDataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.dataset = dataset
        self.root = root
        transform = Transform(dataset)
        
        super(CHEMBL, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.root)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        data_list = []

        print('Create InMemory dataset for {}.'.format(self.dataset))

        mol_prop, rdkit_mol_list, labels = load_chembl_labels(self.root)
        print('processing')
        for i in tqdm(range(len(rdkit_mol_list))):
            rdkit_mol = rdkit_mol_list[i]

            data = mol_to_graph_data_obj_simple(rdkit_mol)
            if data is None:
                continue 
            # manually add mol id
            data.id = torch.tensor([i])  # id here is the index of the mol in
            # the dataset
            data.y = torch.tensor(labels[i, :])
            data_list.append(data)
            

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        mol_prop.to_csv(os.path.join(self.processed_dir, 'mol_properties.csv'))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
def load_chembl_labels(root_path):
    mol_prop = pd.read_csv(os.path.join(root_path, 'mol_properties.csv'), index_col=0)
    print('{} molecules with SMILES'.format(len(mol_prop)))

    labels = np.load(os.path.join(root_path, 'labels.npz'))['labels']
    print('labels\t', labels.shape)

    f = open(os.path.join(root_path, 'rdkit_molecule.pkl'), 'rb')
    rdkit_mol_list = pickle.load(f)
    print('{} rdkit molecule list'.format(len(rdkit_mol_list)))

    return mol_prop, rdkit_mol_list, labels

