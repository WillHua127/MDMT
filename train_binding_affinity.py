import numpy as np
import time
import torch
import networkx as nx
import argparse
import os
import itertools
from torch_geometric import loader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


from hot_pytorch.batch.sparse import make_batch
from hot_pytorch.batch.dense import Batch as D
from hot_pytorch.models import transformer

import argparse
import datasets
from utils import rmse, mae, sd, pearson
from priors import Atomref
from splitters import *

import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss


def measure(n_fea, e_id, e_fea, e_we, e_vec, pos, atoms, nnode, mtl=False):
    out = model(n_fea, e_id, e_fea, e_we, e_vec, pos, atoms, nnode, mtl)
    return out


def train_routine(epoch, device):
    model.train()
    total_loss = 0
    count = 0
    check = 0
    
    print('Start Epoch', epoch, '...')
    #for (g, edge_vec, nnode, y) in train_list:
    for idx, graph in enumerate(tqdm(trn_loader)):
        optimizer.zero_grad()
        
        if graph.dataset[0] in {'md17'}:
            pos, atoms, nnode, y, dy, subname= graph.pos, graph.atom, graph.pos.shape[0], graph.y, graph.dy, graph.subdata[0]
            y_hat, dy_hat = model(pos, atoms, nnode, task='md17', subname=subname)
            loss_y = mse_loss(y_hat, y)
            loss_dy = mse_loss(dy_hat, dy)
            loss = 0.2*loss_y + 0.8*loss_dy
            if subname in {'aspirin'}:
                loss = asp_trn_wg * loss
            elif subname in {'ethanol'}:
                loss = eth_trn_wg * loss
            elif subname in {'malonaldehyde'}:
                loss = mal_trn_wg * loss
            elif subname in {'naphthalene'}:
                loss = nap_trn_wg * loss
            elif subname in {'salicylic_acid'}:
                loss = sal_trn_wg * loss
            elif subname in {'toluene'}:
                loss = tol_trn_wg * loss
            elif subname in {'uracil'}:
                loss = ura_trn_wg * loss
                
        elif graph.dataset[0] in {'qm9'}:
            pos, atoms, nnode, y= graph.pos, graph.atom, graph.pos.shape[0], graph.y
            e_id, e_fea, e_we, e_vec = graph.edge_index, graph.edge_attr, graph.edge_weight, graph.edge_vec
            y_hat = model(pos, atoms, nnode, task='qm9', e_id=e_id, e_fea=e_fea, e_we=e_we, e_vec=e_vec)
            loss = [mse_loss(y_hat[i].squeeze(1), y[i]) for i in range(y.shape[0])]
            loss = sum(loss)
            loss = qm9_trn_wg * loss
        elif graph.dataset[0] in {'chemb'}:
            pos, atoms, nnode, y, subname= graph.pos, graph.atom, graph.pos.shape[0], graph.y.float(), graph.subdata[0]
            n_fea, e_id, e_fea, e_we, e_vec = graph.x, graph.edge_index, graph.edge_attr, graph.edge_weight, graph.edge_vec
            y_hat = model(pos, atoms, nnode, task='chemb', subname=subname, n_fea=n_fea, e_id=e_id, e_fea=e_fea, e_we=e_we, e_vec=e_vec)
            y = y.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)
            # whether y is non-null or not.
            is_valid = y**2 > 0
            # loss matrix
            loss_mat = chemb_criterion(y_hat.double(), (y+1)/2)
            # loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            # loss for each task, some task may have 0 valid labels
            valid_task_count_list = is_valid.sum(0)
            loss_each_task_list = (loss_mat.sum(0)) / (valid_task_count_list+1e-9)
            loss = loss_each_task_list.sum()# / (valid_task_count_list > 0).sum()
            
            if subname in {'chemb10'}:
                loss = ((valid_task_count_list > 0).sum()/chemb10_trn_wg) * loss
            elif subname in {'chemb50'}:
                loss = ((valid_task_count_list > 0).sum()/chemb50_trn_wg) * loss
            elif subname in {'chemb100'}:
                loss = ((valid_task_count_list > 0).sum()/chemb100_trn_wg) * loss

        elif graph.dataset[0] in {'pdb'}:
            n_fea, e_id, e_fea, e_we, e_vec = graph.x, graph.edge_index, graph.edge_attr, graph.edge_weight, graph.edge_vec
            pos, atoms, nnode, y = graph.pos, graph.atom, graph.pos.shape[0], graph.y
            y_hat = model(pos, atoms, nnode, task='pdb', n_fea=n_fea, e_id=e_id, e_fea=e_fea, e_we=e_we, e_vec=e_vec)
            loss = F.l1_loss(y_hat, y.squeeze(), reduction='sum')#mse_loss(y_hat, y.squeeze())
            loss = pdb_trn_wg * loss
        
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        
#         count += 1
#         if count == check_every[check]:
#             print(f'Training data have complete {check_portion[check]:.5f} percent...')
#             check += 1
            
        
    return total_loss / len(trn_dataset)



def val_routine(epoch, device):
    model.eval()
    
    y_true_md17_asp_e = []
    y_pred_md17_asp_e = []
    y_true_md17_asp_f = []
    y_pred_md17_asp_f = []
    y_true_md17_eth_e = []
    y_pred_md17_eth_e = []
    y_true_md17_eth_f = []
    y_pred_md17_eth_f = []
    y_true_md17_mal_e = []
    y_pred_md17_mal_e = []
    y_true_md17_mal_f = []
    y_pred_md17_mal_f = []
    y_true_md17_nap_e = []
    y_pred_md17_nap_e = []
    y_true_md17_nap_f = []
    y_pred_md17_nap_f = []
    y_true_md17_sal_e = []
    y_pred_md17_sal_e = []
    y_true_md17_sal_f = []
    y_pred_md17_sal_f = []
    y_true_md17_tol_e = []
    y_pred_md17_tol_e = []
    y_true_md17_tol_f = []
    y_pred_md17_tol_f = []
    y_true_md17_ura_e = []
    y_pred_md17_ura_e = []
    y_true_md17_ura_f = []
    y_pred_md17_ura_f = []
    
    y_true_qm9_dip = []
    y_pred_qm9_dip = []
    y_true_qm9_pol = []
    y_pred_qm9_pol = []
    y_true_qm9_hom = []
    y_pred_qm9_hom = []
    y_true_qm9_lum = []
    y_pred_qm9_lum = []
    y_true_qm9_dlt = []
    y_pred_qm9_dlt = []
    y_true_qm9_ele = []
    y_pred_qm9_ele = []
    y_true_qm9_zpv = []
    y_pred_qm9_zpv = []
    y_true_qm9_eu0 = []
    y_pred_qm9_eu0 = []
    y_true_qm9_eu1 = []
    y_pred_qm9_eu1 = []
    y_true_qm9_ent = []
    y_pred_qm9_ent = []
    y_true_qm9_efr = []
    y_pred_qm9_efr = []
    y_true_qm9_hea = []
    y_pred_qm9_hea = []
    
    
    y_true_chemb10 = []
    y_pred_chemb10 = []
    y_true_chemb50 = []
    y_pred_chemb50 = []
    y_true_chemb100 = []
    y_pred_chemb100 = []
    
    y_true_pdb = []
    y_pred_pdb = []
    

    for idx, graph in enumerate(tqdm(val_loader)):
        if graph.dataset[0] in {'md17'}:
            pos, atoms, nnode, y, dy, subname= graph.pos, graph.atom, graph.pos.shape[0], graph.y, graph.dy, graph.subdata[0]
            y_hat, dy_hat = model(pos, atoms, nnode, task='md17', subname=subname)

            if subname in {'aspirin'}:
                y_true_md17_asp_e.append(y.item())
                y_pred_md17_asp_e.append(y_hat.item())
                y_true_md17_asp_f.append(dy.detach().numpy())
                y_pred_md17_asp_f.append(dy_hat.detach().numpy())
            elif subname in {'ethanol'}:
                y_true_md17_eth_e.append(y.item())
                y_pred_md17_eth_e.append(y_hat.item())
                y_true_md17_eth_f.append(dy.detach().numpy())
                y_pred_md17_eth_f.append(dy_hat.detach().numpy())
            elif subname in {'malonaldehyde'}:
                y_true_md17_mal_e.append(y.item())
                y_pred_md17_mal_e.append(y_hat.item())
                y_true_md17_mal_f.append(dy.detach().numpy())
                y_pred_md17_mal_f.append(dy_hat.detach().numpy())
            elif subname in {'naphthalene'}:
                y_true_md17_nap_e.append(y.item())
                y_pred_md17_nap_e.append(y_hat.item())
                y_true_md17_nap_f.append(dy.detach().numpy())
                y_pred_md17_nap_f.append(dy_hat.detach().numpy())
            elif subname in {'salicylic_acid'}:
                y_true_md17_sal_e.append(y.item())
                y_pred_md17_sal_e.append(y_hat.item())
                y_true_md17_sal_f.append(dy.detach().numpy())
                y_pred_md17_sal_f.append(dy_hat.detach().numpy())
            elif subname in {'toluene'}:
                y_true_md17_tol_e.append(y.item())
                y_pred_md17_tol_e.append(y_hat.item())
                y_true_md17_tol_f.append(dy.detach().numpy())
                y_pred_md17_tol_f.append(dy_hat.detach().numpy())
            elif subname in {'uracil'}:
                y_true_md17_ura_e.append(y.item())
                y_pred_md17_ura_e.append(y_hat.item())
                y_true_md17_ura_f.append(dy.detach().numpy())
                y_pred_md17_ura_f.append(dy_hat.detach().numpy())

        elif graph.dataset[0] in {'qm9'}:
            with torch.no_grad():
                pos, atoms, nnode, y= graph.pos, graph.atom, graph.pos.shape[0], graph.y
                e_id, e_fea, e_we, e_vec = graph.edge_index, graph.edge_attr, graph.edge_weight, graph.edge_vec
                y_hat = model(pos, atoms, nnode, task='qm9', e_id=e_id, e_fea=e_fea, e_we=e_we, e_vec=e_vec)

                y_true_qm9_dip.append(y[0].item())
                y_pred_qm9_dip.append(y_hat[0].item())
                y_true_qm9_pol.append(y[1].item())
                y_pred_qm9_pol.append(y_hat[1].item())
                y_true_qm9_hom.append(y[2].item())
                y_pred_qm9_hom.append(y_hat[2].item())
                y_true_qm9_lum.append(y[3].item())
                y_pred_qm9_lum.append(y_hat[3].item())
                y_true_qm9_dlt.append(y[4].item())
                y_pred_qm9_dlt.append(y_hat[4].item())
                y_true_qm9_ele.append(y[5].item())
                y_pred_qm9_ele.append(y_hat[5].item())
                y_true_qm9_zpv.append(y[6].item())
                y_pred_qm9_zpv.append(y_hat[6].item())
                y_true_qm9_eu0.append(y[7].item())
                y_pred_qm9_eu0.append(y_hat[7].item())
                y_true_qm9_eu1.append(y[8].item())
                y_pred_qm9_eu1.append(y_hat[8].item())
                y_true_qm9_ent.append(y[9].item())
                y_pred_qm9_ent.append(y_hat[9].item())
                y_true_qm9_efr.append(y[10].item())
                y_pred_qm9_efr.append(y_hat[10].item())
                y_true_qm9_hea.append(y[11].item())
                y_pred_qm9_hea.append(y_hat[11].item())

        elif graph.dataset[0] in {'chemb'}:
            with torch.no_grad():
                pos, atoms, nnode, y, subname= graph.pos, graph.atom, graph.pos.shape[0], graph.y.float(), graph.subdata[0]
                n_fea, e_id, e_fea, e_we, e_vec = graph.x, graph.edge_index, graph.edge_attr, graph.edge_weight, graph.edge_vec
                y_hat = model(pos, atoms, nnode, task='chemb', subname=subname, n_fea=n_fea, e_id=e_id, e_fea=e_fea, e_we=e_we, e_vec=e_vec)
                y = y.unsqueeze(0)
                y_hat = y_hat.unsqueeze(0)

                if subname in {'chemb10'}:
                    y_true_chemb10.append(y.detach().numpy())
                    y_pred_chemb10.append(y_hat.detach().numpy())
                elif subname in {'chemb50'}:
                    y_true_chemb50.append(y.detach().numpy())
                    y_pred_chemb50.append(y_hat.detach().numpy())
                elif subname in {'chemb100'}:
                    y_true_chemb100.append(y.detach().numpy())
                    y_pred_chemb100.append(y_hat.detach().numpy())

        elif graph.dataset[0] in {'pdb'}:
            with torch.no_grad():
                n_fea, e_id, e_fea, e_we, e_vec = graph.x, graph.edge_index, graph.edge_attr, graph.edge_weight, graph.edge_vec
                pos, atoms, nnode, y = graph.pos, graph.atom, graph.pos.shape[0], graph.y
                y_hat = model(pos, atoms, nnode, task='pdb', n_fea=n_fea, e_id=e_id, e_fea=e_fea, e_we=e_we, e_vec=e_vec)

                y_true_pdb.append(y.item())
                y_pred_pdb.append(y_hat.item())
        
        
        
    y_true_qm9_dip = np.array(y_true_qm9_dip)
    y_pred_qm9_dip = np.array(y_pred_qm9_dip)
    y_true_qm9_pol = np.array(y_true_qm9_pol)
    y_pred_qm9_pol = np.array(y_pred_qm9_pol)
    y_true_qm9_hom = np.array(y_true_qm9_hom)
    y_pred_qm9_hom = np.array(y_pred_qm9_hom)
    y_true_qm9_lum = np.array(y_true_qm9_lum)
    y_pred_qm9_lum = np.array(y_pred_qm9_lum)
    y_true_qm9_dlt = np.array(y_true_qm9_dlt)
    y_pred_qm9_dlt = np.array(y_pred_qm9_dlt)
    y_true_qm9_ele = np.array(y_true_qm9_ele)
    y_pred_qm9_ele = np.array(y_pred_qm9_ele)
    y_true_qm9_zpv = np.array(y_true_qm9_zpv)
    y_pred_qm9_zpv = np.array(y_pred_qm9_zpv)
    y_true_qm9_eu0 = np.array(y_true_qm9_eu0)
    y_pred_qm9_eu0 = np.array(y_pred_qm9_eu0)
    y_true_qm9_eu1 = np.array(y_true_qm9_eu1)
    y_pred_qm9_eu1 = np.array(y_pred_qm9_eu1)
    y_true_qm9_ent = np.array(y_true_qm9_ent)
    y_pred_qm9_ent = np.array(y_pred_qm9_ent)
    y_true_qm9_efr = np.array(y_true_qm9_efr)
    y_pred_qm9_efr = np.array(y_pred_qm9_efr)
    y_true_qm9_hea = np.array(y_true_qm9_hea)
    y_pred_qm9_hea = np.array(y_pred_qm9_hea)
    
    y_true_md17_asp_e = np.array(y_true_md17_asp_e)
    y_pred_md17_asp_e = np.array(y_pred_md17_asp_e)
    y_true_md17_asp_f = np.concatenate(y_true_md17_asp_f, axis=0)
    y_pred_md17_asp_f = np.concatenate(y_pred_md17_asp_f, axis=0)
    y_true_md17_eth_e = np.array(y_true_md17_eth_e)
    y_pred_md17_eth_e = np.array(y_pred_md17_eth_e)
    y_true_md17_eth_f = np.concatenate(y_true_md17_eth_f, axis=0)
    y_pred_md17_eth_f = np.concatenate(y_pred_md17_eth_f, axis=0)
    y_true_md17_mal_e = np.array(y_true_md17_mal_e)
    y_pred_md17_mal_e = np.array(y_pred_md17_mal_e)
    y_true_md17_mal_f = np.concatenate(y_true_md17_mal_f, axis=0)
    y_pred_md17_mal_f = np.concatenate(y_pred_md17_mal_f, axis=0)
    y_true_md17_nap_e = np.array(y_true_md17_nap_e)
    y_pred_md17_nap_e = np.array(y_pred_md17_nap_e)
    y_true_md17_nap_f = np.concatenate(y_true_md17_nap_f, axis=0)
    y_pred_md17_nap_f = np.concatenate(y_pred_md17_nap_f, axis=0)
    y_true_md17_sal_e = np.array(y_true_md17_sal_e)
    y_pred_md17_sal_e = np.array(y_pred_md17_sal_e)
    y_true_md17_sal_f = np.concatenate(y_true_md17_sal_f, axis=0)
    y_pred_md17_sal_f = np.concatenate(y_pred_md17_sal_f, axis=0)
    y_true_md17_tol_e = np.array(y_true_md17_tol_e)
    y_pred_md17_tol_e = np.array(y_pred_md17_tol_e)
    y_true_md17_tol_f = np.concatenate(y_true_md17_tol_f, axis=0)
    y_pred_md17_tol_f = np.concatenate(y_pred_md17_tol_f, axis=0)
    y_true_md17_ura_e = np.array(y_true_md17_ura_e)
    y_pred_md17_ura_e = np.array(y_pred_md17_ura_e)
    y_true_md17_ura_f = np.concatenate(y_true_md17_ura_f, axis=0)
    y_pred_md17_ura_f = np.concatenate(y_pred_md17_ura_f, axis=0)
    
    y_true_chemb10 = np.concatenate(y_true_chemb10, axis=0)
    y_pred_chemb10 = np.concatenate(y_pred_chemb10, axis=0)
    y_true_chemb50 = np.concatenate(y_true_chemb50, axis=0)
    y_pred_chemb50 = np.concatenate(y_pred_chemb50, axis=0)
    y_true_chemb100 = np.concatenate(y_true_chemb100, axis=0)
    y_pred_chemb100 = np.concatenate(y_pred_chemb100, axis=0)
    
    y_true_pdb = np.array(y_true_pdb)
    y_pred_pdb = np.array(y_pred_pdb)
    
    
    l1_qm9_dip = mae(y_true_qm9_dip, y_pred_qm9_dip)
    l1_qm9_pol = mae(y_true_qm9_pol, y_pred_qm9_pol)
    l1_qm9_hom = mae(y_true_qm9_hom, y_pred_qm9_hom)
    l1_qm9_lum = mae(y_true_qm9_lum, y_pred_qm9_lum)
    l1_qm9_dlt = mae(y_true_qm9_dlt, y_pred_qm9_dlt)
    l1_qm9_ele = mae(y_true_qm9_ele, y_pred_qm9_ele)
    l1_qm9_zpv = mae(y_true_qm9_zpv, y_pred_qm9_zpv)
    l1_qm9_eu0 = mae(y_true_qm9_eu0, y_pred_qm9_eu0)
    l1_qm9_eu1 = mae(y_true_qm9_eu1, y_pred_qm9_eu1)
    l1_qm9_ent = mae(y_true_qm9_ent, y_pred_qm9_ent)
    l1_qm9_efr = mae(y_true_qm9_efr, y_pred_qm9_efr)
    l1_qm9_hea = mae(y_true_qm9_hea, y_pred_qm9_hea)
    
    l1_md17_asp_e = mae(y_true_md17_asp_e, y_pred_md17_asp_e)
    l1_md17_asp_f = mae(y_true_md17_asp_f, y_pred_md17_asp_f)
    l1_md17_eth_e = mae(y_true_md17_eth_e, y_pred_md17_eth_e)
    l1_md17_eth_f = mae(y_true_md17_eth_f, y_pred_md17_eth_f)
    l1_md17_mal_e = mae(y_true_md17_mal_e, y_pred_md17_mal_e)
    l1_md17_mal_f = mae(y_true_md17_mal_f, y_pred_md17_mal_f)
    l1_md17_nap_e = mae(y_pred_md17_nap_e, y_pred_md17_nap_e)
    l1_md17_nap_f = mae(y_true_md17_nap_f, y_pred_md17_nap_f)
    l1_md17_sal_e = mae(y_true_md17_sal_e, y_pred_md17_sal_e)
    l1_md17_sal_f = mae(y_true_md17_sal_f, y_pred_md17_sal_f)
    l1_md17_tol_e = mae(y_true_md17_tol_e, y_pred_md17_tol_e)
    l1_md17_tol_f = mae(y_true_md17_tol_f, y_pred_md17_tol_f)
    l1_md17_ura_e = mae(y_true_md17_ura_e, y_pred_md17_ura_e)
    l1_md17_ura_f = mae(y_true_md17_ura_f, y_pred_md17_ura_f)
    
    roc_chemb10 = roc(y_true_chemb10, y_pred_chemb10)
    roc_chemb50 = roc(y_true_chemb50, y_pred_chemb50)
    roc_chemb100 = roc(y_true_chemb100, y_pred_chemb100)
    
    rmse_pdb = rmse(y_true_pdb, y_pred_pdb)
    mae_pdb = mae(y_true_pdb, y_pred_pdb)
    sd_pdb = sd(y_true_pdb, y_pred_pdb)
    pearson_pdb = pearson(y_true_pdb, y_pred_pdb)
    return l1_qm9_dip, l1_qm9_pol, l1_qm9_hom, l1_qm9_lum, l1_qm9_dlt, l1_qm9_ele, l1_qm9_zpv, l1_qm9_eu0, l1_qm9_eu1, l1_qm9_ent, l1_qm9_efr, l1_qm9_hea, l1_md17_asp_e, l1_md17_asp_f, l1_md17_eth_e, l1_md17_eth_f, l1_md17_mal_e, l1_md17_mal_f, l1_md17_nap_e, l1_md17_nap_f, l1_md17_sal_e, l1_md17_sal_f, l1_md17_tol_e, l1_md17_tol_f, l1_md17_ura_e, l1_md17_ura_f, roc_chemb10, roc_chemb50, roc_chemb100, rmse_pdb, mae_pdb, sd_pdb, pearson_pdb



def test_routine(epoch, device):
    model.eval()
    
    y_true_md17_asp_e = []
    y_pred_md17_asp_e = []
    y_true_md17_asp_f = []
    y_pred_md17_asp_f = []
    y_true_md17_eth_e = []
    y_pred_md17_eth_e = []
    y_true_md17_eth_f = []
    y_pred_md17_eth_f = []
    y_true_md17_mal_e = []
    y_pred_md17_mal_e = []
    y_true_md17_mal_f = []
    y_pred_md17_mal_f = []
    y_true_md17_nap_e = []
    y_pred_md17_nap_e = []
    y_true_md17_nap_f = []
    y_pred_md17_nap_f = []
    y_true_md17_sal_e = []
    y_pred_md17_sal_e = []
    y_true_md17_sal_f = []
    y_pred_md17_sal_f = []
    y_true_md17_tol_e = []
    y_pred_md17_tol_e = []
    y_true_md17_tol_f = []
    y_pred_md17_tol_f = []
    y_true_md17_ura_e = []
    y_pred_md17_ura_e = []
    y_true_md17_ura_f = []
    y_pred_md17_ura_f = []
    
    y_true_qm9_dip = []
    y_pred_qm9_dip = []
    y_true_qm9_pol = []
    y_pred_qm9_pol = []
    y_true_qm9_hom = []
    y_pred_qm9_hom = []
    y_true_qm9_lum = []
    y_pred_qm9_lum = []
    y_true_qm9_dlt = []
    y_pred_qm9_dlt = []
    y_true_qm9_ele = []
    y_pred_qm9_ele = []
    y_true_qm9_zpv = []
    y_pred_qm9_zpv = []
    y_true_qm9_eu0 = []
    y_pred_qm9_eu0 = []
    y_true_qm9_eu1 = []
    y_pred_qm9_eu1 = []
    y_true_qm9_ent = []
    y_pred_qm9_ent = []
    y_true_qm9_efr = []
    y_pred_qm9_efr = []
    y_true_qm9_hea = []
    y_pred_qm9_hea = []
    
    
    y_true_chemb10 = []
    y_pred_chemb10 = []
    y_true_chemb50 = []
    y_pred_chemb50 = []
    y_true_chemb100 = []
    y_pred_chemb100 = []
    
    y_true_pdb = []
    y_pred_pdb = []
    
    for idx, graph in enumerate(tqdm(tst_loader)):
        if graph.dataset[0] in {'md17'}:
            pos, atoms, nnode, y, dy, subname= graph.pos, graph.atom, graph.pos.shape[0], graph.y, graph.dy, graph.subdata[0]
            y_hat, dy_hat = model(pos, atoms, nnode, task='md17', subname=subname)

            if subname in {'aspirin'}:
                y_true_md17_asp_e.append(y.item())
                y_pred_md17_asp_e.append(y_hat.item())
                y_true_md17_asp_f.append(dy.detach().numpy())
                y_pred_md17_asp_f.append(dy_hat.detach().numpy())
            elif subname in {'ethanol'}:
                y_true_md17_eth_e.append(y.item())
                y_pred_md17_eth_e.append(y_hat.item())
                y_true_md17_eth_f.append(dy.detach().numpy())
                y_pred_md17_eth_f.append(dy_hat.detach().numpy())
            elif subname in {'malonaldehyde'}:
                y_true_md17_mal_e.append(y.item())
                y_pred_md17_mal_e.append(y_hat.item())
                y_true_md17_mal_f.append(dy.detach().numpy())
                y_pred_md17_mal_f.append(dy_hat.detach().numpy())
            elif subname in {'naphthalene'}:
                y_true_md17_nap_e.append(y.item())
                y_pred_md17_nap_e.append(y_hat.item())
                y_true_md17_nap_f.append(dy.detach().numpy())
                y_pred_md17_nap_f.append(dy_hat.detach().numpy())
            elif subname in {'salicylic_acid'}:
                y_true_md17_sal_e.append(y.item())
                y_pred_md17_sal_e.append(y_hat.item())
                y_true_md17_sal_f.append(dy.detach().numpy())
                y_pred_md17_sal_f.append(dy_hat.detach().numpy())
            elif subname in {'toluene'}:
                y_true_md17_tol_e.append(y.item())
                y_pred_md17_tol_e.append(y_hat.item())
                y_true_md17_tol_f.append(dy.detach().numpy())
                y_pred_md17_tol_f.append(dy_hat.detach().numpy())
            elif subname in {'uracil'}:
                y_true_md17_ura_e.append(y.item())
                y_pred_md17_ura_e.append(y_hat.item())
                y_true_md17_ura_f.append(dy.detach().numpy())
                y_pred_md17_ura_f.append(dy_hat.detach().numpy())

        elif graph.dataset[0] in {'qm9'}:
            with torch.no_grad():
                pos, atoms, nnode, y= graph.pos, graph.atom, graph.pos.shape[0], graph.y
                e_id, e_fea, e_we, e_vec = graph.edge_index, graph.edge_attr, graph.edge_weight, graph.edge_vec
                y_hat = model(pos, atoms, nnode, task='qm9', e_id=e_id, e_fea=e_fea, e_we=e_we, e_vec=e_vec)

                y_true_qm9_dip.append(y[0].item())
                y_pred_qm9_dip.append(y_hat[0].item())
                y_true_qm9_pol.append(y[1].item())
                y_pred_qm9_pol.append(y_hat[1].item())
                y_true_qm9_hom.append(y[2].item())
                y_pred_qm9_hom.append(y_hat[2].item())
                y_true_qm9_lum.append(y[3].item())
                y_pred_qm9_lum.append(y_hat[3].item())
                y_true_qm9_dlt.append(y[4].item())
                y_pred_qm9_dlt.append(y_hat[4].item())
                y_true_qm9_ele.append(y[5].item())
                y_pred_qm9_ele.append(y_hat[5].item())
                y_true_qm9_zpv.append(y[6].item())
                y_pred_qm9_zpv.append(y_hat[6].item())
                y_true_qm9_eu0.append(y[7].item())
                y_pred_qm9_eu0.append(y_hat[7].item())
                y_true_qm9_eu1.append(y[8].item())
                y_pred_qm9_eu1.append(y_hat[8].item())
                y_true_qm9_ent.append(y[9].item())
                y_pred_qm9_ent.append(y_hat[9].item())
                y_true_qm9_efr.append(y[10].item())
                y_pred_qm9_efr.append(y_hat[10].item())
                y_true_qm9_hea.append(y[11].item())
                y_pred_qm9_hea.append(y_hat[11].item())

        elif graph.dataset[0] in {'chemb'}:
            with torch.no_grad():
                pos, atoms, nnode, y, subname= graph.pos, graph.atom, graph.pos.shape[0], graph.y.float(), graph.subdata[0]
                n_fea, e_id, e_fea, e_we, e_vec = graph.x, graph.edge_index, graph.edge_attr, graph.edge_weight, graph.edge_vec
                y_hat = model(pos, atoms, nnode, task='chemb', subname=subname, n_fea=n_fea, e_id=e_id, e_fea=e_fea, e_we=e_we, e_vec=e_vec)
                y = y.unsqueeze(0)
                y_hat = y_hat.unsqueeze(0)

                if subname in {'chemb10'}:
                    y_true_chemb10.append(y.detach().numpy())
                    y_pred_chemb10.append(y_hat.detach().numpy())
                elif subname in {'chemb50'}:
                    y_true_chemb50.append(y.detach().numpy())
                    y_pred_chemb50.append(y_hat.detach().numpy())
                elif subname in {'chemb100'}:
                    y_true_chemb100.append(y.detach().numpy())
                    y_pred_chemb100.append(y_hat.detach().numpy())

        elif graph.dataset[0] in {'pdb'}:
            with torch.no_grad():
                n_fea, e_id, e_fea, e_we, e_vec = graph.x, graph.edge_index, graph.edge_attr, graph.edge_weight, graph.edge_vec
                pos, atoms, nnode, y = graph.pos, graph.atom, graph.pos.shape[0], graph.y
                y_hat = model(pos, atoms, nnode, task='pdb', n_fea=n_fea, e_id=e_id, e_fea=e_fea, e_we=e_we, e_vec=e_vec)

                y_true_pdb.append(y.item())
                y_pred_pdb.append(y_hat.item())
        
        
        
    y_true_qm9_dip = np.array(y_true_qm9_dip)
    y_pred_qm9_dip = np.array(y_pred_qm9_dip)
    y_true_qm9_pol = np.array(y_true_qm9_pol)
    y_pred_qm9_pol = np.array(y_pred_qm9_pol)
    y_true_qm9_hom = np.array(y_true_qm9_hom)
    y_pred_qm9_hom = np.array(y_pred_qm9_hom)
    y_true_qm9_lum = np.array(y_true_qm9_lum)
    y_pred_qm9_lum = np.array(y_pred_qm9_lum)
    y_true_qm9_dlt = np.array(y_true_qm9_dlt)
    y_pred_qm9_dlt = np.array(y_pred_qm9_dlt)
    y_true_qm9_ele = np.array(y_true_qm9_ele)
    y_pred_qm9_ele = np.array(y_pred_qm9_ele)
    y_true_qm9_zpv = np.array(y_true_qm9_zpv)
    y_pred_qm9_zpv = np.array(y_pred_qm9_zpv)
    y_true_qm9_eu0 = np.array(y_true_qm9_eu0)
    y_pred_qm9_eu0 = np.array(y_pred_qm9_eu0)
    y_true_qm9_eu1 = np.array(y_true_qm9_eu1)
    y_pred_qm9_eu1 = np.array(y_pred_qm9_eu1)
    y_true_qm9_ent = np.array(y_true_qm9_ent)
    y_pred_qm9_ent = np.array(y_pred_qm9_ent)
    y_true_qm9_efr = np.array(y_true_qm9_efr)
    y_pred_qm9_efr = np.array(y_pred_qm9_efr)
    y_true_qm9_hea = np.array(y_true_qm9_hea)
    y_pred_qm9_hea = np.array(y_pred_qm9_hea)
    
    y_true_md17_asp_e = np.array(y_true_md17_asp_e)
    y_pred_md17_asp_e = np.array(y_pred_md17_asp_e)
    y_true_md17_asp_f = np.concatenate(y_true_md17_asp_f, axis=0)
    y_pred_md17_asp_f = np.concatenate(y_pred_md17_asp_f, axis=0)
    y_true_md17_eth_e = np.array(y_true_md17_eth_e)
    y_pred_md17_eth_e = np.array(y_pred_md17_eth_e)
    y_true_md17_eth_f = np.concatenate(y_true_md17_eth_f, axis=0)
    y_pred_md17_eth_f = np.concatenate(y_pred_md17_eth_f, axis=0)
    y_true_md17_mal_e = np.array(y_true_md17_mal_e)
    y_pred_md17_mal_e = np.array(y_pred_md17_mal_e)
    y_true_md17_mal_f = np.concatenate(y_true_md17_mal_f, axis=0)
    y_pred_md17_mal_f = np.concatenate(y_pred_md17_mal_f, axis=0)
    y_true_md17_nap_e = np.array(y_true_md17_nap_e)
    y_pred_md17_nap_e = np.array(y_pred_md17_nap_e)
    y_true_md17_nap_f = np.concatenate(y_true_md17_nap_f, axis=0)
    y_pred_md17_nap_f = np.concatenate(y_pred_md17_nap_f, axis=0)
    y_true_md17_sal_e = np.array(y_true_md17_sal_e)
    y_pred_md17_sal_e = np.array(y_pred_md17_sal_e)
    y_true_md17_sal_f = np.concatenate(y_true_md17_sal_f, axis=0)
    y_pred_md17_sal_f = np.concatenate(y_pred_md17_sal_f, axis=0)
    y_true_md17_tol_e = np.array(y_true_md17_tol_e)
    y_pred_md17_tol_e = np.array(y_pred_md17_tol_e)
    y_true_md17_tol_f = np.concatenate(y_true_md17_tol_f, axis=0)
    y_pred_md17_tol_f = np.concatenate(y_pred_md17_tol_f, axis=0)
    y_true_md17_ura_e = np.array(y_true_md17_ura_e)
    y_pred_md17_ura_e = np.array(y_pred_md17_ura_e)
    y_true_md17_ura_f = np.concatenate(y_true_md17_ura_f, axis=0)
    y_pred_md17_ura_f = np.concatenate(y_pred_md17_ura_f, axis=0)
    
    y_true_chemb10 = np.concatenate(y_true_chemb10, axis=0)
    y_pred_chemb10 = np.concatenate(y_pred_chemb10, axis=0)
    y_true_chemb50 = np.concatenate(y_true_chemb50, axis=0)
    y_pred_chemb50 = np.concatenate(y_pred_chemb50, axis=0)
    y_true_chemb100 = np.concatenate(y_true_chemb100, axis=0)
    y_pred_chemb100 = np.concatenate(y_pred_chemb100, axis=0)
                                    
    y_true_pdb = np.array(y_true_pdb)
    y_pred_pdb = np.array(y_pred_pdb)
    
    
    l1_qm9_dip = mae(y_true_qm9_dip, y_pred_qm9_dip)
    l1_qm9_pol = mae(y_true_qm9_pol, y_pred_qm9_pol)
    l1_qm9_hom = mae(y_true_qm9_hom, y_pred_qm9_hom)
    l1_qm9_lum = mae(y_true_qm9_lum, y_pred_qm9_lum)
    l1_qm9_dlt = mae(y_true_qm9_dlt, y_pred_qm9_dlt)
    l1_qm9_ele = mae(y_true_qm9_ele, y_pred_qm9_ele)
    l1_qm9_zpv = mae(y_true_qm9_zpv, y_pred_qm9_zpv)
    l1_qm9_eu0 = mae(y_true_qm9_eu0, y_pred_qm9_eu0)
    l1_qm9_eu1 = mae(y_true_qm9_eu1, y_pred_qm9_eu1)
    l1_qm9_ent = mae(y_true_qm9_ent, y_pred_qm9_ent)
    l1_qm9_efr = mae(y_true_qm9_efr, y_pred_qm9_efr)
    l1_qm9_hea = mae(y_true_qm9_hea, y_pred_qm9_hea)
    
    l1_md17_asp_e = mae(y_true_md17_asp_e, y_pred_md17_asp_e)
    l1_md17_asp_f = mae(y_true_md17_asp_f, y_pred_md17_asp_f)
    l1_md17_eth_e = mae(y_true_md17_eth_e, y_pred_md17_eth_e)
    l1_md17_eth_f = mae(y_true_md17_eth_f, y_pred_md17_eth_f)
    l1_md17_mal_e = mae(y_true_md17_mal_e, y_pred_md17_mal_e)
    l1_md17_mal_f = mae(y_true_md17_mal_f, y_pred_md17_mal_f)
    l1_md17_nap_e = mae(y_pred_md17_nap_e, y_pred_md17_nap_e)
    l1_md17_nap_f = mae(y_true_md17_nap_f, y_pred_md17_nap_f)
    l1_md17_sal_e = mae(y_true_md17_sal_e, y_pred_md17_sal_e)
    l1_md17_sal_f = mae(y_true_md17_sal_f, y_pred_md17_sal_f)
    l1_md17_tol_e = mae(y_true_md17_tol_e, y_pred_md17_tol_e)
    l1_md17_tol_f = mae(y_true_md17_tol_f, y_pred_md17_tol_f)
    l1_md17_ura_e = mae(y_true_md17_ura_e, y_pred_md17_ura_e)
    l1_md17_ura_f = mae(y_true_md17_ura_f, y_pred_md17_ura_f)
    
    roc_chemb10 = roc(y_true_chemb10, y_pred_chemb10)
    roc_chemb50 = roc(y_true_chemb50, y_pred_chemb50)
    roc_chemb100 = roc(y_true_chemb100, y_pred_chemb100)
    
    rmse_pdb = rmse(y_true_pdb, y_pred_pdb)
    mae_pdb = mae(y_true_pdb, y_pred_pdb)
    sd_pdb = sd(y_true_pdb, y_pred_pdb)
    pearson_pdb = pearson(y_true_pdb, y_pred_pdb)
    return l1_qm9_dip, l1_qm9_pol, l1_qm9_hom, l1_qm9_lum, l1_qm9_dlt, l1_qm9_ele, l1_qm9_zpv, l1_qm9_eu0, l1_qm9_eu1, l1_qm9_ent, l1_qm9_efr, l1_qm9_hea, l1_md17_asp_e, l1_md17_asp_f, l1_md17_eth_e, l1_md17_eth_f, l1_md17_mal_e, l1_md17_mal_f, l1_md17_nap_e, l1_md17_nap_f, l1_md17_sal_e, l1_md17_sal_f, l1_md17_tol_e, l1_md17_tol_f, l1_md17_ura_e, l1_md17_ura_f, roc_chemb10, roc_chemb50, roc_chemb100, rmse_pdb, mae_pdb, sd_pdb, pearson_pdb


def roc(y_true, y_pred):
    roc_list = []
    invalid_count = 0
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            true = (y_true[is_valid,i] + 1)/2
            pred = y_pred[is_valid,i]
            if np.isnan(true).any() or np.isnan(pred).any() or np.isinf(true).any() or np.isinf(pred).any():
                continue
            roc_list.append(roc_auc_score(true, pred))
        else:
            invalid_count += 1

    print('Invalid task count:\t', invalid_count)

#     if len(roc_list) < y_true.shape[1]:
#         print('Some target is missing!')
#         print('Missing ratio: %f' %(1 - float(len(roc_list))/y_true.shape[1]))

    roc_list = np.array(roc_list)
    roc_value = np.mean(roc_list)
    return roc_value


def main():
    rmse_val_best, res_tst_best = 1e9, ''
    running_log = ''
    for epoch in range(1, args.epochs + 1):
        train_start = time.time()
        train_loss = train_routine(epoch, device)
        train_end = time.time()
        
        
        test_start = time.time()
        dip_tst, pol_tst, hom_tst, lum_tst, dlt_tst, ele_tst, zpv_tst, eu0_tst, eu1_tst, ent_tst, efr_tst, hea_tst, asp_e_tst, asp_f_tst, eth_e_tst, eth_f_tst, mal_e_tst, mal_f_tst, nap_e_tst, nap_f_tst, sal_e_tst, sal_f_tst, tol_e_tst, tol_f_tst, ura_e_tst, ura_f_tst, chemb10_tst, chemb50_tst, chemb100_tst, rmse_tst, mae_tst, sd_tst, pearson_tst = test_routine(epoch, device)
        test_end = time.time()
        
        val_start = time.time()
        dip_val, pol_val, hom_val, lum_val, dlt_val, ele_val, zpv_val, eu0_val, eu1_val, ent_val, efr_val, hea_val, asp_e_val, asp_f_val, eth_e_val, eth_f_val, mal_e_val, mal_f_val, nap_e_val, nap_f_val, sal_e_val, sal_f_val, tol_e_val, tol_f_val, ura_e_val, ura_f_val, chemb10_val, chemb50_val, chemb100_val, rmse_val, mae_val, sd_val, pearson_val = val_routine(epoch, device)
        val_end = time.time()
        
        scheduler.step(train_loss)
        
        log = '-----------------------------------------------------------------------\n'
        log += 'Epoch: %d, train_loss: %.4f, train_time: %.4f, val_time: %.4f, test_time: %.4f.\n' % (
                epoch, train_loss, train_end-train_start, val_end-val_start, test_end-test_start)
        log += 'Val - DIP: %.6f, POL: %.6f, HOM: %.6f, LUM: %.6f, DLT: %.6f, ELE: %.6f, ZPV: %.6f, EU0: %.6f, EU1: %.6f, ENT: %.6f, EFR: %.6f, HEA: %.6f, \n asp_E: %.6f, asp_F: %.6f, eth_E: %.6f, eth_F: %.6f, mal_E: %.6f, mal_F: %.6f, nap_E: %.6f, nap_F: %.6f, sal_E: %.6f, sal_F: %.6f, tol_E: %.6f, tol_F: %.6f, ura_E: %.6f, ura_F: %.6f, \n chemb10: %.6f, chemb50: %.6f, chemb100: %.6f, \n pdb_RMSE: %.6f, pdb_MAE: %.6f, pdb_SD: %.6f, pdb_R: %.6f.\n' % (dip_val, pol_val, hom_val, lum_val, dlt_val, ele_val, zpv_val, eu0_val, eu1_val, ent_val, efr_val, hea_val, asp_e_val, asp_f_val, eth_e_val, eth_f_val, mal_e_val, mal_f_val, nap_e_val, nap_f_val, sal_e_val, sal_f_val, tol_e_val, tol_f_val, ura_e_val, ura_f_val, chemb10_val, chemb50_val, chemb100_val, rmse_val, mae_val, sd_val, pearson_val)
        
        log += '\n'
        
        log += 'Test - DIP: %.6f, POL: %.6f, HOM: %.6f, LUM: %.6f, DLT: %.6f, ELE: %.6f, ZPV: %.6f, EU0: %.6f, EU1: %.6f, ENT: %.6f, EFR: %.6f, HEA: %.6f, \n asp_E: %.6f, asp_F: %.6f, eth_E: %.6f, eth_F: %.6f, mal_E: %.6f, mal_F: %.6f, nap_E: %.6f, nap_F: %.6f, sal_E: %.6f, sal_F: %.6f, tol_E: %.6f, tol_F: %.6f, ura_E: %.6f, ura_F: %.6f, \n chemb10: %.6f, chemb50: %.6f, chemb100: %.6f, \n pdb_RMSE: %.6f, pdb_MAE: %.6f, pdb_SD: %.6f, pdb_R: %.6f.\n' % (dip_tst, pol_tst, hom_tst, lum_tst, dlt_tst, ele_tst, zpv_tst, eu0_tst, eu1_tst, ent_tst, efr_tst, hea_tst, asp_e_tst, asp_f_tst, eth_e_tst, eth_f_tst, mal_e_tst, mal_f_tst, nap_e_tst, nap_f_tst, sal_e_tst, sal_f_tst, tol_e_tst, tol_f_tst, ura_e_tst, ura_f_tst, chemb10_tst, chemb50_tst, chemb100_tst, rmse_tst, mae_tst, sd_tst, pearson_tst)
        print(log)
        
#         if rmse_val < rmse_val_best:
#             rmse_val_best = rmse_val
#             res_tst_best = 'Best - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n' % (rmse_test, mae_test, sd_test, r_test)
#             if args.save_model:
#                 obj = {'model': model.state_dict(), 'epoch': epoch}
#                 path = os.path.join(args.model_dir, f'lr{args.lr}_cut{args.cut_dist}_layer{args.complex_layer}_head{args.complex_head}_hidden{args.complex_hidden}_model')
#                 torch.save(obj, path)
#                 # model.save(os.path.join(args.model_dir, 'saved_model'))

#         running_log += log
#         f = open(os.path.join(args.model_dir, f'lr{args.lr}_cut{args.cut_dist}_layer{args.complex_layer}_head{args.complex_head}_hidden{args.complex_hidden}_log.txt'), 'w')
#         f.write(running_log)
#         f.close()

#     f = open(os.path.join(args.model_dir, f'lr{args.lr}_cut{args.cut_dist}_layer{args.complex_layer}_head{args.complex_head}_hidden{args.complex_hidden}_log.txt'), 'w')
#     f.write(running_log + res_tst_best)
#     f.close()
    
    
def stats(mols):
    y_stats = torch.cat([mol.y for mol in mols])
    mean = y_stats.mean(dim=0)
    std = y_stats.std(dim=0)
    return mean, std

def get_qm9_prior(dataset):
    prior_models = []
    for idx in range(12): #12 tasks for qm9
        prior_models.append(Atomref(max_z=100, dataset=dataset, idx=idx))
    return prior_models
        
def pdb_data():
    pdb_trn = getattr(datasets, 'PDB')('./data/processed/', "%s_train" % 'pdbbind2016', args.cut_dist)[0:10]
    pdb_tst = getattr(datasets, 'PDB')('./data/processed/', "%s_test" % 'pdbbind2016', args.cut_dist)[0:10]
    pdb_val = getattr(datasets, 'PDB')('./data/processed/', "%s_val" % 'pdbbind2016', args.cut_dist)[0:10]
    return pdb_trn, pdb_tst, pdb_val

def chemb_data():
    chemb10_data = getattr(datasets, 'CHEMBL')(root='./data/chembl_dense_10', dataset='chembl_dense_10')[0:10]
    chemb50_data = getattr(datasets, 'CHEMBL')(root='./data/chembl_dense_50', dataset='chembl_dense_50')[0:10]
    chemb100_data = getattr(datasets, 'CHEMBL')(root='./data/chembl_dense_100', dataset='chembl_dense_100')[0:10]
    
    chemb10_trn_idx, chemb10_val_idx, chemb10_tst_idx = random_filtered_split(chemb10_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    chemb50_trn_idx, chemb50_val_idx, chemb50_tst_idx = random_filtered_split(chemb50_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    chemb100_trn_idx, chemb100_val_idx, chemb100_tst_idx = random_filtered_split(chemb100_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    
    chemb10_trn = chemb10_data[chemb10_trn_idx]
    chemb10_val = chemb10_data[chemb10_val_idx]
    chemb10_tst = chemb10_data[chemb10_tst_idx]
    
    chemb50_trn = chemb50_data[chemb50_trn_idx]
    chemb50_val = chemb50_data[chemb50_val_idx]
    chemb50_tst = chemb50_data[chemb50_tst_idx]
    
    chemb100_trn = chemb100_data[chemb100_trn_idx]
    chemb100_val = chemb100_data[chemb100_val_idx]
    chemb100_tst = chemb100_data[chemb100_tst_idx]
    
    return chemb10_trn, chemb10_val, chemb10_tst, chemb50_trn, chemb50_val, chemb50_tst, chemb100_trn, chemb100_val, chemb100_tst

def qm9_data():
    qm_data = getattr(datasets, 'QM9')(root='./data/', cut_dist=args.cut_dist)[0:10]
    
    qm9_trn_idx, qm9_val_idx, qm9_tst_idx = random_split(qm_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    
    qm9_trn = qm_data[qm9_trn_idx]
    qm9_val = qm_data[qm9_val_idx]
    qm9_tst = qm_data[qm9_tst_idx]
    
    return qm9_trn, qm9_val, qm9_tst

def md17_data():
    asp_data = getattr(datasets, 'MD17')(root='./data/', cut_dist=args.cut_dist, dataset_arg='aspirin')[0:10]
    eth_data = getattr(datasets, 'MD17')(root='./data/', cut_dist=args.cut_dist, dataset_arg='ethanol')[0:10]
    mal_data = getattr(datasets, 'MD17')(root='./data/', cut_dist=args.cut_dist, dataset_arg='malonaldehyde')[0:10]
    nap_data = getattr(datasets, 'MD17')(root='./data/', cut_dist=args.cut_dist, dataset_arg='naphthalene')[0:10]
    sal_data = getattr(datasets, 'MD17')(root='./data/', cut_dist=args.cut_dist, dataset_arg='salicylic_acid')[0:10]
    tol_data = getattr(datasets, 'MD17')(root='./data/', cut_dist=args.cut_dist, dataset_arg='toluene')[0:10]
    ura_data = getattr(datasets, 'MD17')(root='./data/', cut_dist=args.cut_dist, dataset_arg='uracil')[0:10]
    
    asp_trn_idx, asp_val_idx, asp_tst_idx = random_split(asp_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    eth_trn_idx, eth_val_idx, eth_tst_idx = random_split(eth_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    mal_trn_idx, mal_val_idx, mal_tst_idx = random_split(mal_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    nap_trn_idx, nap_val_idx, nap_tst_idx = random_split(nap_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    sal_trn_idx, sal_val_idx, sal_tst_idx = random_split(sal_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    tol_trn_idx, tol_val_idx, tol_tst_idx = random_split(tol_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    ura_trn_idx, ura_val_idx, ura_tst_idx = random_split(ura_data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    
    asp_trn = asp_data[asp_trn_idx]
    eth_trn = eth_data[eth_trn_idx]
    mal_trn = mal_data[mal_trn_idx]
    nap_trn = nap_data[nap_trn_idx]
    sal_trn = sal_data[sal_trn_idx]
    tol_trn = tol_data[tol_trn_idx]
    ura_trn = ura_data[ura_trn_idx]
    
    asp_val = asp_data[asp_val_idx]
    eth_val = eth_data[eth_val_idx]
    mal_val = mal_data[mal_val_idx]
    nap_val = nap_data[nap_val_idx]
    sal_val = sal_data[sal_val_idx]
    tol_val = tol_data[tol_val_idx]
    ura_val = ura_data[ura_val_idx]
    
    asp_tst = asp_data[asp_tst_idx]
    eth_tst = eth_data[eth_tst_idx]
    mal_tst = mal_data[mal_tst_idx]
    nap_tst = nap_data[nap_tst_idx]
    sal_tst = sal_data[sal_tst_idx]
    tol_tst = tol_data[tol_tst_idx]
    ura_tst = ura_data[ura_tst_idx]
    
    return asp_trn, asp_val, asp_tst, eth_trn, eth_val, eth_tst, mal_trn, mal_val, mal_tst, nap_trn, nap_val, nap_tst, sal_trn, sal_val, sal_tst, tol_trn, tol_val, tol_tst, ura_trn, ura_val, ura_tst

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ###########complex dataset
    parser.add_argument('--molecule_name', type=str, default='MD17')
    parser.add_argument('--molecule_root', type=str, default='./data/')
    parser.add_argument('--molecule_type', type=str, default='uracil')
    
    ###########complex dataset
    parser.add_argument('--data_dir', type=str, default='./data/processed/')
    parser.add_argument('--dataset', type=str, default='pdbbind2016')
    parser.add_argument('--model_dir', type=str, default='./output/saved')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument('--cut_dist', type=float, default=5.)
    
    
    ###########training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--check', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    
    
    ###########optimizer and scheduler
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--patience_lr', type=int, default=20)
    
    
    ###########first model
    parser.add_argument('--glob_layer', type=int, default=6)
    parser.add_argument('--glob_head', type=int, default=8)
    parser.add_argument('--glob_in', type=int, default=100)
    parser.add_argument('--glob_hidden', type=int, default=128)
    parser.add_argument('--glob_out', type=int, default=64)
    parser.add_argument('--glob_rbf', type=int, default=32)
    
    ###########embedding model
    parser.add_argument('--emb_out', type=int, default=64)
    
    ###########second model
    parser.add_argument('--complex_layer', type=int, default=4)
    parser.add_argument('--dim_att', type=int, default=32)
    #parser.add_argument('--dim_qk', type=int, default=32)
    #parser.add_argument('--dim_v', type=int, default=32)
    parser.add_argument('--complex_out', type=int, default=32)
    #parser.add_argument('--dim_hidden', type=int, default=32)
    parser.add_argument('--complex_head', type=int, default=4)
    parser.add_argument('--complex_ff', type=int, default=32)
    parser.add_argument('--complex_readout_qk', type=int, default=32)
    parser.add_argument('--complex_readout_v', type=int, default=32)
    parser.add_argument('--complex_readout_heads', type=int, default=4)
    
    
    ###########dropout
    parser.add_argument('--drop_input', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--drop_mu', type=float, default=0.)


    args = parser.parse_args()
    
    print('Loading MD17 data...')
    asp_trn, asp_val, asp_tst, eth_trn, eth_val, eth_tst, mal_trn, mal_val, mal_tst, nap_trn, nap_val, nap_tst, sal_trn, sal_val, sal_tst, tol_trn, tol_val, tol_tst, ura_trn, ura_val, ura_tst = md17_data()
    
    stats_dic = {'asp':(stats(asp_trn)),
                 'eth':(stats(eth_trn)),
                 'mal':(stats(mal_trn)),
                 'nap':(stats(nap_trn)),
                 'sal':(stats(sal_trn)),
                 'tol':(stats(tol_trn)),
                 'ura':(stats(ura_trn))}
    
    print('Loading QM9 data...')
    qm9_trn, qm9_val, qm9_tst = qm9_data()
    qm9_prior_models = get_qm9_prior(qm9_trn)   
    
    print('Loading ChEMBL data...')
    chemb10_trn, chemb10_val, chemb10_tst, chemb50_trn, chemb50_val, chemb50_tst, chemb100_trn, chemb100_val, chemb100_tst = chemb_data()
    chemb_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    print('Loading PDBbind data...')
    pdb_trn, pdb_val, pdb_tst = pdb_data()
    
    
    asp_trn_num, eth_trn_num, mal_trn_num, nap_trn_num, sal_trn_num, tol_trn_num, ura_trn_num = len(asp_trn), len(eth_trn), len(mal_trn), len(nap_trn), len(sal_trn), len(tol_trn), len(ura_trn)
    qm9_trn_num = len(qm9_trn)
    chemb10_trn_num, chemb50_trn_num, chemb100_trn_num = len(chemb10_trn), len(chemb50_trn), len(chemb100_trn)
    pdb_trn_num = len(pdb_trn)
    
    asp_trn_wg, eth_trn_wg, mal_trn_wg, nap_trn_wg, sal_trn_wg, tol_trn_wg, ura_trn_wg = 1/asp_trn_num, 1/eth_trn_num, 1/mal_trn_num, 1/nap_trn_num, 1/sal_trn_num, 1/tol_trn_num, 1/ura_trn_num
    qm9_trn_wg = 12/qm9_trn_num
    chemb10_trn_wg, chemb50_trn_wg, chemb100_trn_wg = chemb10_trn_num, chemb50_trn_num, chemb100_trn_num
    pdb_trn_wg = 1/pdb_trn_num
    
    trn_dataset = torch.utils.data.ConcatDataset([pdb_trn, chemb10_trn, chemb50_trn, chemb100_trn, qm9_trn, asp_trn, eth_trn, mal_trn, nap_trn, sal_trn, tol_trn, ura_trn])
    
    val_dataset = torch.utils.data.ConcatDataset([chemb10_val, chemb50_val, chemb100_val, pdb_val, qm9_val, asp_val, eth_val, mal_val, nap_val, sal_val, tol_val, ura_val])
    
    tst_dataset = torch.utils.data.ConcatDataset([chemb10_tst, chemb50_tst, chemb100_tst, pdb_tst, qm9_tst, asp_tst, eth_tst, mal_tst, nap_tst, sal_tst, tol_tst, ura_tst])

#     trn_dataset = torch.utils.data.ConcatDataset([chemb100_trn])
    
#     val_dataset = torch.utils.data.ConcatDataset([chemb100_val])
    
#     tst_dataset = torch.utils.data.ConcatDataset([chemb100_tst])
    
    trn_loader = loader.DataLoader(trn_dataset, batch_size=1, shuffle=True)
    val_loader = loader.DataLoader(val_dataset, batch_size=1, shuffle=False)
    tst_loader = loader.DataLoader(tst_dataset, batch_size=1, shuffle=False)
    
    args.pbd_in = pdb_trn[0].x.shape[1]
    args.chemb_in = chemb10_trn[0].x.shape[1]
    args.chemb10_out = chemb10_trn[0].y.shape[0]
    args.chemb50_out = chemb50_trn[0].y.shape[0]
    args.chemb100_out = chemb100_trn[0].y.shape[0]
    
    args.complex_qk = args.complex_hidden = args.complex_v = args.dim_att
    args.complex_in = args.glob_out + args.emb_out + pdb_trn[0].edge_attr.shape[1]
    
    device = torch.device("cuda:" + str(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")

    
    #train_list, val_list, test_list = get_batched_data(args.batch_size)
    
    check_portion = np.linspace(1, args.check, args.check, endpoint=True)/args.check
    check_every = np.floor(check_portion * len(trn_dataset))
    
    
    print('Building Sparse Kernalized Transformer...')
    model = transformer.MTLModel(stats_dic, args.glob_layer, args.glob_head, args.glob_in, args.glob_hidden, args.glob_out, args.glob_rbf, args.cut_dist, args.pbd_in, args.chemb_in, args.emb_out, args.complex_layer, args.complex_in, args.complex_out, args.complex_hidden, args.complex_qk, args.complex_v, args.complex_ff, args.complex_head, args.complex_readout_qk, args.complex_readout_v, args.complex_readout_heads, args.chemb10_out, args.chemb50_out, args.chemb100_out, 'default', 'generalized_kernel', args.drop_input, args.dropout, args.drop_mu, prior=qm9_prior_models, sparse=True).to(device)

    model.skip_redraw_projections = True
    
    print('Building Optimizer and Scheduler...')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience_lr, min_lr=args.min_lr)
    
    print('Done building...')
    
    main()
