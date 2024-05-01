# Based on the code from: https://github.com/klicperajo/dimenet,
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py

import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import pi as PI
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def xyz_to_dat(pos, edge_index, num_nodes, use_torsion = False):
    """
    Compute the diatance, angle, and torsion from geometric information.

    Args:
        pos: Geometric information for every node in the graph.
        edgee_index: Edge index of the graph.
        number_nodes: Number of nodes in the graph.
        use_torsion: If set to :obj:`True`, will return distance, angle and torsion, otherwise only return distance and angle (also retrun some useful index). (default: :obj:`False`)
    """
    j, i = edge_index  # j->i
    #print(pos.shape)#[4643, 3]
    #print(j)#[72822]
    #print(i)#[72822]#edges
    
    # Calculate distances. # number of edges
    pre_1 = (pos[i] - pos[j])
    #print("pre_1.shape: "+str(pre_1.shape))#([72822, 3]
    pre_2 = pre_1.pow(2)
    #print("pre_2.shape: "+str(pre_2.shape))#([72822, 3]
    pre_3 = pre_2.sum(dim=-1)
    #print("pre_3.shape: "+str(pre_3.shape))#([72822]
    pre_4 = pre_3.sqrt()
    #print("pre_4.shape: "+str(pre_4.shape))#([72822]
    dist = pre_4#73730,1
    #print(dist.shape)
    
    value = torch.arange(j.size(0), device=j.device)
    #print("value:"+str(value.shape))#71776
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    #print("adj_t:"+str(adj_t))#size=(4634, 4634), nnz=72680, density=0.34%)
    adj_t_row = adj_t[j]
    #print("adj_t_row:"+str(adj_t_row))# size=(72680, 4634), nnz=1192610
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
    #print("num_triplets:"+str(num_triplets.shape))#([72680]
    #print("num_triplets:"+str(num_triplets))#tensor([15, 15, 14,  ..., 17, 15, 15])
    
    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    #print("idx_i:"+str(idx_i.shape))
    idx_j = j.repeat_interleave(num_triplets)
    #print("idx_j:"+str(idx_j.shape))#torch.Size([1188816])
    idx_k = adj_t_row.storage.col()
    #print("idx_k:"+str(idx_k.shape))#torch.Size([1188816])
    mask = idx_i != idx_k
    #print("mask:"+str(mask.shape))#torch.Size([1188816])
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    ##print("mask:"+str(mask))
    
    
    
    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]
    
    #print("idx_kj:"+str((adj_t_row.storage.value()[mask]).shape))#1054776 #tensor([   98,   100,   102,  ..., 74325, 74321, 74322])
    #print("idx_ji:"+str((adj_t_row.storage.row()[mask]).shape))#1054776 #tensor([    0,     0,     0,  ..., 74421, 74421, 74421])
    
    
    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    
    #time.sleep(999)
    if use_torsion:
        # Prepare torsion idxes.
        idx_batch = torch.arange(len(idx_i),device=device)
        idx_k_n = adj_t[idx_j].storage.col()
        repeat = num_triplets
        num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t.cuda())
        mask = idx_i_t != idx_k_n   
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]

        # Calculate torsions.
        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(pos_ji, pos_j0)
        plane2 = torch.cross(pos_ji, pos_jk)
        a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji 
        torsion1 = torch.atan2(b, a) # -pi to pi
        torsion1[torsion1<=0]+=2*PI # 0 to 2pi
        torsion = scatter(torsion1,idx_batch_t,reduce='min')

        return dist, angle, torsion, i, j, idx_kj, idx_ji
    
    else:
        return dist, angle, i, j, idx_kj, idx_ji



