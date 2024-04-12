import torch
from torch import nn
from torch.nn import Linear, Embedding 
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from math import sqrt
import time

from ...utils import xyz_to_dat
from .features import dist_emb, angle_emb

try:
    import sympy as sym
except ImportError:
    sym = None

def swish(x):
    return x * torch.sigmoid(x)

class emb(torch.nn.Module):
    def __init__(self, device, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        return dist_emb, angle_emb


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, device, num_radial, hidden_channels, act=swish):
        super(init, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf,_ = emb
        x = self.emb(x)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, device, hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial, 
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        x_kj = scatter(x_kj.to(self.device), idx_ji.to(self.device), dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1
        
        #e1 is x1
        return e1, e2 





class update_v(torch.nn.Module):
    def __init__(self, device, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init, var_effective_patch_dim, var_num_patches, var_total_feature_dim, slot_0, slot_1):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init

        
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)
        
        self.var_effective_patch_dim = var_effective_patch_dim
        self.var_num_patches =  var_num_patches
        self.var_total_feature_dim = var_total_feature_dim
        
        self.fc_all = nn.Linear(self.var_effective_patch_dim,self.var_effective_patch_dim*self.var_num_patches,bias=False)
        
        #self.fc_pre_0 = nn.Linear(hidden_channels, self.var_total_feature_dim)
        #self.fc_pre_1 = nn.Linear(hidden_channels, self.var_total_feature_dim)
        
        self.slot_0 = slot_0
        self.slot_1 = slot_1
        
        
        self.lin_up = nn.Linear(self.var_total_feature_dim, out_emb_channels, bias=True)
        
        self.device = device
        
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i, batch_):
        e1, e2 = e
        #print(e1.shape)#1999,128
        #e1 = self.fc_pre_0(e1)
        #e2 = self.fc_pre_1(e2)
        
        choice_agg = "mlp"
        
        if choice_agg == "mlp":
            if self.slot_0 == "e1":
                e1_0 = e1.reshape(-1, self.var_num_patches, self.var_effective_patch_dim)#(-1,4,32)
            elif self.slot_0 == "e2":
                e1_0 = e2.reshape(-1, self.var_num_patches, self.var_effective_patch_dim)#(-1,4,32)
            
            if self.slot_1 == "e1":
                e2_0 = e1.reshape(-1, self.var_num_patches, self.var_effective_patch_dim)#(-1,4,32)
            elif self.slot_1 == "e2":
                e2_0 = e2.reshape(-1, self.var_num_patches, self.var_effective_patch_dim)#(-1,4,32)
            
            e1_1 = torch.clamp(self.fc_all(e1_0),0.,1.)#-1,4,4*32
            
            #e1_lst = [e1_1[:,:,(0*self.var_effective_patch_dim):(1*self.var_effective_patch_dim)],
            #          e1_1[:,:,(1*self.var_effective_patch_dim):(2*self.var_effective_patch_dim)],
            #          e1_1[:,:,(2*self.var_effective_patch_dim):(3*self.var_effective_patch_dim)],
            #          e1_1[:,:,(3*self.var_effective_patch_dim):(4*self.var_effective_patch_dim)]
            #          ]
            
            e1_lst = [e1_1[:, :, i * self.var_effective_patch_dim : (i + 1) * self.var_effective_patch_dim] for i in range(self.var_num_patches)]
            
            out_lst = torch.zeros(e1_0.shape[0],1).to(self.device)
            for _i in range(len(e1_lst)):
                out_ = e2_0 * e1_lst[_i]#(-1,4,32)
                out_ = out_.sum(1) #-1,32
                #print(out_.shape)
                out_lst = torch.cat((out_lst,out_),-1)
                
            out_lst = out_lst[:,1:]
            v = scatter(out_lst.to(self.device), i.to(self.device), dim=0)
        elif choice_agg == "sum":
            v = scatter(e2.to(self.device), i.to(self.device), dim=0)
        
        v = self.lin_up(v)
        
        
        for lin in self.lins:
            v = self.act(lin(v))
        
        
        v = self.lin(v)
        return (v, 0.)



class update_u(torch.nn.Module):
    def __init__(self, device):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


class Custom_Model(torch.nn.Module):
    r"""
        The re-implementation for DimeNet++ from the `"Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" <https://arxiv.org/abs/2011.14115>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
            basis_emb_size (int, optional): Embedding size used in the basis transformation. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
            act: (function, optional): The activation funtion. (default: :obj:`swish`) 
            output_init: (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)       
    """
    def __init__(
        self, device = "cuda:0", energy_and_force=False, cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=1, int_emb_size=64, basis_emb_size=8, out_emb_channels=256, 
        num_spherical=7, num_radial=6, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, 
        act=swish, output_init='GlorotOrthogonal', var_effective_patch_dim = 64, var_num_patches = 2, var_total_feature_dim=128, slot_0 = "e1", slot_1 = "e2"):
        super(Custom_Model, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force
        self.device = device

        self.init_e = init(self.device, num_radial, hidden_channels, act)
        self.init_v = update_v(self.device, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init, var_effective_patch_dim, var_num_patches, var_total_feature_dim, slot_0, slot_1)
        self.init_u = update_u(self.device)
        self.emb = emb(self.device, num_spherical, num_radial, self.cutoff, envelope_exponent)
        
        self.update_vs = torch.nn.ModuleList([
            update_v(self.device, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init, var_effective_patch_dim, var_num_patches, var_total_feature_dim, slot_0, slot_1) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(
                self.device, hidden_channels, int_emb_size, basis_emb_size,
                num_spherical, num_radial,
                num_before_skip, num_after_skip,
                act,
            )
            for _ in range(num_layers)
        ])

        self.update_us = torch.nn.ModuleList([update_u(self.device) for _ in range(num_layers)])
        
        
        
        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()


    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes=z.size(0)
        dist, angle, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=False)

        emb = self.emb(dist, angle, idx_kj)

        #Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v, euclidean_distance_matrix = self.init_v(e, i, batch_data.batch)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v, euclidean_distance_matrix_sub = update_v(e, i, batch_data.batch)
            u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)
            
            #euclidean_distance_matrix = euclidean_distance_matrix + euclidean_distance_matrix_sub
        
        
        u = u
        #euclidean_distance_matrix = euclidean_distance_matrix / (4+1)
        aux_loss_all = 0.#-1. * euclidean_distance_matrix#0.
        node_sim_loss = 0.#euclidean_distance_matrix
        opt_aux = True
        return (u,aux_loss_all,opt_aux)#(u,aux_loss_all,opt_aux,node_sim_loss)
