from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from visnet.models.utils import (
    CosineCutoff,
    Distance, 
    EdgeEmbedding,
    NeighborEmbedding, 
    Sphere, 
    VecLayerNorm,
    act_class_mapping, 
    rbf_class_mapping
)


class ViSNetBlock(nn.Module):

    def __init__(
        self,
        lmax=2,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=9,
        hidden_channels=256,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        max_z=100,
        cutoff=5.0,
        max_num_neighbors=32,
        vertex_type="Edge",
    ):
        super(ViSNetBlock, self).__init__()
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.max_z = max_z
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
    
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors, loop=True)
        self.sphere = Sphere(l=lmax)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff, max_z).jittable()
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels).jittable()

        self.vis_mp_layers = nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=num_heads, 
            hidden_channels=hidden_channels, 
            activation=activation, 
            attn_activation=attn_activation, 
            cutoff=cutoff, 
            vecnorm_type=vecnorm_type, 
            trainable_vecnorm=trainable_vecnorm
        )
        vis_mp_class = VIS_MP_MAP.get(vertex_type, ViS_MP)
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs).jittable()
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(vis_mp_class(last_layer=True, **vis_mp_kwargs).jittable())

        self.out_norm = nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()
        
    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        
        z, pos, batch = data.z, data.pos, data.batch
        
        # Embedding Layers
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1), device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)
        
        # ViS-MP Layers
        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec)
        x = x + dx
        vec = vec + dvec
        
        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec





class aggregation_custom(torch.nn.Module):
    def __init__(self):
        super(aggregation_custom, self).__init__()
        
        self.fc_all = torch.nn.Linear(int(128/2),128,bias=False)
        
        
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, index, dim, dim_size):
        msg = x
        
        ##print(msg.shape)#-1,16,144
        #time.sleep(999)
        
        
        var_num_patches = 2
        var_effective_patch_dim = int(128/2)
        
        e = msg#torch.permute(msg, (0,2,3,1))
        e = torch.reshape(e, (-1, 128))
        
        e1_0 = e.reshape(-1, var_num_patches, var_effective_patch_dim)#(-1,4,32)
        e2_0 = e.reshape(-1, var_num_patches, var_effective_patch_dim)#(-1,4,32)
        
        ###print(e1_0.shape)
        
        e1_1 = torch.clamp(self.fc_all(e1_0),0.,1.)#-1,4,4*32
        e1_lst = [e1_1[:, :, i * var_effective_patch_dim : (i + 1) * var_effective_patch_dim] for i in range(var_num_patches)]
        out_lst = torch.zeros(e1_0.shape[0],1).to("cuda:0")
        for _i in range(len(e1_lst)):
            out_ = e2_0 * e1_lst[_i]#(-1,4,32)
            out_ = out_.sum(1) #-1,32
            out_lst = torch.cat((out_lst,out_),-1)
        out_lst = out_lst[:,1:]#+e2.to(self.device)#num_edges,128
        out = torch.reshape(out_lst,(-1,128))
        ###print(out.shape)
        
        
        
        
        
        ###print(out.shape)
        out = scatter(out, index, dim=0)
        ###print(out.shape)
        #scatter(t_dot, index, dim=self.node)
        return out







class ViS_MP(MessagePassing):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        activation,
        attn_activation,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
    ):
        super(ViS_MP, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        
        self.act = act_class_mapping[activation]()
        self.attn_activation = act_class_mapping[attn_activation]()
        
        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)
        
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)
        
        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.aggregation_method = aggregation_custom()
        
        self.reset_parameters()
        
    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)
        
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

        
    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):

        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        
        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)
    
        return v_j, vec_j
    
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        #print("274")
        #print(x.shape)#10627, 128
        #print(vec.shape)#10627, 8, 128
        #print("277")
        x = self.aggregation_method(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        #print(x.shape)#617, 128
        #print(vec.shape)#617, 8, 128
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
    
class ViS_MP_Vertex_Edge(ViS_MP):
    
    def __init__(
        self, 
        num_heads, 
        hidden_channels, 
        activation, 
        attn_activation, 
        cutoff, 
        vecnorm_type, 
        trainable_vecnorm, 
        last_layer=False
    ):
        super().__init__(num_heads, hidden_channels, activation, attn_activation, cutoff, vecnorm_type, trainable_vecnorm, last_layer)
        
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels * 2)
            self.t_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.t_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):

        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        
        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)
        
        f1, f2 = torch.split(self.act(self.f_proj(f_ij)), self.hidden_channels, dim=-1)

        return f1 * w_dot + f2 * t_dot

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None
    
class ViS_MP_Vertex_Node(ViS_MP):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        activation,
        attn_activation,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
    ):
        super().__init__(num_heads, hidden_channels, activation, attn_activation, cutoff, vecnorm_type, trainable_vecnorm, last_layer)

        self.t_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.t_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 4)
        
    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out, t_dot = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3, o4 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + t_dot * o3 + o4
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def message(self, q_i, k_j, v_j, vec_i, vec_j, dk, dv, r_ij, d_ij):

        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        
        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)
        
        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)
    
        return v_j, vec_j, t_dot

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec, t_dot = features
        #print("442")
        #print(x.shape)
        #print(vec.shape)
        #print(t_dot.shape)
        #print("446")
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        t_dot = scatter(t_dot, index, dim=self.node_dim, dim_size=dim_size)
        #print(x.shape)
        #print(vec.shape)
        #print(t_dot.shape)
        return x, vec, t_dot
    
VIS_MP_MAP = {'Node': ViS_MP_Vertex_Node, 'Edge': ViS_MP_Vertex_Edge, 'None': ViS_MP}