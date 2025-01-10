from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse

from models.HPNet_layers import GraphAttention
from models.HPNet_layers import TwoLayerMLP
from HPNet_utils import compute_angles_lengths_2D
from HPNet_utils import init_weights
from HPNet_utils import wrap_angle
from HPNet_utils import drop_edge_between_samples
from HPNet_utils import transform_point_to_local_coordinate
from HPNet_utils import transform_point_to_global_coordinate
from HPNet_utils import transform_traj_to_global_coordinate
from HPNet_utils import transform_traj_to_local_coordinate

class Backbone(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 num_attn_layers: int, 
                 num_modes: int,
                 num_heads: int,
                 dropout: float) -> None:
        super(Backbone, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.dropout = dropout

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)     #[K,D]

        self.a_emb_layer = TwoLayerMLP(input_dim=2, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        self.t2m_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.m2m_h_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_a_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_s_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.t2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.m2m_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=False, if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_propose = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*2)

        self.proposal_to_anchor = TwoLayerMLP(input_dim=self.num_future_steps*2, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.n2n_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_refine = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*2)

        self.apply(init_weights)

    def forward(self, data: Batch) -> torch.Tensor:
        # initialization
        a_velocity_length = data['agent']['velocity_length']                            #[(N1,...,Nb),H]
        a_velocity_theta = data['agent']['velocity_theta']                              #[(N1,...,Nb),H]

        a_input = torch.stack([a_velocity_length, a_velocity_theta], dim=-1)
        a_embs = self.a_emb_layer(input=a_input)    #[(N1,...,Nb),H,D]
        
        num_all_agent = a_velocity_length.size(0)                # N1+...+Nb
        m_embs = self.mode_tokens.weight.unsqueeze(0).repeat_interleave(self.num_historical_steps,0)            #[H,K,D]
        m_embs = m_embs.unsqueeze(1).repeat_interleave(num_all_agent,1).reshape(-1, self.hidden_dim)            #[H*(N1,...,Nb)*K,D]

        m_batch = data['agent']['batch'].unsqueeze(1).repeat_interleave(self.num_modes,1)                       # [(N1,...,Nb),K]
        m_position = data['agent']['position'][:,:self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes,2)  #[(N1,...,Nb),H,K,2]
        m_heading = data['agent']['heading'][:,:self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes,2)    #[(N1,...,Nb),H,K]
        m_valid_mask = data['agent']['visible_mask'][:,:self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes,2)  #[(N1,...,Nb),H,K]

        #ALL EDGE
        #t2m edge
        t2m_position_t = data['agent']['position'][:,:self.num_historical_steps].reshape(-1,2)      #[(N1,...,Nb)*H,2]
        t2m_position_m = m_position.reshape(-1,2)                                                   #[(N1,...,Nb)*H*K,2]
        t2m_heading_t = data['agent']['heading'].reshape(-1)                                        #[(N1,...,Nb)]
        t2m_heading_m = m_heading.reshape(-1)                                                       #[(N1,...,Nb)*H*K]
        t2m_valid_mask_t = data['agent']['visible_mask'][:,:self.num_historical_steps]              #[(N1,...,Nb),H]
        t2m_valid_mask_m = m_valid_mask.reshape(num_all_agent,-1)                                   #[(N1,...,Nb),H*K]
        t2m_valid_mask = t2m_valid_mask_t.unsqueeze(2) & t2m_valid_mask_m.unsqueeze(1)              #[(N1,...,Nb),H,H*K]
        t2m_edge_index = dense_to_sparse(t2m_valid_mask)[0]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) >= t2m_edge_index[0]]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) - t2m_edge_index[0] <= self.duration]
        # start_point = t2m_position_t[t2m_edge_index[0]]
        # end_point = t2m_position_m[t2m_edge_index[1]]
        # end_point_heading = t2m_heading_m[t2m_edge_index[1]]
        t2m_edge_vector = transform_point_to_local_coordinate(t2m_position_t[t2m_edge_index[0]], t2m_position_m[t2m_edge_index[1]], t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_length, t2m_edge_attr_theta = compute_angles_lengths_2D(t2m_edge_vector)
        t2m_edge_attr_heading = wrap_angle(t2m_heading_t[t2m_edge_index[0]] - t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_interval = t2m_edge_index[0] - torch.floor(t2m_edge_index[1]/self.num_modes)
        t2m_edge_attr_input = torch.stack([t2m_edge_attr_length, t2m_edge_attr_theta, t2m_edge_attr_heading, t2m_edge_attr_interval], dim=-1)
        t2m_edge_attr_embs = self.t2m_emb_layer(input=t2m_edge_attr_input)


        #mode edge
        #m2m_a_edge
        m2m_a_position = m_position.permute(1,2,0,3).reshape(-1, 2)    #[H*K*(N1,...,Nb),2]
        m2m_a_heading = m_heading.permute(1,2,0).reshape(-1)           #[H*K*(N1,...,Nb)]
        m2m_a_batch = data['agent']['batch']                           #[(N1,...,Nb)]
        m2m_a_valid_mask = m_valid_mask.permute(1,2,0).reshape(self.num_historical_steps * self.num_modes, -1)  #[H*K,(N1,...,Nb)] 表示每个时间步和模态中，各个参与体的有效性掩码
        m2m_a_valid_mask = m2m_a_valid_mask.unsqueeze(2) & m2m_a_valid_mask.unsqueeze(1)                        #[H*K,(N1,...,Nb),(N1,...,Nb)] 表示每个时间步和模态下，各个参与体两两之间的有效性
        m2m_a_valid_mask = drop_edge_between_samples(m2m_a_valid_mask, m2m_a_batch)                             #移除不同样本之间的边，确保边仅存在于同一个样本的节点之间。
        m2m_a_edge_index = dense_to_sparse(m2m_a_valid_mask)[0] # 将稠密矩阵稀疏化，(self.num_historical_steps * self.num_modes, num_all_agent, num_all_agent)->(2, num_edges) edge_index[0] 和 edge_index[1] 分别表示边的起点和终点。
        m2m_a_edge_index = m2m_a_edge_index[:, m2m_a_edge_index[1] != m2m_a_edge_index[0]] # 移除自连接边
        m2m_a_edge_index = m2m_a_edge_index[:, torch.norm(m2m_a_position[m2m_a_edge_index[1]] - m2m_a_position[m2m_a_edge_index[0]],p=2,dim=-1) < self.a2a_radius] # 仅保留距离小于 a2a_radius(80) 的边。
        m2m_a_edge_vector = transform_point_to_local_coordinate(m2m_a_position[m2m_a_edge_index[0]], m2m_a_position[m2m_a_edge_index[1]], m2m_a_heading[m2m_a_edge_index[1]]) # 将终点指向起点的向量旋转到终点节点的朝向（heading）方向上，从而将向量转换到终点节点的局部坐标系中。
        m2m_a_edge_attr_length, m2m_a_edge_attr_theta = compute_angles_lengths_2D(m2m_a_edge_vector) # 计算边的长度和角度
        m2m_a_edge_attr_heading = wrap_angle(m2m_a_heading[m2m_a_edge_index[0]] - m2m_a_heading[m2m_a_edge_index[1]]) # 计算边的角度差, wrap_angle 保证角度在 -pi 到 pi 之间。
        m2m_a_edge_attr_input = torch.stack([m2m_a_edge_attr_length, m2m_a_edge_attr_theta, m2m_a_edge_attr_heading], dim=-1) # 将边的长度、角度和角度差拼接在一起。
        m2m_a_edge_attr_embs = self.m2m_a_emb_layer(input=m2m_a_edge_attr_input)

        #m2m_h                        
        m2m_h_position = m_position.permute(2,0,1,3).reshape(-1, 2)    #[K*(N1,...,Nb)*H,2]
        m2m_h_heading = m_heading.permute(2,0,1).reshape(-1)           #[K*(N1,...,Nb)*H]
        m2m_h_valid_mask = m_valid_mask.permute(2,0,1).reshape(-1, self.num_historical_steps)   #[K*(N1,...,Nb),H]
        m2m_h_valid_mask = m2m_h_valid_mask.unsqueeze(2) & m2m_h_valid_mask.unsqueeze(1)        #[K*(N1,...,Nb),H,H]     
        m2m_h_edge_index = dense_to_sparse(m2m_h_valid_mask)[0]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] > m2m_h_edge_index[0]]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] - m2m_h_edge_index[0] <= self.duration] # duration=10
        m2m_h_edge_vector = transform_point_to_local_coordinate(m2m_h_position[m2m_h_edge_index[0]], m2m_h_position[m2m_h_edge_index[1]], m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_length, m2m_h_edge_attr_theta = compute_angles_lengths_2D(m2m_h_edge_vector)
        m2m_h_edge_attr_heading = wrap_angle(m2m_h_heading[m2m_h_edge_index[0]] - m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_interval = m2m_h_edge_index[0] - m2m_h_edge_index[1]
        m2m_h_edge_attr_input = torch.stack([m2m_h_edge_attr_length, m2m_h_edge_attr_theta, m2m_h_edge_attr_heading, m2m_h_edge_attr_interval], dim=-1)
        m2m_h_edge_attr_embs = self.m2m_h_emb_layer(input=m2m_h_edge_attr_input)

        #m2m_s edge
        m2m_s_valid_mask = m_valid_mask.transpose(0,1).reshape(-1, self.num_modes)              #[H*(N1,...,Nb),K]
        m2m_s_valid_mask = m2m_s_valid_mask.unsqueeze(2) & m2m_s_valid_mask.unsqueeze(1)        #[H*(N1,...,Nb),K,K]
        m2m_s_edge_index = dense_to_sparse(m2m_s_valid_mask)[0]
        m2m_s_edge_index = m2m_s_edge_index[:, m2m_s_edge_index[0] != m2m_s_edge_index[1]]

        #ALL ATTENTION
        #t2m attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  #[(N1,...,Nb)*H,D]
        m_embs_t = self.t2m_attn_layer(x = [t_embs, m_embs], edge_index = t2m_edge_index, edge_attr = t2m_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]
        
        m_embs = m_embs_t
        m_embs = m_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1,self.hidden_dim)       #[H*(N1,...,Nb)*K,D]
        #moda attention  
        for i in range(self.num_attn_layers):
            #m2m_a关注 智能体（agents）之间的关系，即不同参与体如何在同一时间步和同一模态下相互影响。
            m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1,2).reshape(-1, self.hidden_dim)  #[H*K*(N1,...,Nb),D] H是指当前的历史步数，K是指当前的模态数，N1,...,Nb是指当前的agent数
            m_embs = self.m2m_a_attn_layers[i](x = m_embs, edge_index = m2m_a_edge_index, edge_attr = m2m_a_edge_attr_embs)
            #m2m_h关注 历史时间步（historical steps）之间的关系，即不同时间步如何在同一智能体和同一模态下相互影响。
            m_embs = m_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1,2,0,3).reshape(-1, self.hidden_dim)  #[K*(N1,...,Nb)*H,D]
            m_embs = self.m2m_h_attn_layers[i](x = m_embs, edge_index = m2m_h_edge_index, edge_attr = m2m_h_edge_attr_embs)
            #m2m_s关注 模态（modes）之间的关系，即不同模态如何在同一时间步和同一智能体下相互影响。
            m_embs = m_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0,2).reshape(-1, self.hidden_dim)  #[H*(N1,...,Nb)*K,D]
            m_embs = self.m2m_s_attn_layers[i](x = m_embs, edge_index = m2m_s_edge_index)
        m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1, self.hidden_dim)      #[(N1,...,Nb)*H*K,D]

        #generate traj
        traj_propose = self.traj_propose(m_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2)         #[(N1,...,Nb),H,K,F,2]
        traj_propose = transform_traj_to_global_coordinate(traj_propose, m_position, m_heading)        #[(N1,...,Nb),H,K,F,2]

        return traj_propose         #[(N1,...,Nb),H,K,F,2]