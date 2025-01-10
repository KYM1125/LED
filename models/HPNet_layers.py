from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from HPNet_utils import init_weights


class GraphAttention(MessagePassing):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float,
                 has_edge_attr: bool,
                 if_self_attention: bool,
                 **kwargs) -> None:
        super(GraphAttention, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        if has_edge_attr:
            self.edge_k = nn.Linear(hidden_dim, hidden_dim)
            self.edge_v = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.attn_drop = nn.Dropout(dropout)
        if if_self_attention:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
        else:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
            self.mha_prenorm_dst = nn.LayerNorm(hidden_dim)
        if has_edge_attr:
            self.mha_prenorm_edge = nn.LayerNorm(hidden_dim)
        self.ffn_prenorm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], # [H*K*N,D]
                edge_index: torch.Tensor, # (2, num_edges)
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor: # (num_edges, 128)
        if self.if_self_attention:
            x_src = x_dst = self.mha_prenorm_src(x)
        else:
            x_src, x_dst = x
            x_src = self.mha_prenorm_src(x_src)
            x_dst = self.mha_prenorm_dst(x_dst)
        if self.has_edge_attr:
            edge_attr = self.mha_prenorm_edge(edge_attr)
        x_dst = x_dst + self._mha_layer(x_src, x_dst, edge_index, edge_attr) # [H*K*N,D]
        x_dst = x_dst + self._ffn_layer(self.ffn_prenorm(x_dst))
        return x_dst

    def message(self,
                x_dst_i: torch.Tensor,
                x_src_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: Optional[torch.Tensor]) -> torch.Tensor:
        query_i = self.q(x_dst_i).view(-1, self.num_heads, self.head_dim) # [Nodes,8,16]
        key_j = self.k(x_src_j).view(-1, self.num_heads, self.head_dim)
        value_j = self.v(x_src_j).view(-1, self.num_heads, self.head_dim)
        if self.has_edge_attr:
            key_j = key_j + self.edge_k(edge_attr).view(-1, self.num_heads, self.head_dim) # k由起点的节点特征和边特征共同决定 # [Nodes,8,16]
            value_j = value_j + self.edge_v(edge_attr).view(-1, self.num_heads, self.head_dim) # v也是由起点的节点特征和边特征共同决定
        scale = self.head_dim ** 0.5
        weight = (query_i * key_j).sum(dim=-1) / scale
        weight = softmax(weight, index, ptr) # [Nodes,8]
        weight = self.attn_drop(weight)
        return (value_j * weight.unsqueeze(-1)).view(-1, self.num_heads*self.head_dim) # [Nodes,128]

    def _mha_layer(self,
                   x_src: torch.Tensor,
                   x_dst: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x_dst=x_dst, x_src=x_src) # 会调用类中定义的 message、aggregate 和 update 函数

    def _ffn_layer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class TwoLayerMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int) -> None:
        super(TwoLayerMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.mlp(input)