from typing import Optional
import pdb

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import MLP, GINConv, global_add_pool


class DenseGATConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GATConv`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        use_diff: bool = True, 
    ):
        # TODO Add support for edge features.
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.use_diff = use_diff

        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')
        if self.use_diff:
            self.lin_diff = Linear(1, heads, bias=False,
                              weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, 1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, 1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.use_diff:
            self.lin_diff.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Tensor, adj: Tensor, diff: Tensor = None, mask: Optional[Tensor] = None,
            add_loop: bool = True):
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
            diff: the difference matrix
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]
        if self.use_diff:
            diff = diff.unsqueeze(0) if diff.dim() == 2 else diff  # [B, N, N]
            diff = diff.unsqueeze(3) # [B, N, N, 1]

        H, C = self.heads, self.out_channels
        B, N, _ = x.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1.0

        if self.use_diff:
            diff = self.lin_diff(diff)
            x = torch.ones(x.shape).to(x.device)

        x = self.lin(x)
        x = x.reshape(B, N, H, C)  # [B, N, H, C]

        alpha_src = torch.sum(x * self.att_src, dim=-1)  # [B, N, H]
        alpha_dst = torch.sum(x * self.att_dst, dim=-1)  # [B, N, H]

        alpha = alpha_src.unsqueeze(1) + alpha_dst.unsqueeze(2)  # [B, N, N, H]


        # Weighted and masked softmax:
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))

        if self.use_diff:
            alpha = alpha + diff

        alpha = alpha.softmax(dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha.movedim(3, 1), x.movedim(2, 1))
        out = out.movedim(1, 2)  # [B,N,H,C]

        if self.concat:
            out = out.reshape(B, N, H * C)
        else:
            out = out.mean(dim=2)

        ''' # bias will also be added to dummy nodes so disable here
        if self.bias is not None:
            out = out + self.bias
        '''

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class DenseGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(DenseGATConv(in_channels, hidden_channels, use_diff=use_diff))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, adj, diff):
        for conv in self.convs:
            x = conv(x, adj, diff).relu()
        x = x.sum(1)
        return self.mlp(x)



class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


class DiffGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs_diff = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            mlp_diff = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs_diff.append(GINConv(nn=mlp_diff, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, edge_index_diff, batch):
        for conv, conv_diff in zip(self.convs, self.convs_diff):
            x1 = conv(x, edge_index).relu()
            x2 = conv_diff(x, edge_index_diff).relu()
            x = x1 + x2
        x = global_add_pool(x, batch)
        return self.mlp(x)


class DeepDiff(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        self.adj_mlps = torch.nn.ModuleList()
        self.diff_mlps = torch.nn.ModuleList()
        in_channels = 1
        for _ in range(num_layers):
            adj_mlp = MLP([in_channels, hidden_channels, hidden_channels], norm=None)
            diff_mlp = MLP([in_channels, hidden_channels, hidden_channels], norm=None)
            self.adj_mlps.append(adj_mlp)
            self.diff_mlps.append(diff_mlp)
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, adj, diff):
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]
        adj = adj.unsqueeze(3) # [B, N, N, 1]
        diff = diff.unsqueeze(0) if diff.dim() == 2 else diff  # [B, N, N]
        diff = diff.unsqueeze(3) # [B, N, N, 1]
        for mlp1, mlp2 in zip(self.adj_mlps, self.diff_mlps):
            adj = mlp1(adj)
            diff = mlp2(diff)
            if False:
                A = adj + diff
            else:
                A = torch.matmul(adj.movedim(3, 1), diff.movedim(3, 1)).movedim(1, 3) / diff.shape[1]
                adj = adj + A
                diff = diff + A
        A = A.sum(dim=[1, 2])
        A = self.mlp(A)
        return A



class DeepSet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        self.mlps = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.mlps.append(mlp)
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, batch):
        for mlp in self.mlps:
            x = mlp(x)
        x = global_add_pool(x, batch)
        return self.mlp(x)


class DeepCount(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        self.mlps = torch.nn.ModuleList()
        in_channels = 1
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.mlps.append(mlp)
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, batch):
        for mlp in self.mlps:
            x = mlp(x)
        x = global_add_pool(x, batch)
        return self.mlp(x)






