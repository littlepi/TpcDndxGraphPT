import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, PointTransformerConv, fps, TransformerConv
from torch_geometric.nn import knn_graph, knn_interpolate, knn
from torch_geometric.utils import scatter, to_undirected


class DotProductTransformerBlock(TransformerConv):
    """Dot-product transformer block with multi-head attention.
    """
    def __init__(self, in_channels, out_channels, pos_dims=3, heads=1, dropout=0.0, **kwargs):
        super(DotProductTransformerBlock, self).__init__(
            in_channels, out_channels, heads=heads, dropout=dropout, edge_dim=heads, concat=False, **kwargs)

        self.rel_pos_mlp = MLP([pos_dims, 64, 64, heads], norm=None, plain_last=False, act='leaky_relu')

    def forward(self, x, pos, edge_index, **kwargs):
        row, col = edge_index
        rel_pos = pos[col] - pos[row]  # [E, pos_dim]
        rel_pos_bias = self.rel_pos_mlp(rel_pos)  # [E, heads]
        
        out = super(DotProductTransformerBlock, self).forward(x, edge_index, edge_attr=rel_pos_bias, **kwargs)
        return out

class SubtractionTransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, in_channels)
        self.lin_out = nn.Linear(out_channels, out_channels)
        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False, act='leaky_relu')
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)
        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x
    
class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16, reduce='max', enable_mlp=True):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.reduce = reduce
        self.enable_mlp = enable_mlp
        if enable_mlp:
            self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, 
                            pos[id_clusters], 
                            k=self.k, batch_x=batch, batch_y=sub_batch)

        # transformation of features through a simple MLP

        if self.enable_mlp:
            x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce=self.reduce)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch

class TransitionUp(torch.nn.Module):
    """Reduce features dimensionality and interpolate back to higher
    resolution and cardinality.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels], plain_last=False)
        self.mlp = MLP([out_channels, out_channels], plain_last=False)

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x

class GraphPointTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model=[32, 128, 512, 2048, 8192], k=16, down_ratio=0.25, 
                 undirected=False, reduce='max', self_attention=False, num_heads=4, debug=False):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        if not self_attention:
            self.transformer_input = SubtractionTransformerBlock(
                in_channels=dim_model[0],
                out_channels=dim_model[0]
            )
        else:
            self.transformer_input = DotProductTransformerBlock(
                in_channels=dim_model[0], out_channels=dim_model[0], pos_dims=3, heads=num_heads, dropout=0.0)

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(0, len(dim_model) - 1):

            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k, ratio=down_ratio, reduce=reduce))

            if not self_attention:
                self.transformers_down.append(
                    SubtractionTransformerBlock(in_channels=dim_model[i + 1],
                                     out_channels=dim_model[i + 1]))
            else:
                self.transformers_down.append(
                    DotProductTransformerBlock(in_channels=dim_model[i + 1], 
                                              out_channels=dim_model[i + 1], pos_dims=3, heads=num_heads, dropout=0.0))

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1],
                             out_channels=dim_model[i]))

            if not self_attention:
                self.transformers_up.append(
                    SubtractionTransformerBlock(in_channels=dim_model[i],
                                     out_channels=dim_model[i]))
            else:
                self.transformers_up.append(
                    DotProductTransformerBlock(in_channels=dim_model[i],
                                              out_channels=dim_model[i], pos_dims=3, heads=num_heads, dropout=0.0))

        # summit layers
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], norm=None, plain_last=False)

        if not self_attention:
            self.transformer_summit = SubtractionTransformerBlock(
                in_channels=dim_model[-1],
                out_channels=dim_model[-1]
            )
        else:
            self.transformer_summit = DotProductTransformerBlock(
                in_channels=dim_model[-1], out_channels=dim_model[-1], pos_dims=3, heads=num_heads, dropout=0.0)

        # class score computation
        self.mlp_output = MLP([dim_model[0], 64, out_channels], norm=None)

        self.undirected = undirected
        self.debug = debug

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        if self.debug:
            print(f'input x: {x.shape}')
       
        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        if self.undirected: edge_index = to_undirected(edge_index)
        x = self.transformer_input(x, pos, edge_index)
        if self.debug:
            print(f'first block x: {x.shape}')

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            if self.undirected: edge_index = to_undirected(edge_index)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

            if self.debug:
                print(f'backbone down {i} x: {x.shape}')

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        if self.undirected: edge_index = to_undirected(edge_index)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](x=out_x[-i - 2], x_sub=x,
                                           pos=out_pos[-i - 2],
                                           pos_sub=out_pos[-i - 1],
                                           batch_sub=out_batch[-i - 1],
                                           batch=out_batch[-i - 2])

            edge_index = knn_graph(out_pos[-i - 2], k=self.k,
                                   batch=out_batch[-i - 2])
            if self.undirected: edge_index = to_undirected(edge_index)
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

            if self.debug:
                print(f'backbone up {i} x: {x.shape}')

        # Class score
        x = self.mlp_output(x)
        x = F.softmax(x, dim=-1)

        if self.debug:
            print(f'output x: {x.shape}')

        return x
