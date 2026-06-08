import itertools
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import HypergraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GlobalAttention

class PreNorm(nn.Module):
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if 'x_kv' in kwargs.keys():
            kwargs['x_kv'] = self.norm(kwargs['x_kv'])

        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads=4, dropout=0.3, attn_bias=False, resolution=(5, 5)):
        super().__init__()

        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'

        dim_head = emb_dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.attn_bias = attn_bias
        if attn_bias:
            points = list(itertools.product(
                range(resolution[0]), range(resolution[1])))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])
            self.attention_biases = torch.nn.Parameter(
                torch.zeros(heads, len(attention_offsets)))
            self.register_buffer('attention_bias_idxs',
                                 torch.LongTensor(idxs).view(N, N),
                                 persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        if self.attn_bias:
            super().train(mode)
            if mode and hasattr(self, 'ab'):
                del self.ab
            else:
                self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x, return_attn=False):


        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.attn_bias:
            qk += (self.attention_biases[:, self.attention_bias_idxs]
                   if self.training else self.ab)


        attn_weights = self.attend(qk)
        if return_attn:
            attn_weights_averaged = attn_weights.mean(dim=1)

        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_attn:
            return self.to_out(out), attn_weights_averaged[:, 0]
        else:
            return self.to_out(out)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, heads=4, dropout=0., attn_bias=False):
        super().__init__()

        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'

        dim_head = emb_dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_kv = nn.Linear(emb_dim, emb_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_kv, return_attn=False):


        q = self.to_q(x_q)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(x_kv).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale


        attn_weights = self.attend(qk)
        if return_attn:
            attn_weights_averaged = attn_weights.mean(dim=1)

        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_attn:
            return self.to_out(out), attn_weights_averaged[:, 0]
        else:
            return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., attn_bias=False, resolution=(5, 5)):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadAttention(emb_dim, heads=heads, dropout=dropout, attn_bias=attn_bias,
                                                    resolution=resolution)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x, return_attn=return_attn)
                x += attn_out
                x = ff(x) + x

            else:
                x = attn(x) + x
                x = ff(x) + x

        if return_attn:
            return x, attn_weights
        else:
            return x


class CrossEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., attn_bias=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadCrossAttention(emb_dim, heads=heads, dropout=dropout, attn_bias=attn_bias)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x_q, x_kv, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x_q, x_kv=x_kv, return_attn=return_attn)
                x_q += attn_out
                x_q = ff(x_q) + x_q
            else:
                x_q = attn(x_q, x_kv=x_kv) + x_q
                x_q = ff(x_q) + x_q

        if return_attn:
            return x_q, attn_weights
        else:
            return x_q


class PEGH(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super(PEGH, self).__init__()

        self.proj1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, bias=True, groups=dim)

    def forward(self, x, pos):
        pos = pos - pos.min(0)[0]
        x_sparse = torch.sparse_coo_tensor(pos.T, x.squeeze())
        x_dense = x_sparse.to_dense().permute(2, 1, 0).unsqueeze(dim=0)

        x_pos = self.proj1(x_dense)


        mask = (x_dense.sum(dim=1) != 0.)
        x_pos = x_pos.masked_fill(~mask, 0.) + x_dense
        x_pos_sparse = x_pos.squeeze().permute(2, 1, 0).to_sparse(2)
        x_out = x_pos_sparse.values().unsqueeze(dim=0)

        return x_out


class GlobalEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., kernel_size=3):
        super().__init__()

        self.pos_layer = PEGH(dim=emb_dim, kernel_size=kernel_size)

        self.layer1 = TransformerEncoder(emb_dim, 1, heads, mlp_dim, dropout)
        self.layer2 = TransformerEncoder(emb_dim, depth - 1, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def foward_features(self, x, pos):
        x = self.layer1(x)

        x = self.pos_layer(x, pos)

        x = self.layer2(x)
        x = self.norm(x)

        return x

    def forward(self, x, position):
        x = self.foward_features(x, position)

        return x


class NeighborEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., resolution=(5, 5)):
        super().__init__()

        self.layer = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout, attn_bias=True, resolution=resolution)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):

        x = self.layer(x)
        x = self.norm(x)

        return x


class FusionEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        self.fusion_layer = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x_t=None, x_n=None, x_g=None):

        if x_g.dim() == 2:
            x_g = x_g.unsqueeze(1)
        fus1 = self.fusion_layer(x_g, x_t)
        fus2 = self.fusion_layer(x_g, x_n)
        fusion = (fus1 + fus2).squeeze(1)
        fusion = self.norm(fusion)

        return fusion


class TWOFusionEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        self.fusion_layer = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x_t=None,x_g=None):

        if x_g.dim() == 1:
            x_g = x_g.unsqueeze(0)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        if x_g.dim() == 2:
            x_g = x_g.unsqueeze(1)
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        x_g = self.fusion_layer(x_g, x_t)
        x_g = self.norm(x_g.squeeze())

        return x_g


class GuidedFeatureEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        self.fusion_layer = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x_e=None, x_s=None, x=None, whether=False):
        if whether is True:
            fusion = self.fusion_layer(x, x_e)
            fusion = self.norm(fusion)
            return fusion
        else:
            fus1 = self.fusion_layer(x, x_e)

            fus2 = self.fusion_layer(x, x_s)

            fusion = (fus1 + fus2).squeeze(1)
            fusion = self.norm(fusion)

            return fusion


class TargetFeatureExtractor(nn.Module):
    def __init__(self):
        super(TargetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool(x)

        return x.view(1, -1, 512)


class NeighborFeatureExtractor(nn.Module):
    def __init__(self):
        super(NeighborFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 512)

        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        return x


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

class GraphFeatureFusion(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphFeatureFusion, self).__init__()
        self.layer1 = GCNLayer(in_channels, hidden_channels)
        self.layer2 = GCNLayer(hidden_channels, out_channels)
        self.attention = GlobalAttention(gate_nn=nn.Linear(out_channels, 1))

    def forward(self, x, edge_index, batch):
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        x = self.layer2(x, edge_index)
        x = F.relu(x)
        x = self.attention(x, batch)
        return x

class HGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HGNNLayer, self).__init__()
        self.conv = pyg_nn.HypergraphConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        return x

class HGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HGNN, self).__init__()
        self.layer1 = HGNNLayer(in_channels, hidden_channels)
        self.layer2 = HGNNLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = torch.mean(x, dim=0, keepdim=True)
        return x

class EXPLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EXPLayer, self).__init__()
        self.conv = pyg_nn.HypergraphConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = x.float()
        x = self.conv(x, edge_index)
        x = F.relu(x)
        return x

class EXPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EXPNN, self).__init__()
        self.layer1 = EXPLayer(in_channels, hidden_channels)
        self.layer2 = EXPLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = torch.mean(x, dim=0, keepdim=True)
        return x

class GraphNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = torch.mean(x, dim=0, keepdim=True)
        return x
        
class Decoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=250):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
