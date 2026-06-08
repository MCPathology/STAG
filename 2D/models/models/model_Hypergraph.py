import os
import wget
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import HypergraphConv

from .module import (
    HGNN,
    EXPNN,
    TWOFusionEncoder,
    Decoder
)

def load_model_weights(path: str,weights_only = False):
    resnet = torchvision.models.__dict__["resnet18"](weights=None)
    ckpt_dir = "./weights"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/tenpercent_resnet18.ckpt"

    if not os.path.exists(ckpt_path):
        print(f"权重文件未找到，正在从网络下载到 {ckpt_dir}...")
        ckpt_url = "https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt"
        wget.download(ckpt_url, out=ckpt_dir)
        print("下载完成。")

    state = torch.load(path,weights_only=weights_only)
    state_dict = state["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(key)

    model_dict = resnet.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    if not state_dict:
        print("警告：没有权重可以被加载到模型中。")
    
    model_dict.update(state_dict)
    resnet.load_state_dict(model_dict)
    
    resnet.fc = nn.Identity()
    return resnet


class STAG(nn.Module):
    def __init__(self, num_genes=250, emb_dim=512, depth1=2, num_heads1=8, mlp_ratio1=2.0, dropout1=0.1):
        super().__init__()
        

        resnet18 = load_model_weights("weights/tenpercent_resnet18.ckpt",weights_only = False)
        module = list(resnet18.children())[:-2]
        self.target_encoder = nn.Sequential(*module)
        
        self.exp_encoder = nn.Sequential(
            nn.Linear(num_genes, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.neighbor_encoder = HGNN(in_channels=25088, hidden_channels=1024, out_channels=emb_dim)
        
        self.neighbor_exp_encoder = EXPNN(in_channels=emb_dim, hidden_channels=1024, out_channels=emb_dim)
        
        mlp_dim = int(emb_dim * mlp_ratio1)
        self.cross_encoder = TWOFusionEncoder(
            emb_dim=emb_dim, 
            depth=depth1, 
            heads=num_heads1, 
            mlp_dim=mlp_dim, 
            dropout=dropout1
        )
        
        self.fc = nn.Linear(25088, emb_dim)
        
        self.decoder = Decoder(input_dim=emb_dim, output_dim=num_genes)

    def build_hypergraph(self, neighbor_nodes, neighbor_exp, k=3):
        num_nodes = neighbor_nodes.size(0)
        x = self.target_encoder(neighbor_nodes)
        x_exp = self.exp_encoder(neighbor_exp)
        x = x.view(-1, 25088)
        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.T)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1)
        _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        topk_indices = topk_indices.long()
        hyperedge_indices = []
        for i in range(num_nodes):
            hyperedge_indices.extend([(node_idx, i) for node_idx in topk_indices[i]])
            hyperedge_indices.append((i, i))
        if hyperedge_indices:
            rows, cols = zip(*hyperedge_indices)
            hyperedge_index = torch.tensor([rows, cols], dtype=torch.long)
        else:
            hyperedge_index = torch.tensor([[[], []]], dtype=torch.long)
        hyperedge_index = hyperedge_index.long()
        return x, x_exp, hyperedge_index

    def forward(self, x, exp, x_neighbor, x_neighbor_exp):
        x = x.squeeze()
        if x.dim() != 4:
            x = x.unsqueeze(0)
        x = self.target_encoder(x)
        _, dim, w, h = x.shape
        x = rearrange(x, "b d h w -> b (h w) d", d=dim, w=w, h=h)
        x = self.fc(x.reshape(x.shape[0], -1, 25088))

        x_neighbor = x_neighbor.to(torch.float32)
        x_neighbor_exp = x_neighbor_exp.to(torch.float32)

        if x_neighbor.dim() == 4:
            neighbor, neighbor_exp, hyperedge = self.build_hypergraph(x_neighbor, x_neighbor_exp)
        elif x_neighbor.dim() == 5:
            batch_size, num_patches, channels, height, width = x_neighbor.size()
            neighbor = []
            neighbor_exp = []
            hyperedge = []
            for i in range(batch_size):
                n, n_exp, h = self.build_hypergraph(x_neighbor[i].squeeze(0), x_neighbor_exp[i].squeeze(0))
                neighbor.append(n)
                neighbor_exp.append(n_exp)
                hyperedge.append(h)
            neighbor = torch.stack(neighbor).view(batch_size, -1, 25088).to(x_neighbor.device)
            neighbor_exp = torch.stack(neighbor_exp).view(batch_size, -1, 512).to(x_neighbor.device)
            hyperedge = torch.stack(hyperedge).to(x_neighbor.device)

        neighbor = neighbor.view(x.shape[0], -1, 25088).to(x.device)
        neighbor_exp = neighbor_exp.view(x.shape[0], -1, 512).to(x.device)
        if hyperedge.dim() == 2:
            hyperedge = hyperedge.unsqueeze(0)
        hyperedge = hyperedge.to(x.device)

        all_neighbors = []
        all_neighbor_exps = []
        for i in range(x.shape[0]):
            neighbor_i = neighbor[i]
            neighbor_exp_i = neighbor_exp[i]
            hyperedge_i = hyperedge[i]
            neighbors = self.neighbor_encoder(neighbor_i, hyperedge_i).view(1, -1).to(x.device)
            neighbor_exps = self.neighbor_exp_encoder(neighbor_exp_i, hyperedge_i).view(1, -1).to(x.device)
            all_neighbors.append(neighbors)
            all_neighbor_exps.append(neighbor_exps)
        neighbors = torch.stack(all_neighbors, dim=0)
        neighbor_exps = torch.stack(all_neighbor_exps, dim=0)

        patch_fusion = x.reshape(x.shape[0], -1).to(x.device)
        patch_exp = self.exp_encoder(exp).reshape(x.shape[0], -1).to(x.device)
        neighbors = neighbors.reshape(x.shape[0], -1).to(x.device)
        neighbor_exps = neighbor_exps.reshape(x.shape[0], -1).to(x.device)
        pred_exp = self.decoder(patch_fusion)
        decoded_exp = self.decoder(patch_fusion)

        patch_fusion = self.cross_encoder(patch_exp, patch_fusion)
        patch_exp = self.cross_encoder(patch_fusion, patch_exp)
        neighbors = self.cross_encoder(neighbor_exps, neighbors)
        neighbor_exps = self.cross_encoder(neighbors, neighbor_exps)

        return patch_fusion, patch_exp, neighbors, neighbor_exps, decoded_exp, pred_exp

def contrastive_loss(features1, features2, temperature, negative_weight=0.1):
    if features1.dim() == 1:
        features1 = features1.unsqueeze(0)
    if features2.dim() == 1:
        features2 = features2.unsqueeze(0)
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)
    similarity_matrix = torch.mm(features1, features2.t()) / temperature
    batch_size = features1.size(0)
    mask = torch.eye(batch_size, device=features1.device)
    similarity_matrix = similarity_matrix * mask + similarity_matrix * (1 - mask) * negative_weight
    labels = torch.arange(batch_size, device=features1.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
