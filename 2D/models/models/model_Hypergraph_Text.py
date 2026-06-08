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
    def __init__(self, num_genes=250, emb_dim=512, depth1=2, num_heads1=8, mlp_ratio1=2.0, dropout1=0.1, ablation_mode='full'):
        super().__init__()
        

        resnet18 = load_model_weights("weights/tenpercent_resnet18.ckpt",weights_only = False)
        module = list(resnet18.children())[:-2]
        self.target_encoder = nn.Sequential(*module)
        self.ablation_mode = ablation_mode
        self.num_genes = num_genes
        self.emb_dim = emb_dim
        
        self.exp_encoder = nn.Sequential(
            nn.Linear(num_genes, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(768, emb_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.text_to_image_attn = TWOFusionEncoder(
            emb_dim=emb_dim, 
            depth=depth1,
            heads=num_heads1, 
            mlp_dim=int(emb_dim * mlp_ratio1), 
            dropout=dropout1
        )
        
        self.text_to_exp_attn = TWOFusionEncoder(
            emb_dim=emb_dim, 
            depth=depth1, 
            heads=num_heads1, 
            mlp_dim=int(emb_dim * mlp_ratio1), 
            dropout=dropout1
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


    def forward(self, x, exp, x_neighbor, x_neighbor_exp, text):
        if self.ablation_mode == 'full':
            self.ablation_mode = 'full_with_ca_cl'
            
        use_neighbor_branch = 'query' not in self.ablation_mode
        use_ca = 'with_ca' in self.ablation_mode

        B = x.shape[0] if x.dim() == 4 else x.squeeze(0).shape[0]
        x = x.squeeze()
        if x.dim() != 4:
            x = x.unsqueeze(0)
        
        x_feat = self.target_encoder(x)
        _, dim, w, h = x_feat.shape
        x_feat = rearrange(x_feat, "b d h w -> b (h w) d", d=dim, w=w, h=h)
        x_feat = self.fc(x_feat.reshape(x.shape[0], -1, 25088)) 
        
        exp_orig = self.exp_encoder(exp)
        
        if text.dim() == 3 and text.shape[0] > 1:
            text_single = text[0]
        elif text.dim() == 2:
            text_single = text
        else:
            text_single = text.squeeze(0) 
        
        text_feat = self.text_encoder(text_single) 
        text_query = text_feat.unsqueeze(0).repeat(B, 1, 1) 
        patch_fusion = self.text_to_image_attn(text_query, x_feat).reshape(B, -1)
        patch_exp = self.text_to_exp_attn(text_query, exp_orig.unsqueeze(1)).reshape(B, -1)
        
        
        patch_fusion = x_feat.reshape(B, -1)
        patch_exp = exp_orig.reshape(B, -1)

        if use_neighbor_branch:
            x_neighbor = x_neighbor.to(torch.float32)
            x_neighbor_exp = x_neighbor_exp.to(torch.float32)
            
            if x_neighbor.dim() == 4:
                neighbor_nodes, neighbor_exp_nodes, hyperedge = self.build_hypergraph(x_neighbor, x_neighbor_exp)
            elif x_neighbor.dim() == 5:
                batch_size = x_neighbor.size(0)
                neighbor_nodes_list, neighbor_exp_nodes_list, hyperedge_list = [], [], []
                for i in range(batch_size):
                    n, n_exp, h = self.build_hypergraph(x_neighbor[i].squeeze(0), x_neighbor_exp[i].squeeze(0))
                    neighbor_nodes_list.append(n)
                    neighbor_exp_nodes_list.append(n_exp)
                    hyperedge_list.append(h)
                neighbor_nodes = torch.stack(neighbor_nodes_list).view(batch_size, -1, 25088).to(x_neighbor.device)
                neighbor_exp_nodes = torch.stack(neighbor_exp_nodes_list).view(batch_size, -1, self.emb_dim).to(x_neighbor.device)
                hyperedge = torch.stack(hyperedge_list).to(x_neighbor.device)
            
            neighbor_nodes = neighbor_nodes.view(B, -1, 25088).to(x.device)
            neighbor_exp_nodes = neighbor_exp_nodes.view(B, -1, self.emb_dim).to(x.device)
            if hyperedge.dim() == 2:
                hyperedge = hyperedge.unsqueeze(0)
            hyperedge = hyperedge.to(x.device)

            all_neighbors, all_neighbor_exps = [], []
            for i in range(B):
                neighbors_i = self.neighbor_encoder(neighbor_nodes[i], hyperedge[i]).view(1, -1)
                neighbor_exps_i = self.neighbor_exp_encoder(neighbor_exp_nodes[i], hyperedge[i]).view(1, -1)
                all_neighbors.append(neighbors_i)
                all_neighbor_exps.append(neighbor_exps_i)
            
            neighbors = torch.cat(all_neighbors, dim=0).to(x.device)
            neighbor_exps = torch.cat(all_neighbor_exps, dim=0).to(x.device)
        else:
            placeholder_shape = (B, self.emb_dim)
            neighbors = torch.zeros(placeholder_shape, device=x.device, dtype=x.dtype)
            neighbor_exps = torch.zeros(placeholder_shape, device=x.device, dtype=x.dtype)

        if use_ca:
            fused_target_img = self.cross_encoder(patch_exp, patch_fusion)
            fused_target_exp = self.cross_encoder(patch_fusion, patch_exp)
            
            if use_neighbor_branch:
                fused_neighbor_img = self.cross_encoder(neighbor_exps, neighbors)
                fused_neighbor_exp = self.cross_encoder(neighbors, neighbor_exps)
            else:
                fused_neighbor_img = neighbors
                fused_neighbor_exp = neighbor_exps
        else:
            fused_target_img = patch_fusion
            fused_target_exp = patch_exp
            fused_neighbor_img = neighbors
            fused_neighbor_exp = neighbor_exps

        pred_exp = self.decoder(patch_fusion)
        
        return fused_target_img, fused_target_exp, fused_neighbor_img, fused_neighbor_exp, pred_exp, pred_exp

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
