import os
import inspect
import importlib
import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import torch.nn.functional as F
from einops import rearrange
import torch.distributed as dist
from model.modules import (
    HGNN,
    EXPNN,
    TWOFusionEncoder,
    Decoder,
    LEXPNN,
    LHGNN
)

def load_model_weights(path: str):
    resnet = torchvision.models.__dict__["resnet18"](weights=None)
    ckpt_dir = "./weights"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/tenpercent_resnet18.ckpt"

    if not os.path.exists(ckpt_path):
        ckpt_url = "https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt"
        wget.download(ckpt_url, out=ckpt_dir)

    state = torch.load(path,weights_only=False)
    state_dict = state["state_dict"]
    for i, key in enumerate(state_dict.keys()):
        print(f"  {i+1}. {key}")
    print("------------------------------------------------------------------")
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(key)
    model_dict = resnet.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    if state_dict == {}:
        print("No weight could be loaded..")
    model_dict.update(state_dict)
    resnet.load_state_dict(model_dict)
    resnet.fc = nn.Identity()
    return resnet

def print_shapes(data_list, name):
    if len(data_list) > 0:
        print(f"{name} first element shape:", data_list[0].shape)
    else:
        print(f"{name} is empty")

class STAG3D(pl.LightningModule):
    def __init__(self,fold,num_genes=250, emb_dim=512, depth1=2, num_heads1=8, mlp_ratio1=2.0, dropout1=0.1, res_neighbor=(5, 5), learning_rate=0.0001, temperature1=0.05, temperature2=0.05, temperature3=0.05, loss_ratio1=1.0, loss_ratio2=0.5, img_head=1,gene_head=1):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.best_loss = np.inf
        self.best_cor = -1
        self.num_genes = num_genes
        self.alpha = 0.3
        self.num_n = res_neighbor[0]
        self.ratio1 = loss_ratio1
        self.ratio2 = loss_ratio2
        self.temperature1 = temperature1
        self.temperature2 = temperature2
        self.temperature3 = temperature3
        self.img_head = img_head
        self.gene_head = gene_head
        self.fold = fold
        resnet18 = load_model_weights("weights/tenpercent_resnet18.ckpt")
        module = list(resnet18.children())[:-2]
        self.target_encoder = nn.Sequential(*module)
        self.fc_target = nn.Linear(emb_dim, num_genes)
        self.exp_encoder = nn.Sequential(
            nn.Linear(num_genes, emb_dim),
            nn.Linear(emb_dim, emb_dim)
        )
        self.fc = nn.Linear(25088, emb_dim)
        
        self.neighbor_encoder = HGNN(25088, 1024, 512,heads=self.img_head, num_edges=108)
        self.neighbor_exp_encoder = EXPNN(512, 1024, 512,heads=self.gene_head,num_edges=108)
        
        self.neighbor_encoder_mini = LHGNN(25088, 1024, 25088)
        self.neighbor_exp_encoder_mini = LEXPNN(512 , 1024, 512)
        
        
        self.cross_encoder = TWOFusionEncoder(emb_dim, depth1, num_heads1, int(emb_dim * mlp_ratio1), dropout1)
        self.decoder = Decoder(input_dim=emb_dim, output_dim=num_genes)
        self.data = 'stnet'
        self.best_mse = 1
        self.validation_step_outputs = []
        print("Model init")

    def build_hypergraph(self, neighbor_nodes, neighbor_exp, pad_size, k=3):
        actual_num_nodes = pad_size[0]
        neighbor_nodes = neighbor_nodes[:actual_num_nodes]
        neighbor_exp = neighbor_exp[:actual_num_nodes]
        num_nodes = actual_num_nodes
        x = self.target_encoder(neighbor_nodes)
        x_exp = self.exp_encoder(neighbor_exp)
        x = x.view(num_nodes, -1) 
        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.T)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1)
        _, topk_indices=torch.topk(sim_matrix ,k=k ,dim=1)
        topk_indices=topk_indices.long()
        hyperedge_indices=[]
        for i in range(num_nodes):
            hyperedge_indices.extend([(node_idx ,i)for node_idx in topk_indices[i]])
            hyperedge_indices.append((i ,i))
        
        if hyperedge_indices:
            rows ,cols=zip(*hyperedge_indices)
            hyperedge_index=torch.tensor([rows ,cols],dtype=torch.long)
        else:
            hyperedge_index=torch.tensor([[[],[]]],dtype=torch.long)
        return x,x_exp,hyperedge_index

    def build_hypergraph_mini(self, neighbor_nodes, neighbor_exp, pad_size, k=3):
        actual_num_nodes = 9
        neighbor_nodes = neighbor_nodes[:actual_num_nodes]
        neighbor_exp = neighbor_exp[:actual_num_nodes]
        num_nodes = actual_num_nodes
        x = self.target_encoder(neighbor_nodes)
        x_exp = self.exp_encoder(neighbor_exp)
        x = x.view(num_nodes, -1)  
        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.T)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1)
        _, topk_indices=torch.topk(sim_matrix ,k=k ,dim=1)
        topk_indices=topk_indices.long()
        hyperedge_indices=[]
        for i in range(num_nodes):
            hyperedge_indices.extend([(node_idx ,i)for node_idx in topk_indices[i]])
            hyperedge_indices.append((i ,i))
        if hyperedge_indices:
            rows ,cols=zip(*hyperedge_indices)
            hyperedge_index=torch.tensor([rows ,cols],dtype=torch.long)
        else:
            hyperedge_index=torch.tensor([[[],[]]],dtype=torch.long)
        return x,x_exp,hyperedge_index
    
    def forward(self, x, exp, x_neighbor, x_neighbor_exp, pad_sizes, image_names):
        x = x.squeeze()
        if x.dim() != 4:
            x = x.unsqueeze(0)
        x = self.target_encoder(x)
        _, dim, w, h = x.shape
        x = rearrange(x, "b d h w -> b (h w) d", d=dim, w=w, h=h)
        x = self.fc(x.reshape(x.shape[0], -1, 25088))
        x_neighbor = x_neighbor.to(torch.float32)
        x_neighbor_exp = x_neighbor_exp.to(torch.float32)
        assert(x_neighbor.dim() == 5)
        if x_neighbor.dim() == 4:
            neighbor, neighbor_exp, hyperedge = self.build_hypergraph(x_neighbor, x_neighbor_exp)
        elif x_neighbor.dim() == 5:
            batch_size, num_patches, channels, height, width = x_neighbor.size()
            neighbor = []
            neighbor_exp = []
            hyperedge = []
            neighbors_group1 = []
            neighbors_group2 = []
            neighbors_group3 = []
            neighbors_exp_group1 = []
            neighbors_exp_group2 = []
            neighbors_exp_group3 = []
            hyperedges_group1 = []
            hyperedges_group2 = []
            hyperedges_group3 = []
            for i in range(batch_size):
                for group in range(3):
                    start_idx = group * 9
                    end_idx = start_idx + 9
                    current_x_neighbor = x_neighbor[i][start_idx:end_idx].squeeze(0)
                    current_x_neighbor_exp = x_neighbor_exp[i][start_idx:end_idx].squeeze(0)
                    current_pad_sizes = pad_sizes[i][start_idx:end_idx]
                    n, n_exp, h = self.build_hypergraph_mini(current_x_neighbor.squeeze(0), current_x_neighbor_exp.squeeze(0), current_pad_sizes)
                    if group == 0:
                        neighbors_group1.append(n)
                        neighbors_exp_group1.append(n_exp)
                        hyperedges_group1.append(h)
                    elif group == 1:
                        neighbors_group2.append(n)
                        neighbors_exp_group2.append(n_exp)
                        hyperedges_group2.append(h)
                    elif group == 2:
                        neighbors_group3.append(n)
                        neighbors_exp_group3.append(n_exp)
                        hyperedges_group3.append(h)
                n, n_exp, h = self.build_hypergraph(x_neighbor[i].squeeze(0), x_neighbor_exp[i].squeeze(0),pad_sizes[i])
                neighbor.append(n)
                neighbor_exp.append(n_exp)
                hyperedge.append(h)
            
            neighbor = torch.stack(neighbor).view(batch_size, -1, 25088).to(x_neighbor.device)  
            neighbor_exp = torch.stack(neighbor_exp).view(batch_size, -1, 512).to(x_neighbor.device)
            hyperedge = torch.stack(hyperedge).to(x_neighbor.device)
            neighbors_group1 = torch.stack(neighbors_group1).view(batch_size, -1, 25088).to(x_neighbor.device)
            neighbors_exp_group1 = torch.stack(neighbors_exp_group1).view(batch_size, -1, 512).to(x_neighbor.device)
            hyperedges_group1 = torch.stack(hyperedges_group1).to(x_neighbor.device)
            neighbors_group2 = torch.stack(neighbors_group2).view(batch_size, -1, 25088).to(x_neighbor.device)
            neighbors_exp_group2 = torch.stack(neighbors_exp_group2).view(batch_size, -1, 512).to(x_neighbor.device)
            hyperedges_group2 = torch.stack(hyperedges_group2).to(x_neighbor.device)
            neighbors_group3 = torch.stack(neighbors_group3).view(batch_size, -1, 25088).to(x_neighbor.device)
            neighbors_exp_group3 = torch.stack(neighbors_exp_group3).view(batch_size, -1, 512).to(x_neighbor.device)
            hyperedges_group3 = torch.stack(hyperedges_group3).to(x_neighbor.device)
        neighbor = neighbor.view(x.shape[0], -1, 25088).to(x.device)
        neighbor_exp = neighbor_exp.view(x.shape[0], -1, 512).to(x.device)
        if hyperedge.dim() == 2:
            hyperedge = hyperedge.unsqueeze(0)
        hyperedge = hyperedge.to(x.device)
        
        neighbor_mini_1 = []
        neighbor_mini_1_exp = []
        neighbor_mini_2 = []
        neighbor_mini_2_exp = []
        neighbor_mini_3 = []
        neighbor_mini_3_exp = []
        for i in range(x.shape[0]):
            neighbor1_i = neighbors_group1[i]
            neighbor1_exp_i = neighbors_exp_group1[i]
            hyperedges_group1_i = hyperedges_group1[i]
            neighbors1 = self.neighbor_encoder_mini(neighbor1_i,hyperedges_group1_i).to(x.device)
            neighbors_exp1 = self.neighbor_exp_encoder_mini(neighbor1_exp_i,hyperedges_group1_i).to(x.device)
            neighbor_mini_1.append(neighbors1)
            neighbor_mini_1_exp.append(neighbors_exp1)
        for i in range(x.shape[0]):
            neighbor2_i = neighbors_group2[i]
            neighbor2_exp_i = neighbors_exp_group2[i]
            hyperedges_group2_i = hyperedges_group2[i]
            neighbors2 = self.neighbor_encoder_mini(neighbor2_i,hyperedges_group2_i).to(x.device)
            neighbors_exp2 = self.neighbor_exp_encoder_mini(neighbor2_exp_i,hyperedges_group2_i).to(x.device)
            neighbor_mini_2.append(neighbors2)
            neighbor_mini_2_exp.append(neighbors_exp2)
        for i in range(x.shape[0]):
            neighbor3_i = neighbors_group3[i]
            neighbor3_exp_i = neighbors_exp_group3[i]
            hyperedges_group3_i = hyperedges_group3[i]
            neighbors3 = self.neighbor_encoder_mini(neighbor3_i,hyperedges_group3_i).to(x.device)
            neighbors_exp3 = self.neighbor_exp_encoder_mini(neighbor3_exp_i,hyperedges_group3_i).to(x.device)
            neighbor_mini_3.append(neighbors3)
            neighbor_mini_3_exp.append(neighbors_exp3)
        
        
        neighbor_mini_combined = torch.stack([
            torch.cat([neighbor_mini_1[i], neighbor_mini_2[i], neighbor_mini_3[i]], dim=0)
            for i in range(x.shape[0])
        ], dim=0) 
        neighbor_mini_exp_combined = torch.stack([
            torch.cat([neighbor_mini_1_exp[i], neighbor_mini_2_exp[i], neighbor_mini_3_exp[i]], dim=0)
            for i in range(x.shape[0])
        ], dim=0)
        
        all_neighbors = []
        all_neighbor_exps = []
        for i in range(x.shape[0]):
            neighbor_i = neighbor_mini_combined[i]           
            neighbor_exp_i = neighbor_mini_exp_combined[i]
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

    def contrastive_loss(self, features1, features2, temperature, negative_weight=0.1):
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
    
    def training_step(self, batch, batch_idx):
        patch, exp, neighbor, neighbor_exp, pad_sizes,image_names = batch
        
        exp = exp.to(dtype=torch.float32)
        outputs = self(patch,exp,neighbor,neighbor_exp,pad_sizes,image_names)
        loss_patch = self.contrastive_loss(outputs[0].squeeze(), outputs[1].squeeze(), self.temperature1)
        loss_neighbor = self.contrastive_loss(outputs[2].squeeze(), outputs[3].squeeze(), self.temperature2)
        reconstruction_loss = F.mse_loss(outputs[4], exp)
        loss = self.ratio1 * loss_patch + self.ratio2 * loss_neighbor + reconstruction_loss
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, exp, neighbor, neighbor_exp, pad_sizes,image_names = batch
        exp = exp.to(dtype=torch.float32)
        outputs = self(patch,exp,neighbor,neighbor_exp,pad_sizes,image_names)
        pred_tensor = outputs[5]
        exp_tensor = exp
        self.validation_step_outputs.append({"pred": pred_tensor, "exp": exp_tensor})
    
        
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        if not outputs:
            return

        all_preds = torch.cat([x["pred"] for x in outputs], dim=0)
        all_exps = torch.cat([x["exp"] for x in outputs], dim=0)

        val_mse = F.mse_loss(all_preds, all_exps)
        val_mae = F.l1_loss(all_preds, all_exps)

        def vectorized_pcc(x, y, dim):
            x_mean = torch.mean(x, dim=dim, keepdim=True)
            y_mean = torch.mean(y, dim=dim, keepdim=True)
            x_centered = x - x_mean
            y_centered = y - y_mean
            
            covariance = torch.sum(x_centered * y_centered, dim=dim)
            x_std_dev = torch.sqrt(torch.sum(x_centered**2, dim=dim))
            y_std_dev = torch.sqrt(torch.sum(y_centered**2, dim=dim))
            
            pcc = covariance / (x_std_dev * y_std_dev)
            
            return torch.nan_to_num(pcc)

        gene_wise_pcc = vectorized_pcc(all_preds, all_exps, dim=0)
        
        sorted_gene_pcc, _ = torch.sort(gene_wise_pcc, descending=True)

        pcc_gene_mean_top10 = sorted_gene_pcc[:10].mean()
        pcc_gene_mean_top20 = sorted_gene_pcc[:20].mean()
        pcc_gene_mean_top50 = sorted_gene_pcc[:50].mean()
        pcc_gene_mean_top100 = sorted_gene_pcc[:100].mean()
        pcc_gene_mean_all = gene_wise_pcc.mean()

        spot_wise_pcc = vectorized_pcc(all_preds, all_exps, dim=1)
        pcc_spot_mean = spot_wise_pcc.mean()

        log_opts = {'on_epoch': True, 'prog_bar': True, 'logger': True, 'sync_dist': True}
        
        self.log("val_mse", val_mse, **log_opts)
        self.log("val_mae", val_mae, **log_opts)
        print(f"val_mse : {val_mse}")
        print(f"val_mae : {val_mae}")
        
        self.log("R_spot_mean", pcc_spot_mean, **log_opts) 
        
        log_opts_no_bar = {'on_epoch': True, 'prog_bar': False, 'logger': True, 'sync_dist': True}
        self.log("R_gene_mean_all", pcc_gene_mean_all, **log_opts_no_bar)
        self.log("R_gene_mean_top10", pcc_gene_mean_top10, **log_opts_no_bar)
        self.log("R_gene_mean_top20", pcc_gene_mean_top20, **log_opts_no_bar)
        self.log("R_gene_mean_top50", pcc_gene_mean_top50, **log_opts_no_bar)
        self.log("R_gene_mean_top100", pcc_gene_mean_top100, **log_opts_no_bar)
        print(f"pcc_gene_mean_all : {pcc_gene_mean_all}")
        print(f"pcc_gene_mean_top10 : {pcc_gene_mean_top10}")
        print(f"pcc_gene_mean_top20 : {pcc_gene_mean_top20}")
        print(f"pcc_gene_mean_top50 : {pcc_gene_mean_top50}")
        print(f"pcc_gene_mean_top100 : {pcc_gene_mean_top100}")
        print(f"pcc_spot_mean : {pcc_spot_mean}")

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {"optimizer": optim, "lr_scheduler": StepLR}
        return optim_dict
