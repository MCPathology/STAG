
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import os
import sys
import time
import json
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from einops import rearrange

from torch_geometric.nn import HypergraphConv, GCNConv, GATConv

from models.models.module import TWOFusionEncoder, Decoder
from dataset.TextcSCCDataset import cSCCDataset as Text_cSCC_Dataset
from dataset.cSCCDataset import cSCCDataset as Original_cSCCDataset


class HGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = HypergraphConv(in_channels, out_channels)
    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))

class HGNN_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.layer1 = HGNNLayer(in_channels, hidden_channels)
        self.layer2 = HGNNLayer(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        return torch.mean(x, dim=0, keepdim=True)

class GCN_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.mean(x, dim=0, keepdim=True)

class GAT_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.mean(x, dim=0, keepdim=True)

def load_model_weights(path):
    resnet = torchvision.models.resnet18(weights=None)
    ckpt_path = path
    if not os.path.exists(ckpt_path):
        print(f"Warning: {ckpt_path} not found.")
        resnet.fc = nn.Identity()
        return resnet
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = state["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(key)
    model_dict = resnet.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    resnet.load_state_dict(model_dict)
    resnet.fc = nn.Identity()
    return resnet


class STAG_Ablation(nn.Module):

    def __init__(self, num_genes=250, emb_dim=512, depth1=2, num_heads1=8,
                 mlp_ratio1=2.0, dropout1=0.1, gnn_type='hgnn'):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_genes = num_genes
        self.emb_dim = emb_dim

        resnet18 = load_model_weights("weights/tenpercent_resnet18.ckpt")
        module = list(resnet18.children())[:-2]
        self.target_encoder = nn.Sequential(*module)

        self.exp_encoder = nn.Sequential(
            nn.Linear(num_genes, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(768, emb_dim), nn.GELU(), nn.Dropout(0.1)
        )

        mlp_dim = int(emb_dim * mlp_ratio1)
        self.text_to_image_attn = TWOFusionEncoder(emb_dim, depth1, num_heads1, mlp_dim, dropout1)
        self.text_to_exp_attn = TWOFusionEncoder(emb_dim, depth1, num_heads1, mlp_dim, dropout1)

        if gnn_type == 'hgnn':
            self.neighbor_encoder = HGNN_Encoder(25088, 1024, emb_dim)
            self.neighbor_exp_encoder = HGNN_Encoder(emb_dim, 1024, emb_dim)
        elif gnn_type == 'gcn':
            self.neighbor_encoder = GCN_Encoder(25088, 1024, emb_dim)
            self.neighbor_exp_encoder = GCN_Encoder(emb_dim, 1024, emb_dim)
        elif gnn_type == 'gat':
            self.neighbor_encoder = GAT_Encoder(25088, 1024, emb_dim)
            self.neighbor_exp_encoder = GAT_Encoder(emb_dim, 1024, emb_dim)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.cross_encoder = TWOFusionEncoder(emb_dim, depth1, num_heads1, mlp_dim, dropout1)

        self.fc = nn.Linear(25088, emb_dim)
        self.decoder = Decoder(input_dim=emb_dim, output_dim=num_genes)

    def build_graph(self, neighbor_nodes, neighbor_exp, k=3):
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

        if self.gnn_type == 'hgnn':
            hyperedge_indices = []
            for i in range(num_nodes):
                hyperedge_indices.extend([(node_idx.item(), i) for node_idx in topk_indices[i]])
                hyperedge_indices.append((i, i))
            if hyperedge_indices:
                rows, cols = zip(*hyperedge_indices)
                edge_index = torch.tensor([rows, cols], dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            src, dst = [], []
            for i in range(num_nodes):
                for j in topk_indices[i]:
                    src.extend([i, j.item()])
                    dst.extend([j.item(), i])
                src.append(i)
                dst.append(i)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_index = torch.unique(edge_index, dim=1)

        return x, x_exp, edge_index.long()

    def forward(self, x, exp, x_neighbor, x_neighbor_exp, text):
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

        x_neighbor = x_neighbor.to(torch.float32)
        x_neighbor_exp = x_neighbor_exp.to(torch.float32)

        if x_neighbor.dim() == 4:
            neighbor_nodes, neighbor_exp_nodes, edge_idx = self.build_graph(x_neighbor, x_neighbor_exp)
            neighbor_nodes = neighbor_nodes.unsqueeze(0)
            neighbor_exp_nodes = neighbor_exp_nodes.unsqueeze(0)
            e_list = [edge_idx]
        elif x_neighbor.dim() == 5:
            batch_size = x_neighbor.size(0)
            n_list, ne_list, e_list = [], [], []
            for i in range(batch_size):
                n, ne, e = self.build_graph(x_neighbor[i], x_neighbor_exp[i])
                n_list.append(n)
                ne_list.append(ne)
                e_list.append(e)
            neighbor_nodes = torch.stack(n_list).view(batch_size, -1, 25088)
            neighbor_exp_nodes = torch.stack(ne_list).view(batch_size, -1, self.emb_dim)

        neighbor_nodes = neighbor_nodes.view(B, -1, 25088).to(x.device)
        neighbor_exp_nodes = neighbor_exp_nodes.view(B, -1, self.emb_dim).to(x.device)
        if isinstance(e_list, list) and len(e_list) > 0 and isinstance(e_list[0], torch.Tensor):
            edge_idx_list = [e.to(x.device) for e in e_list]
        else:
            if edge_idx.dim() == 2:
                edge_idx = edge_idx.unsqueeze(0)
            edge_idx = edge_idx.to(x.device)
            edge_idx_list = [edge_idx[i] for i in range(B)]

        all_neighbors, all_neighbor_exps = [], []
        for i in range(B):
            neighbors_i = self.neighbor_encoder(neighbor_nodes[i], edge_idx_list[i]).view(1, -1)
            neighbor_exps_i = self.neighbor_exp_encoder(neighbor_exp_nodes[i], edge_idx_list[i]).view(1, -1)
            all_neighbors.append(neighbors_i)
            all_neighbor_exps.append(neighbor_exps_i)

        neighbors = torch.cat(all_neighbors, dim=0).to(x.device)
        neighbor_exps = torch.cat(all_neighbor_exps, dim=0).to(x.device)

        fused_target_img = self.cross_encoder(patch_exp, patch_fusion)
        fused_target_exp = self.cross_encoder(patch_fusion, patch_exp)
        fused_neighbor_img = self.cross_encoder(neighbor_exps, neighbors)
        fused_neighbor_exp = self.cross_encoder(neighbors, neighbor_exps)

        pred_exp = self.decoder(patch_fusion)

        return fused_target_img, fused_target_exp, fused_neighbor_img, fused_neighbor_exp, pred_exp, pred_exp


def contrastive_loss(features1, features2, temperature, negative_weight=0.1):
    if features1.dim() == 1: features1 = features1.unsqueeze(0)
    if features2.dim() == 1: features2 = features2.unsqueeze(0)
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)
    similarity_matrix = torch.mm(features1, features2.t()) / temperature
    batch_size = features1.size(0)
    mask = torch.eye(batch_size, device=features1.device)
    similarity_matrix = similarity_matrix * mask + similarity_matrix * (1 - mask) * negative_weight
    labels = torch.arange(batch_size, device=features1.device)
    return F.cross_entropy(similarity_matrix, labels)


def evaluate_model(model, valloader, device):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for data in tqdm(valloader, desc="  Evaluating", leave=False, ncols=80):
            _, img, exp, hg_x, hg_x_exp, text = data
            img = img.to(device, dtype=torch.float)
            exp = exp.to(device, dtype=torch.float)
            hg_x = hg_x.to(device, dtype=torch.float)
            hg_x_exp = hg_x_exp.to(device, dtype=torch.float)
            text = text[0].to(device, dtype=torch.float)

            outputs = model(x=img, exp=exp, x_neighbor=hg_x, x_neighbor_exp=hg_x_exp, text=text)
            pred_exp = outputs[4]

            preds_np = pred_exp.cpu().numpy()
            gts_np = exp.cpu().numpy()
            if preds_np.ndim == 1: preds_np = preds_np.reshape(1, -1)
            if gts_np.ndim == 1: gts_np = gts_np.reshape(1, -1)
            all_preds.append(preds_np)
            all_gts.append(gts_np)

    all_preds_np = np.concatenate(all_preds, axis=0)
    all_gts_np = np.concatenate(all_gts, axis=0)

    per_gene_pcc = []
    for g in range(all_gts_np.shape[1]):
        pcc, _ = pearsonr(all_preds_np[:, g], all_gts_np[:, g])
        per_gene_pcc.append(pcc if not np.isnan(pcc) else 0)

    sorted_pcc = np.sort(np.array(per_gene_pcc))[::-1]
    mse = np.mean((all_preds_np - all_gts_np) ** 2)

    return {
        'pcc_250': np.mean(sorted_pcc),
        'pcc_200': np.mean(sorted_pcc[:200]) if len(sorted_pcc) >= 200 else np.mean(sorted_pcc),
        'pcc_150': np.mean(sorted_pcc[:150]) if len(sorted_pcc) >= 150 else np.mean(sorted_pcc),
        'pcc_100': np.mean(sorted_pcc[:100]) if len(sorted_pcc) >= 100 else np.mean(sorted_pcc),
        'pcc_50': np.mean(sorted_pcc[:50]) if len(sorted_pcc) >= 50 else np.mean(sorted_pcc),
        'pcc_20': np.mean(sorted_pcc[:20]) if len(sorted_pcc) >= 20 else np.mean(sorted_pcc),
        'pcc_10': np.mean(sorted_pcc[:10]) if len(sorted_pcc) >= 10 else np.mean(sorted_pcc),
        'rmse': np.sqrt(mse),
        'mse': mse,
        'mae': np.mean(np.abs(all_preds_np - all_gts_np)),
    }


def run_one_variant(gnn_type, device, args):
    print(f"\n{'#'*60}")
    print(f"  GNN Type: {gnn_type.upper()}")
    print(f"{'#'*60}")

    data_path = './data/GSE144240'
    split_save_path = os.path.join(args.save_dir, f'cSCC_kfold_splits.json')

    _ = Original_cSCCDataset(path=data_path, mode='train', k_folds=args.k_folds,
                              fold_index=0, seed=args.seed, split_save_path=split_save_path)
    with open(split_save_path, 'r') as f:
        split_data = json.load(f)

    fold_results = []
    for fold_info in split_data['folds']:
        fold = fold_info['fold_index']
        print(f"\n{'='*25} {gnn_type.upper()} - FOLD {fold+1}/{args.k_folds} {'='*25}")

        model = STAG_Ablation(
            num_genes=250, emb_dim=512, depth1=2, num_heads1=8,
            gnn_type=gnn_type
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        gnn_params = sum(p.numel() for p in model.neighbor_encoder.parameters()) + \
                     sum(p.numel() for p in model.neighbor_exp_encoder.parameters())
        print(f"  Total params: {total_params/1e6:.2f}M | GNN params: {gnn_params/1e6:.2f}M")

        dataset_kwargs = {
            'path': data_path, 'k_folds': args.k_folds,
            'fold_index': fold, 'seed': args.seed, 'split_save_path': split_save_path,
        }
        train_dataset = Text_cSCC_Dataset(mode='train', **dataset_kwargs)
        val_dataset = Text_cSCC_Dataset(mode='val', **dataset_kwargs)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        mse_loss_fn = nn.MSELoss()

        best_pcc = -1.0
        best_metrics = {}
        epoch_times = []

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            torch.cuda.synchronize()
            epoch_start = time.time()

            for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False, ncols=80):
                _, img, exp, hg_x, hg_x_exp, text = data
                img = img.to(device, dtype=torch.float)
                exp = exp.to(device, dtype=torch.float)
                hg_x = hg_x.to(device, dtype=torch.float)
                hg_x_exp = hg_x_exp.to(device, dtype=torch.float)
                text = text[0].to(device, dtype=torch.float)

                optimizer.zero_grad()
                outputs = model(x=img, exp=exp, x_neighbor=hg_x, x_neighbor_exp=hg_x_exp, text=text)
                fused_t_img, fused_t_exp, fused_n_img, fused_n_exp, pred_exp, _ = outputs

                loss_patch = contrastive_loss(fused_t_img, fused_t_exp, args.temp1)
                loss_neighbor = contrastive_loss(fused_n_img, fused_n_exp, args.temp2)
                recon_loss = mse_loss_fn(pred_exp, exp)
                loss = args.loss_ratio1 * loss_patch + args.loss_ratio2 * loss_neighbor + recon_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            torch.cuda.synchronize()
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            avg_loss = total_loss / len(train_loader)

            metrics = evaluate_model(model, val_loader, device)
            print(f"  [{gnn_type.upper()}] Fold {fold+1} Epoch {epoch+1}: loss={avg_loss:.4f} | "
                  f"PCC_250={metrics['pcc_250']:.4f} | PCC_50={metrics['pcc_50']:.4f} | time={epoch_time:.1f}s")

            if metrics['pcc_250'] > best_pcc:
                best_pcc = metrics['pcc_250']
                best_metrics = metrics.copy()
                best_metrics['best_epoch'] = epoch + 1

        model.eval()
        torch.cuda.synchronize()
        infer_start = time.time()
        with torch.no_grad():
            for data in val_loader:
                _, img, exp, hg_x, hg_x_exp, text = data
                img = img.to(device, dtype=torch.float)
                exp = exp.to(device, dtype=torch.float)
                hg_x = hg_x.to(device, dtype=torch.float)
                hg_x_exp = hg_x_exp.to(device, dtype=torch.float)
                text = text[0].to(device, dtype=torch.float)
                _ = model(x=img, exp=exp, x_neighbor=hg_x, x_neighbor_exp=hg_x_exp, text=text)
        torch.cuda.synchronize()
        infer_total = time.time() - infer_start
        infer_per_sample = infer_total / len(val_dataset) * 1000

        best_metrics['fold'] = fold + 1
        best_metrics['gnn_type'] = gnn_type
        best_metrics['total_params_M'] = total_params / 1e6
        best_metrics['gnn_params_M'] = gnn_params / 1e6
        best_metrics['avg_epoch_time_s'] = np.mean(epoch_times)
        best_metrics['infer_ms_per_sample'] = infer_per_sample
        fold_results.append(best_metrics)
        print(f"  [{gnn_type.upper()}] Fold {fold+1} Best PCC: {best_pcc:.4f} (epoch {best_metrics['best_epoch']})")
        print(f"  Avg epoch time: {np.mean(epoch_times):.1f}s | Inference: {infer_per_sample:.2f}ms/sample")

    return fold_results


def main():
    parser = argparse.ArgumentParser(description="R1-Q6: GNN Ablation on cSCC")
    parser.add_argument('--gnn_type', type=str, required=True, choices=['hgnn', 'gcn', 'gat'],
                        help="GNN variant to run (run 3 processes in parallel for all variants)")
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1553)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--loss_ratio1', type=float, default=0.4)
    parser.add_argument('--loss_ratio2', type=float, default=0.2)
    parser.add_argument('--temp1', type=float, default=0.05)
    parser.add_argument('--temp2', type=float, default=0.05)
    parser.add_argument('--save_dir', type=str, default='./ablation_gnn_results')
    args = parser.parse_args()

    gnn_type = args.gnn_type
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"R1-Q6: GNN Ablation — {gnn_type.upper()} on cSCC")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"  Device: {device}")
    print("=" * 60)

    fold_results = run_one_variant(gnn_type, device, args)

    df = pd.DataFrame(fold_results)
    detail_path = os.path.join(args.save_dir, f'gnn_ablation_{gnn_type}_per_fold.csv')
    df.to_csv(detail_path, index=False, float_format='%.4f')
    print(f"\nPer-fold results saved to: {detail_path}")

    metric_cols = ['pcc_250', 'pcc_200', 'pcc_150', 'pcc_100', 'pcc_50', 'pcc_20', 'pcc_10', 'rmse', 'mse', 'mae']
    row = {'gnn_type': gnn_type.upper()}
    for col in metric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        row[f'{col}_mean'] = mean_val
        row[f'{col}_std'] = std_val
        row[col] = f"{mean_val:.4f}+/-{std_val:.4f}"

    row['total_params_M'] = df['total_params_M'].iloc[0]
    row['gnn_params_M'] = df['gnn_params_M'].iloc[0]
    row['avg_epoch_time_s'] = df['avg_epoch_time_s'].mean()
    row['infer_ms_per_sample'] = df['infer_ms_per_sample'].mean()

    summary_df = pd.DataFrame([row])
    summary_path = os.path.join(args.save_dir, f'gnn_ablation_{gnn_type}_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    print("\n" + "=" * 70)
    print(f"GNN Ablation Summary — {gnn_type.upper()} (cSCC)")
    print("=" * 70)
    print(f"{'GNN Type':<10} {'PCC_250':<18} {'PCC_50':<18} {'RMSE':<18} {'Params(M)':<12} {'GNN(M)':<10} {'Epoch(s)':<10} {'Infer(ms)':<10}")
    print("-" * 98)
    print(f"{row['gnn_type']:<10} {row['pcc_250']:<18} {row['pcc_50']:<18} {row['rmse']:<18} "
          f"{row['total_params_M']:<12.2f} {row['gnn_params_M']:<10.2f} {row['avg_epoch_time_s']:<10.1f} {row['infer_ms_per_sample']:<10.2f}")
    print("=" * 98)


if __name__ == '__main__':
    main()
