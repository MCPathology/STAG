
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

from torch_geometric.nn import HypergraphConv

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


def load_resnet18_weights(path):
    resnet = torchvision.models.resnet18(weights=None)
    if not os.path.exists(path):
        print(f"Warning: {path} not found, using random init.")
        resnet.fc = nn.Identity()
        return resnet
    state = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = state["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(key)
    model_dict = resnet.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    resnet.load_state_dict(model_dict)
    resnet.fc = nn.Identity()
    return resnet


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = load_resnet18_weights("weights/tenpercent_resnet18.ckpt")
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feat_dim = 25088

    def forward(self, x):
        feat = self.backbone(x)
        return feat.reshape(feat.size(0), -1)


class UNIEncoder(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        import timm
        self.model = timm.create_model(
            'vit_large_patch16_224',
            init_values=1e-5,
            dynamic_img_size=True,
            num_classes=0,
            pretrained=False
        )
        print(f"Loading UNI checkpoint from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.blocks[-2:].parameters():
            p.requires_grad = True
        for p in self.model.norm.parameters():
            p.requires_grad = True
        self.feat_dim = 1024

    def forward(self, x):
        return self.model(x)


class ConchEncoder(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self._load_model(ckpt_path)
        for p in self.parameters():
            p.requires_grad = False
        if self._use_conch_api:
            visual = self.model.visual
        else:
            visual = self.model.visual
        if hasattr(visual, 'transformer'):
            for p in visual.transformer.resblocks[-2:].parameters():
                p.requires_grad = True
            if hasattr(visual, 'ln_post'):
                for p in visual.ln_post.parameters():
                    p.requires_grad = True
        elif hasattr(visual, 'trunk'):
            for p in visual.trunk.blocks[-2:].parameters():
                p.requires_grad = True
            if hasattr(visual.trunk, 'norm'):
                for p in visual.trunk.norm.parameters():
                    p.requires_grad = True
        self.feat_dim = 512

    def _load_model(self, ckpt_path):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            model, _ = create_model_from_pretrained('conch_ViT-B-16', ckpt_path)
            self.model = model
            self._use_conch_api = True
            print(f"Loaded CONCH via conch package from: {ckpt_path}")
        except (ImportError, Exception) as e:
            print(f"conch package not available ({e}), trying open_clip...")
            import open_clip
            model = open_clip.create_model('ViT-B-16', pretrained=False)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'model' in ckpt:
                ckpt = ckpt['model']
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"Loaded CONCH via open_clip. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            self.model = model
            self._use_conch_api = False

    def forward(self, x):
        if self._use_conch_api:
            return self.model.encode_image(x, proj_contrast=False, normalize=False)
        else:
            return self.model.encode_image(x)


class MLPExpEncoder(nn.Module):
    def __init__(self, num_genes, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_genes, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
    def forward(self, x):
        return self.net(x)


class SNNExpEncoder(nn.Module):
    def __init__(self, num_genes, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_genes, emb_dim),
            nn.SELU(),
            nn.AlphaDropout(p=0.1),
            nn.Linear(emb_dim, emb_dim),
            nn.SELU(),
            nn.AlphaDropout(p=0.1),
            nn.Linear(emb_dim, emb_dim),
            nn.SELU(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class GeneformerExpEncoder(nn.Module):
    def __init__(self, geneformer_dim, emb_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(geneformer_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.proj(x)


class STAG_EncoderAblation(nn.Module):

    def __init__(self, img_encoder_type='resnet18', exp_encoder_type='mlp',
                 uni_ckpt=None, conch_ckpt=None,
                 num_genes=250, emb_dim=512, depth1=2, num_heads1=8,
                 mlp_ratio1=2.0, dropout1=0.1, geneformer_dim=256):
        super().__init__()
        self.img_encoder_type = img_encoder_type
        self.exp_encoder_type = exp_encoder_type
        self.num_genes = num_genes
        self.emb_dim = emb_dim

        if img_encoder_type == 'resnet18':
            self.target_encoder = ResNet18Encoder()
        elif img_encoder_type == 'uni':
            assert uni_ckpt is not None, "UNI checkpoint path required (--uni_ckpt)"
            self.target_encoder = UNIEncoder(uni_ckpt)
        elif img_encoder_type == 'conch':
            assert conch_ckpt is not None, "CONCH checkpoint path required (--conch_ckpt)"
            self.target_encoder = ConchEncoder(conch_ckpt)
        else:
            raise ValueError(f"Unknown img_encoder: {img_encoder_type}")

        img_feat_dim = self.target_encoder.feat_dim
        print(f"Image encoder: {img_encoder_type}, feat_dim={img_feat_dim}")

        if exp_encoder_type == 'mlp':
            self.exp_encoder = MLPExpEncoder(num_genes, emb_dim)
        elif exp_encoder_type == 'snn':
            self.exp_encoder = SNNExpEncoder(num_genes, emb_dim)
        elif exp_encoder_type == 'geneformer':
            self.exp_encoder = GeneformerExpEncoder(geneformer_dim, emb_dim)
        else:
            raise ValueError(f"Unknown exp_encoder: {exp_encoder_type}")
        print(f"Expression encoder: {exp_encoder_type}")

        self.text_encoder = nn.Sequential(
            nn.Linear(768, emb_dim), nn.GELU(), nn.Dropout(0.1)
        )

        mlp_dim = int(emb_dim * mlp_ratio1)
        self.text_to_image_attn = TWOFusionEncoder(emb_dim, depth1, num_heads1, mlp_dim, dropout1)
        self.text_to_exp_attn = TWOFusionEncoder(emb_dim, depth1, num_heads1, mlp_dim, dropout1)

        self.neighbor_encoder = HGNN_Encoder(img_feat_dim, 1024, emb_dim)
        self.neighbor_exp_encoder = HGNN_Encoder(emb_dim, 1024, emb_dim)

        self.cross_encoder = TWOFusionEncoder(emb_dim, depth1, num_heads1, mlp_dim, dropout1)

        self.fc = nn.Linear(img_feat_dim, emb_dim)
        self.decoder = Decoder(input_dim=emb_dim, output_dim=num_genes)

    def build_graph(self, neighbor_nodes, neighbor_exp, k=3):
        num_nodes = neighbor_nodes.size(0)

        x = self.target_encoder(neighbor_nodes)
        x_exp = self.exp_encoder(neighbor_exp)

        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.T)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1)
        _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        topk_indices = topk_indices.long()

        hyperedge_indices = []
        for i in range(num_nodes):
            hyperedge_indices.extend([(node_idx.item(), i) for node_idx in topk_indices[i]])
            hyperedge_indices.append((i, i))
        if hyperedge_indices:
            rows, cols = zip(*hyperedge_indices)
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return x, x_exp, edge_index.long()

    def forward(self, x, exp, x_neighbor, x_neighbor_exp, text):
        B = x.shape[0] if x.dim() == 4 else x.squeeze(0).shape[0]
        x = x.squeeze()
        if x.dim() != 4:
            x = x.unsqueeze(0)

        x_feat = self.target_encoder(x)
        x_feat = self.fc(x_feat)
        x_feat = x_feat.unsqueeze(1)

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
            neighbor_nodes = torch.stack(n_list)
            neighbor_exp_nodes = torch.stack(ne_list)

        img_feat_dim = self.target_encoder.feat_dim
        neighbor_nodes = neighbor_nodes.view(B, -1, img_feat_dim).to(x.device)
        neighbor_exp_nodes = neighbor_exp_nodes.view(B, -1, self.emb_dim).to(x.device)
        edge_idx_list = [e.to(x.device) for e in e_list]

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
    features1 = F.normalize(features1, p=2, dim=-1)
    features2 = F.normalize(features2, p=2, dim=-1)
    batch_size = features1.shape[0]
    similarity_matrix = torch.mm(features1, features2.T) / temperature
    labels = torch.arange(batch_size).to(features1.device)
    positive_loss = F.cross_entropy(similarity_matrix, labels)
    negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=features1.device)
    negative_similarities = similarity_matrix[negative_mask].view(batch_size, -1)
    negative_loss = torch.logsumexp(negative_similarities, dim=-1).mean()
    return positive_loss + negative_weight * negative_loss


def evaluate_model(model, valloader, device, text_encoding, exp_encoder_type='mlp'):
    model.eval()
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(valloader, desc="  Eval", leave=False, ncols=80):
            if exp_encoder_type == 'geneformer':
                _, img, raw_exp, neighbor_img, _, _, gf_exp, gf_neighbor_exp = batch
                img = img.to(device, dtype=torch.float)
                raw_exp = raw_exp.to(device, dtype=torch.float)
                neighbor_img = neighbor_img.to(device, dtype=torch.float)
                gf_exp = gf_exp.to(device, dtype=torch.float)
                gf_neighbor_exp = gf_neighbor_exp.to(device, dtype=torch.float)
                text = text_encoding.to(device)
                _, _, _, _, pred, _ = model(img, gf_exp, neighbor_img, gf_neighbor_exp, text)
                all_outputs.append(pred.cpu().numpy())
                all_labels.append(raw_exp.cpu().numpy())
            else:
                _, img, exp, neighbor_img, neighbor_exp, _ = batch
                img = img.to(device, dtype=torch.float)
                exp = exp.to(device, dtype=torch.float)
                neighbor_img = neighbor_img.to(device, dtype=torch.float)
                neighbor_exp = neighbor_exp.to(device, dtype=torch.float)
                text = text_encoding.to(device)
                _, _, _, _, pred, _ = model(img, exp, neighbor_img, neighbor_exp, text)
                all_outputs.append(pred.cpu().numpy())
                all_labels.append(exp.cpu().numpy())

    all_outputs_np = np.concatenate(all_outputs, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    per_gene_pcc = []
    for g in range(all_labels_np.shape[1]):
        if np.std(all_labels_np[:, g]) > 1e-6 and np.std(all_outputs_np[:, g]) > 1e-6:
            pcc, _ = pearsonr(all_outputs_np[:, g], all_labels_np[:, g])
            if not np.isnan(pcc):
                per_gene_pcc.append(pcc)

    if not per_gene_pcc:
        return {f'pcc_{k}': 0 for k in [250, 200, 150, 100, 50, 20, 10]} | {'rmse': float('inf'), 'mse': float('inf'), 'mae': float('inf')}

    sorted_pcc = np.sort(np.array(per_gene_pcc))[::-1]
    mse = np.mean((all_outputs_np - all_labels_np) ** 2)

    metrics = {}
    for k in [250, 200, 150, 100, 50, 20, 10]:
        metrics[f'pcc_{k}'] = np.mean(sorted_pcc[:k]) if len(sorted_pcc) >= k else np.mean(sorted_pcc)
    metrics['rmse'] = np.sqrt(mse)
    metrics['mse'] = mse
    metrics['mae'] = np.mean(np.abs(all_outputs_np - all_labels_np))
    return metrics


def run_one_config(img_encoder, exp_encoder, device, args):
    config_name = f"{img_encoder}_{exp_encoder}"
    print(f"\n{'#'*60}")
    print(f"  Config: img={img_encoder}, exp={exp_encoder}")
    print(f"{'#'*60}")

    data_path = './data/GSE144240'
    text_path = 'select_genes/cSCC_loki_text_encode.npy'
    split_save_path = os.path.join(args.save_dir, 'cSCC_kfold_splits.json')

    _ = Original_cSCCDataset(path=data_path, mode='train', k_folds=args.k_folds,
                              fold_index=0, seed=args.seed, split_save_path=split_save_path)
    with open(split_save_path, 'r') as f:
        split_data = json.load(f)

    text_encoding = torch.from_numpy(np.load(text_path)).float()

    fold_results = []
    for fold, fold_info in enumerate(split_data['folds']):
        print(f"\n{'='*25} {config_name} - FOLD {fold+1}/{args.k_folds} {'='*25}")

        model = STAG_EncoderAblation(
            img_encoder_type=img_encoder,
            exp_encoder_type=exp_encoder,
            uni_ckpt=args.uni_ckpt,
            conch_ckpt=args.conch_ckpt,
            num_genes=250, emb_dim=512,
            geneformer_dim=args.geneformer_dim
        ).to(device)

        gf_dir = args.geneformer_emb_dir if exp_encoder == 'geneformer' else None
        train_dataset = Text_cSCC_Dataset(
            path=data_path, mode='train', k_folds=args.k_folds,
            fold_index=fold_info['fold_index'], seed=args.seed,
            split_save_path=split_save_path, text_path=text_path,
            geneformer_emb_dir=gf_dir
        )
        val_dataset = Text_cSCC_Dataset(
            path=data_path, mode='val', k_folds=args.k_folds,
            fold_index=fold_info['fold_index'], seed=args.seed,
            split_save_path=split_save_path, text_path=text_path,
            geneformer_emb_dir=gf_dir
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        loss_fn = nn.MSELoss()

        best_pcc = -1.0
        best_metrics = {}
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False, ncols=80):
                if exp_encoder == 'geneformer':
                    _, img, raw_exp, neighbor_img, _, _, gf_exp, gf_neighbor_exp = batch
                    img = img.to(device, dtype=torch.float)
                    raw_exp = raw_exp.to(device, dtype=torch.float)
                    gf_exp = gf_exp.to(device, dtype=torch.float)
                    gf_neighbor_exp = gf_neighbor_exp.to(device, dtype=torch.float)
                    neighbor_img = neighbor_img.to(device, dtype=torch.float)
                    text = text_encoding.to(device)
                    optimizer.zero_grad()
                    ft_img, ft_exp, fn_img, fn_exp, pred, _ = model(
                        img, gf_exp, neighbor_img, gf_neighbor_exp, text)
                    reconstruction_loss = loss_fn(pred, raw_exp)
                else:
                    _, img, exp, neighbor_img, neighbor_exp, _ = batch
                    img = img.to(device, dtype=torch.float)
                    exp = exp.to(device, dtype=torch.float)
                    neighbor_img = neighbor_img.to(device, dtype=torch.float)
                    neighbor_exp = neighbor_exp.to(device, dtype=torch.float)
                    text = text_encoding.to(device)
                    optimizer.zero_grad()
                    ft_img, ft_exp, fn_img, fn_exp, pred, _ = model(
                        img, exp, neighbor_img, neighbor_exp, text)
                    reconstruction_loss = loss_fn(pred, exp)
                cl_target = contrastive_loss(ft_img, ft_exp, args.temp1)
                cl_neighbor = contrastive_loss(fn_img, fn_exp, args.temp2)
                loss = reconstruction_loss + args.loss_ratio1 * cl_target + args.loss_ratio2 * cl_neighbor

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
                val_metrics = evaluate_model(model, val_loader, device, text_encoding, exp_encoder)
                current_pcc = val_metrics['pcc_250']
                print(f"  [{config_name}] Epoch {epoch+1}: loss={avg_loss:.4f}, pcc_250={current_pcc:.4f}")
                if current_pcc > best_pcc:
                    best_pcc = current_pcc
                    best_metrics = val_metrics.copy()
                    best_metrics['best_epoch'] = epoch + 1

        if best_metrics:
            best_metrics['fold'] = fold + 1
            best_metrics['config'] = config_name
            fold_results.append(best_metrics)
            print(f"  [{config_name}] Fold {fold+1} Best PCC_250: {best_pcc:.4f} (epoch {best_metrics['best_epoch']})")

    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Encoder Ablation on cSCC")
    parser.add_argument('--img_encoder', type=str, required=True, choices=['resnet18', 'uni', 'conch'])
    parser.add_argument('--exp_encoder', type=str, required=True, choices=['mlp', 'snn', 'geneformer'])
    parser.add_argument('--uni_ckpt', type=str, default=None, help="Path to UNI checkpoint (pytorch_model.bin)")
    parser.add_argument('--conch_ckpt', type=str, default=None, help="Path to CONCH checkpoint (pytorch_model.bin)")
    parser.add_argument('--geneformer_emb_dir', type=str, default=None,
                        help="Directory with pre-computed Geneformer embeddings (required when exp_encoder=geneformer)")
    parser.add_argument('--geneformer_dim', type=int, default=256,
                        help="Geneformer hidden dim (256 for 10M, 768 for 106M)")
    parser.add_argument('--k_folds', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1553)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--loss_ratio1', type=float, default=0.4)
    parser.add_argument('--loss_ratio2', type=float, default=0.2)
    parser.add_argument('--temp1', type=float, default=0.05)
    parser.add_argument('--temp2', type=float, default=0.05)
    parser.add_argument('--val_interval', type=int, default=5,
                        help='每隔多少个 epoch 验证一次 (默认: 5)')
    parser.add_argument('--save_dir', type=str, default='./ablation_encoder_results')
    args = parser.parse_args()

    if args.exp_encoder == 'geneformer' and not args.geneformer_emb_dir:
        parser.error("--geneformer_emb_dir is required when --exp_encoder=geneformer")

    config_name = f"{args.img_encoder}_{args.exp_encoder}"
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Encoder Ablation: img={args.img_encoder}, exp={args.exp_encoder}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"  Device: {device}")
    print("=" * 60)

    fold_results = run_one_config(args.img_encoder, args.exp_encoder, device, args)

    df = pd.DataFrame(fold_results)
    detail_path = os.path.join(args.save_dir, f'encoder_ablation_{config_name}_per_fold.csv')
    df.to_csv(detail_path, index=False, float_format='%.4f')
    print(f"\nPer-fold results saved to: {detail_path}")

    metric_cols = ['pcc_250', 'pcc_200', 'pcc_150', 'pcc_100', 'pcc_50', 'pcc_20', 'pcc_10', 'rmse', 'mse', 'mae']
    row = {'config': config_name}
    for col in metric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        row[f'{col}_mean'] = mean_val
        row[f'{col}_std'] = std_val
        row[col] = f"{mean_val:.4f}+/-{std_val:.4f}"

    summary_df = pd.DataFrame([row])
    summary_path = os.path.join(args.save_dir, f'encoder_ablation_{config_name}_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    print(f"\n{'='*70}")
    print(f"Encoder Ablation Summary — {config_name} (cSCC)")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'PCC_250':<18} {'PCC_50':<18} {'PCC_10':<18} {'RMSE':<18}")
    print(f"-" * 74)
    print(f"{row['config']:<20} {row['pcc_250']:<18} {row['pcc_50']:<18} {row['pcc_10']:<18} {row['rmse']:<18}")
    print("=" * 74)


if __name__ == '__main__':
    main()
