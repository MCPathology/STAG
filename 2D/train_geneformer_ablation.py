
import torch
import torch.nn as nn
import sys
import os
import argparse
import numpy as np
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter
import time
import json
from scipy.stats import pearsonr
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.models.model_Hypergraph_Text import STAG, contrastive_loss
from dataset.TextcSCCDataset import cSCCDataset as Text_cSCC_Dataset
from dataset.cSCCDataset import cSCCDataset as Original_cSCCDataset

import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Geneformer vs OmiCLIP text encoder ablation")

    parser.add_argument('--model_name', type=str, default='STAG')
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--num_neighbors', type=int, default=9)

    parser.add_argument('--data_name', type=str, default='cSCC', choices=['cSCC'])
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1553)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--loss_ratio1', type=float, default=0.4)
    parser.add_argument('--loss_ratio2', type=float, default=0.2)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--temp1', type=float, default=0.05)
    parser.add_argument('--temp2', type=float, default=0.05)
    parser.add_argument('--select_fold', type=int, default=-1, help="-1=all folds")
    parser.add_argument('--neighbors', type=int, default=9)

    parser.add_argument('--text_encoder', type=str, default='geneformer',
                        choices=['omiclip', 'bert', 'geneformer'],
                        help="Text encoder: omiclip (loki), bert, or geneformer")

    return parser.parse_args()


DATA_PATHS = {
    'cSCC': './data/GSE144240',
}

TEXT_ENCODE_FILES = {
    'omiclip': 'select_genes/cSCC_loki_text_encode.npy',
    'bert': 'select_genes/cSCC_bert_text_encode.npy',
    'geneformer': 'select_genes/cSCC_geneformer_text_encode.npy',
}


class Tee(object):
    def __init__(self, filename, mode="w", encoding='utf-8'):
        self.file = open(filename, mode, encoding=encoding)
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, data): self.file.write(data); self.stdout.write(data)
    def flush(self): self.file.flush()
    def close(self): sys.stdout = self.stdout; self.file.close()


def save_results_to_csv(results, output_path):
    df = pd.DataFrame(results)
    mean_row, std_row = df.mean(), df.std()
    mean_row.name, std_row.name = 'mean', 'std'
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])])
    df.loc[['mean', 'std'], 'fold'] = ''
    df.to_csv(output_path, index=True, float_format='%.4f')
    print(f"\nK-Fold results saved to: {output_path}")


def evaluate_model(model, valloader, device):
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for data in tqdm(valloader, desc="  - Eval", leave=False, ncols=80):
            _, img_tensor, true_expression, hypergraph_x, hypergraph_x_exp, text = data
            img_tensor = img_tensor.to(device, dtype=torch.float)
            true_expression = true_expression.to(device, dtype=torch.float)
            hypergraph_x = hypergraph_x.to(device, dtype=torch.float)
            hypergraph_x_exp = hypergraph_x_exp.to(device, dtype=torch.float)
            text = text[0].to(device, dtype=torch.float)

            outputs = model(x=img_tensor, exp=true_expression,
                            x_neighbor=hypergraph_x, x_neighbor_exp=hypergraph_x_exp, text=text)
            pred_exp = outputs[4]
            pred_np = pred_exp.cpu().numpy()
            if pred_np.ndim != 2:
                continue
            all_predictions.append(pred_np)
            all_true_labels.append(true_expression.cpu().numpy())

    all_preds_np = np.concatenate(all_predictions, axis=0)
    all_gts_np = np.concatenate(all_true_labels, axis=0)

    per_gene_pcc = []
    for g in range(all_gts_np.shape[1]):
        if np.std(all_gts_np[:, g]) > 1e-6 and np.std(all_preds_np[:, g]) > 1e-6:
            pcc, _ = pearsonr(all_preds_np[:, g], all_gts_np[:, g])
            if not np.isnan(pcc):
                per_gene_pcc.append(pcc)

    if not per_gene_pcc:
        return {'pcc_all': 0, 'pcc_top10': 0, 'pcc_top50': 0, 'pcc_top100': 0,
                'rmse': float('inf'), 'mae': float('inf')}

    sorted_pcc = np.sort(np.array(per_gene_pcc))[::-1]
    mse = np.mean((all_preds_np - all_gts_np) ** 2)

    return {
        'pcc_all': np.mean(sorted_pcc),
        'pcc_top10': np.mean(sorted_pcc[:10]) if len(sorted_pcc) >= 10 else np.mean(sorted_pcc),
        'pcc_top50': np.mean(sorted_pcc[:50]) if len(sorted_pcc) >= 50 else np.mean(sorted_pcc),
        'pcc_top100': np.mean(sorted_pcc[:100]) if len(sorted_pcc) >= 100 else np.mean(sorted_pcc),
        'rmse': np.sqrt(mse),
        'mae': np.mean(np.abs(all_preds_np - all_gts_np)),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    text_file = TEXT_ENCODE_FILES[args.text_encoder]
    if not os.path.exists(text_file):
        raise FileNotFoundError(
            f"Text encoding file not found: {text_file}\n"
            f"For geneformer, run: python preprocess/extract_geneformer_embeddings.py"
        )

    log_dir = f'./logs/{args.model_name}_ablation_textenc/{args.text_encoder}/{run_timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    tee = Tee(os.path.join(log_dir, 'training_log.txt'))

    print("=" * 60)
    print(f"Text Encoder Ablation: {args.text_encoder}")
    print(f"Text encoding file: {text_file}")
    for key, value in vars(args).items():
        print(f"{key:<15}: {value}")
    print(f"{'device':<15}: {device}")
    print("=" * 60)

    data_path = DATA_PATHS[args.data_name]
    split_save_path = os.path.join(log_dir, f"{args.data_name}_kfold_splits.json")
    _ = Original_cSCCDataset(path=data_path, mode='train', k_folds=args.k_folds,
                              fold_index=0, seed=args.seed, split_save_path=split_save_path)
    with open(split_save_path, 'r') as f:
        split_data = json.load(f)

    kfold_split_plans = []
    for fold_info in split_data['folds']:
        kfold_split_plans.append({
            'data_name': args.data_name,
            'data_path': data_path,
            'fold_index': fold_info['fold_index']
        })

    num_genes = 250
    fold_results = []

    for fold, plan in enumerate(kfold_split_plans):
        if args.select_fold >= 0 and fold != args.select_fold:
            continue

        print(f"\n{'='*25} FOLD {fold + 1}/{len(kfold_split_plans)} {'='*25}")
        print(f"Text encoder: {args.text_encoder}")

        model = STAG(
            num_genes=num_genes, emb_dim=args.emb_dim,
            depth1=args.depth, num_heads1=args.heads, ablation_mode='full'
        ).to(device)

        dataset_kwargs = {
            'path': plan['data_path'],
            'k_folds': args.k_folds,
            'fold_index': plan['fold_index'],
            'seed': args.seed,
            'split_save_path': split_save_path,
            'text_path': text_file,
        }

        train_dataset = Text_cSCC_Dataset(mode='train', **dataset_kwargs)
        val_dataset = Text_cSCC_Dataset(mode='val', **dataset_kwargs)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        mse_loss_fn = torch.nn.MSELoss()
        writer = SummaryWriter(log_dir=os.path.join(log_dir, f'fold_{fold+1}'))

        best_fold_pcc = -1.0
        best_fold_metrics = {}

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False, ncols=80):
                spot_info, img_tensor, true_expression, hypergraph_x, hypergraph_x_exp, text = data
                img_tensor = img_tensor.to(device, dtype=torch.float)
                true_expression = true_expression.to(device, dtype=torch.float)
                hypergraph_x = hypergraph_x.to(device, dtype=torch.float)
                hypergraph_x_exp = hypergraph_x_exp.to(device, dtype=torch.float)
                text = text[0].to(device, dtype=torch.float)

                optimizer.zero_grad()
                outputs = model(x=img_tensor, exp=true_expression,
                                x_neighbor=hypergraph_x, x_neighbor_exp=hypergraph_x_exp, text=text)

                fused_target_img, fused_target_exp, fused_neighbor_img, fused_neighbor_exp, pred_exp, _ = outputs
                loss_patch = 0.5 * (contrastive_loss(fused_target_img, fused_target_exp, args.temp1) +
                                    contrastive_loss(fused_target_exp, fused_target_img, args.temp1))
                loss_neighbor = 0.5 * (contrastive_loss(fused_neighbor_img, fused_neighbor_exp, args.temp2) +
                                       contrastive_loss(fused_neighbor_exp, fused_neighbor_img, args.temp2))
                reconstruction_loss = mse_loss_fn(pred_exp, true_expression)
                loss = args.loss_ratio1 * loss_patch + args.loss_ratio2 * loss_neighbor + args.recon_weight * reconstruction_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"\n--- [Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} ---")

            val_metrics = evaluate_model(model, val_loader, device)
            print("   --- Validation ---")
            for key, value in val_metrics.items():
                print(f"   {key:<12}: {value:.4f}")
                writer.add_scalar(f'Validation/{key}', value, epoch)

            writer.add_scalar('Loss/Train', avg_loss, epoch)

            if val_metrics['pcc_all'] > best_fold_pcc:
                best_fold_pcc = val_metrics['pcc_all']
                best_fold_metrics = val_metrics
                print(f"   Best PCC: {best_fold_pcc:.4f}! Model saved.")
                torch.save(model.state_dict(), os.path.join(log_dir, f'best_fold_{fold+1}.pth'))

        if best_fold_metrics:
            best_fold_metrics['fold'] = fold + 1
            best_fold_metrics['text_encoder'] = args.text_encoder
            fold_results.append(best_fold_metrics)
        writer.close()

    if fold_results:
        csv_path = os.path.join(log_dir, f'kfold_summary_{args.text_encoder}.csv')
        save_results_to_csv(fold_results, csv_path)
        df = pd.DataFrame(fold_results)
        print(f"\n{'='*60}")
        print(f"Text Encoder: {args.text_encoder}")
        print(f"PCC@250: {df['pcc_all'].mean():.4f} +/- {df['pcc_all'].std():.4f}")
        print(f"PCC@50:  {df['pcc_top50'].mean():.4f} +/- {df['pcc_top50'].std():.4f}")
        print(f"RMSE:    {df['rmse'].mean():.4f} +/- {df['rmse'].std():.4f}")
        print(f"MAE:     {df['mae'].mean():.4f} +/- {df['mae'].std():.4f}")
        print(f"{'='*60}")

    tee.close()


if __name__ == '__main__':
    main()
