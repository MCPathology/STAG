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
from dataset.TextHER2Dataset import HER2Dataset as Text_HER2_Dataset
from dataset.TextHBCDataset import HBCDataset as Text_HBC_Dataset
from dataset.TextHESTDataset import HESTDataset as Text_HEST_Dataset, get_full_kfold_splits

from dataset.cSCCDataset import cSCCDataset as Original_cSCCDataset
from dataset.HER2Dataset import HER2Dataset as Original_HER2Dataset
from dataset.HBCDataset import HBCDataset as Original_HBCDataset

import torch.nn.functional as F


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="STAG-TextGuided HVG 预测训练脚本")

    parser.add_argument('--model_name', type=str, default='STAG', choices=['STAG'])
    parser.add_argument('--emb_dim', type=int, default=512, help="模型内部嵌入维度")
    parser.add_argument('--depth', type=int, default=2, help="交叉注意力编码器深度")
    parser.add_argument('--heads', type=int, default=8, help="交叉注意力编码器头数")
    parser.add_argument('--num_neighbors', type=int, default=9, help="邻居数量")

    parser.add_argument('--data_name', type=str, required=True,
                        choices=['cSCC', 'HER2', 'HBC'],
                        help="数据集名称 (HVG模式仅支持 cSCC/HER2/HBC)")

    parser.add_argument('--k_folds', type=int, default=5, help="K折交叉验证折数")
    parser.add_argument('--epochs', type=int, default=50, help="每折训练轮数")
    parser.add_argument('--lr', type=float, default=1e-4, help="学习率")
    parser.add_argument('--batch_size', type=int, default=8, help="批处理大小")
    parser.add_argument('--seed', type=int, default=1553, help="随机种子")
    parser.add_argument('--num_workers', type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument('--save_checkpoints', action='store_true', help="Save best-fold model weights.")
    parser.add_argument('--save_tensorboard', action='store_true', help="Save TensorBoard event files.")
    parser.add_argument('--save_logs', action='store_true', help="Save full stdout training logs.")

    parser.add_argument('--loss_ratio1', type=float, default=0.4, help="目标Spot对比损失权重")
    parser.add_argument('--loss_ratio2', type=float, default=0.2, help="邻域对比损失权重")
    parser.add_argument('--recon_weight', type=float, default=1.0, help="重建损失权重")
    parser.add_argument('--temp1', type=float, default=0.05, help="目标Spot对比损失温度")
    parser.add_argument('--temp2', type=float, default=0.05, help="邻域对比损失温度")
    parser.add_argument('--select_fold', type=int, default=-1, help="指定运行的折数 (-1=全部)")
    parser.add_argument('--neighbors', type=int, default=9, help="邻居数量")
    parser.add_argument('--gene_mode', type=str, default='hvg', choices=['default', 'hvg'],
                        help="基因选择模式: default=top-expression, hvg=highly-variable-genes")

    parser.add_argument('--weight_decay', type=float, default=1e-5, help="AdamW 权重衰减")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="学习率预热轮数")
    parser.add_argument('--patience', type=int, default=15, help="早停耐心值 (0=不使用早停)")
    parser.add_argument('--train_seed', type=int, default=42, help="训练随机种子 (模型初始化/shuffle，不影响fold划分)")

    return parser.parse_args()

DATA_PATHS = {
    'HBC': './data/Human_breast_cancer_in_situ_capturing_transcriptomics/',
    'cSCC': './data/GSE144240',
    'HER2': './data/HER2/',
}

ORIGINAL_DATASET_CLASSES = {
    'cSCC': Original_cSCCDataset,
    'HER2': Original_HER2Dataset,
    'HBC': Original_HBCDataset,
}
TEXT_DATASET_CLASSES = {
    'cSCC': Text_cSCC_Dataset,
    'HER2': Text_HER2_Dataset,
    'HBC': Text_HBC_Dataset,
}

GENE_FILES = {
    'cSCC': 'select_genes/cSCC_Selected_Genes.npy',
    'HER2': 'select_genes/HER2_Selected_Genes.npy',
    'HBC': 'select_genes/HBC_Selected_Genes.npy',
}

TEXT_FILES = {
    'cSCC': 'select_genes/cSCC_bert_text_encode.npy',
    'HER2': 'select_genes/HER2_loki_text_encode.npy',
    'HBC': 'select_genes/STNet_loki_text_encode.npy',
}

HVG_GENE_FILES = {
    'cSCC': 'select_genes/cSCC_HVG_Genes.npy',
    'HER2': 'select_genes/HER2_HVG_Genes.npy',
    'HBC': 'select_genes/HBC_HVG_Genes.npy',
}

HVG_TEXT_FILES = {
    'cSCC': 'select_genes/cSCC_HVG_loki_text_encode.npy',
    'HER2': 'select_genes/HER2_HVG_loki_text_encode.npy',
    'HBC': 'select_genes/HBC_HVG_loki_text_encode.npy',
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
    print(f"\nK-Fold 评估结果已汇总并保存到: {output_path}")


def evaluate_model(model, valloader, device):
    print("   开始执行端到端评估...")
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for data in tqdm(valloader, desc="  - 验证中", leave=False, ncols=80):
            _, img_tensor, true_expression, hypergraph_x, hypergraph_x_exp, text = data

            img_tensor = img_tensor.to(device, dtype=torch.float)
            true_expression = true_expression.to(device, dtype=torch.float)
            hypergraph_x = hypergraph_x.to(device, dtype=torch.float)
            hypergraph_x_exp = hypergraph_x_exp.to(device, dtype=torch.float)
            text = text[0].to(device, dtype=torch.float)

            outputs = model(
                x=img_tensor,
                exp=true_expression,
                x_neighbor=hypergraph_x,
                x_neighbor_exp=hypergraph_x_exp,
                text=text
            )

            pred_exp = outputs[4]
            pred_np = pred_exp.cpu().numpy()
            if pred_np.ndim != 2:
                continue
            true_np = true_expression.cpu().numpy()

            all_predictions.append(pred_np)
            all_true_labels.append(true_np)

    print("   计算评估指标...")
    all_preds_np = np.concatenate(all_predictions, axis=0)
    all_gts_np = np.concatenate(all_true_labels, axis=0)

    per_gene_pcc = []
    for g in range(all_gts_np.shape[1]):
        if np.std(all_gts_np[:, g]) > 1e-6 and np.std(all_preds_np[:, g]) > 1e-6:
            pcc, _ = pearsonr(all_preds_np[:, g], all_gts_np[:, g])
            if not np.isnan(pcc):
                per_gene_pcc.append(pcc)

    if not per_gene_pcc:
        return {
            'pcc_all': 0, 'pcc_top10': 0, 'pcc_top20': 0, 'pcc_top50': 0, 'pcc_top100': 0,
            'rmse': float('inf'), 'mse': float('inf'), 'mae': float('inf')
        }

    sorted_pcc = np.sort(np.array(per_gene_pcc))[::-1]

    mse = np.mean((all_preds_np - all_gts_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds_np - all_gts_np))

    metrics = {
        'pcc_all': np.mean(sorted_pcc),
        'pcc_top10': np.mean(sorted_pcc[:10]) if len(sorted_pcc) >= 10 else np.mean(sorted_pcc),
        'pcc_top20': np.mean(sorted_pcc[:20]) if len(sorted_pcc) >= 20 else np.mean(sorted_pcc),
        'pcc_top50': np.mean(sorted_pcc[:50]) if len(sorted_pcc) >= 50 else np.mean(sorted_pcc),
        'pcc_top100': np.mean(sorted_pcc[:100]) if len(sorted_pcc) >= 100 else np.mean(sorted_pcc),
        'rmse': rmse,
        'mse': mse,
        'mae': mae,
    }

    return metrics

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    gene_mode_tag = f"_{args.gene_mode}" if args.gene_mode != 'default' else ""
    log_dir = f'./logs/{args.model_name}{gene_mode_tag}/{args.data_name}/{run_timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    tee = Tee(os.path.join(log_dir, 'training_log.txt')) if args.save_logs else None

    print("=" * 60)
    print(f"K-Fold 训练脚本 for {args.model_name} (gene_mode={args.gene_mode})")
    for key, value in vars(args).items(): print(f"{key:<15}: {value}")
    print(f"{'device':<15}: {device}")
    print("=" * 60)

    if args.gene_mode == 'hvg':
        gene_file = HVG_GENE_FILES.get(args.data_name)
        text_file = HVG_TEXT_FILES.get(args.data_name)
        if not gene_file:
            raise ValueError(f"未找到 {args.data_name} 的 HVG 基因文件！HVG模式仅支持: {list(HVG_GENE_FILES.keys())}")
        if not text_file:
            raise ValueError(f"未找到 {args.data_name} 的 HVG 文本编码文件！")
        print(f"--- [HVG模式] 基因列表: {gene_file} ---")
        print(f"--- [HVG模式] 文本编码: {text_file} ---")
    else:
        gene_file = GENE_FILES.get(args.data_name)
        text_file = TEXT_FILES.get(args.data_name)
        print(f"--- [Default模式] 基因列表: {gene_file} ---")
        print(f"--- [Default模式] 文本编码: {text_file} ---")

    selected_genes = np.load(gene_file, allow_pickle=True).tolist()
    num_genes = len(selected_genes)
    print(f"--- 基因数量: {num_genes} ---")

    kfold_split_plans = []
    split_save_path = os.path.join(log_dir, f"{args.data_name}_kfold_splits.json")

    data_path = DATA_PATHS[args.data_name]
    DatasetClassForSplit = ORIGINAL_DATASET_CLASSES[args.data_name]
    _ = DatasetClassForSplit(path=data_path, mode='train', k_folds=args.k_folds,
                             fold_index=0, seed=args.seed, split_save_path=split_save_path)
    with open(split_save_path, 'r') as f:
        split_data = json.load(f)
    for fold_info in split_data['folds']:
        kfold_split_plans.append({
            'data_name': args.data_name,
            'data_path': data_path,
            'fold_index': fold_info['fold_index']
        })
    print(f"--- K-Fold 划分方案已加载/生成于: {split_save_path} ---")

    fold_results = []
    for fold, plan in enumerate(kfold_split_plans):
        if args.select_fold >= 0 and fold != args.select_fold:
            continue

        print(f"\n{'='*25} FOLD {fold + 1}/{len(kfold_split_plans)} {'='*25}")
        print(f"--- 基因数量: {num_genes}, 基因模式: {args.gene_mode}, train_seed: {args.train_seed} ---")

        fold_seed = args.train_seed + fold
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
        np.random.seed(fold_seed)
        import random as _random
        _random.seed(fold_seed)

        model = STAG(
            num_genes=num_genes,
            emb_dim=args.emb_dim,
            depth1=args.depth,
            num_heads1=args.heads,
            ablation_mode='full'
        ).to(device)

        print("\n--- 正在创建数据集 (这可能需要一些时间)... ---")
        DatasetClass = TEXT_DATASET_CLASSES[plan['data_name']]

        dataset_kwargs = {
            'path': plan['data_path'],
            'k_folds': args.k_folds,
            'fold_index': plan['fold_index'],
            'seed': args.seed,
            'split_save_path': split_save_path,
        }

        if args.gene_mode == 'hvg':
            dataset_kwargs['selected_genes'] = selected_genes
            dataset_kwargs['text_path'] = text_file
            print(f"--- [HVG模式] 传入 selected_genes (len={len(selected_genes)}) 和 text_path ---")

        train_dataset = DatasetClass(mode='train', **dataset_kwargs)
        val_dataset = DatasetClass(mode='val', **dataset_kwargs)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        total_steps = args.epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        mse_loss_fn = torch.nn.MSELoss()
        writer = SummaryWriter(log_dir=os.path.join(log_dir, f'fold_{fold+1}')) if args.save_tensorboard else NullSummaryWriter()

        best_fold_pcc = -1.0
        best_fold_metrics = {}
        no_improve_count = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", leave=False, ncols=80):
                spot_info, img_tensor, true_expression, hypergraph_x, hypergraph_x_exp, text = data

                img_tensor = img_tensor.to(device, dtype=torch.float)
                true_expression = true_expression.to(device, dtype=torch.float)
                hypergraph_x = hypergraph_x.to(device, dtype=torch.float)
                hypergraph_x_exp = hypergraph_x_exp.to(device, dtype=torch.float)
                text = text[0].to(device, dtype=torch.float)

                optimizer.zero_grad()

                outputs = model(
                    x=img_tensor,
                    exp=true_expression,
                    x_neighbor=hypergraph_x,
                    x_neighbor_exp=hypergraph_x_exp,
                    text=text
                )

                fused_target_img, fused_target_exp, fused_neighbor_img, fused_neighbor_exp, pred_exp, _ = outputs
                loss_patch = 0.5 * (
                    contrastive_loss(fused_target_img, fused_target_exp, args.temp1) +
                    contrastive_loss(fused_target_exp, fused_target_img, args.temp1)
                )
                loss_neighbor = 0.5 * (
                    contrastive_loss(fused_neighbor_img, fused_neighbor_exp, args.temp2) +
                    contrastive_loss(fused_neighbor_exp, fused_neighbor_img, args.temp2)
                )
                reconstruction_loss = mse_loss_fn(pred_exp, true_expression)

                loss = args.loss_ratio1 * loss_patch + args.loss_ratio2 * loss_neighbor + args.recon_weight * reconstruction_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n--- [epoch {epoch+1}/{args.epochs}] 结束 (lr={current_lr:.6f}) ---")
            print(f"   平均总损失: {avg_train_loss:.4f}")

            val_metrics = evaluate_model(model, val_loader, device)

            print("   --- 验证结果 ---")
            for key, value in val_metrics.items():
                print(f"   {key:<12}: {value:.4f}")
                writer.add_scalar(f'Validation/{key}', value, epoch)
            print("   -----------------------------")

            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)

            if val_metrics['pcc_all'] > best_fold_pcc:
                best_fold_pcc = val_metrics['pcc_all']
                best_fold_metrics = val_metrics
                no_improve_count = 0
                print(f"   Best PCC updated: {best_fold_pcc:.4f}.")
                if args.save_checkpoints:
                    torch.save(model.state_dict(), os.path.join(log_dir, f'best_fold_{fold+1}.pth'))
            else:
                no_improve_count += 1
                if args.patience > 0 and no_improve_count >= args.patience:
                    print(f"   早停: 连续 {args.patience} 轮无提升，停止训练。")
                    break

        if best_fold_metrics:
            best_fold_metrics['fold'] = fold + 1
            fold_results.append(best_fold_metrics)
        writer.close()

    if fold_results:
        csv_results_path = os.path.join(log_dir, f'kfold_summary_{args.gene_mode}.csv')
        save_results_to_csv(fold_results, csv_results_path)
        df_summary = pd.DataFrame(fold_results)
        mean_pcc = df_summary['pcc_all'].mean()
        std_pcc = df_summary['pcc_all'].std()
        print("\n--- 最终平均性能 ---")
        print(f"基因模式: {args.gene_mode}")
        print(f"平均 PCC (所有基因): {mean_pcc:.4f} +/- {std_pcc:.4f}")
        print("=" * 56)

    if tee is not None:
        tee.close()

if __name__ == '__main__':
    main()
