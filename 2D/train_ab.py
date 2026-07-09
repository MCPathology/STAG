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
    parser = argparse.ArgumentParser(description="STAG-TextGuided 模型通用 K-Fold 训练脚本")
    
    parser.add_argument('--model_name', type=str, default='STAG', choices=['STAG'], help="模型名称")
    parser.add_argument('--emb_dim', type=int, default=512, help="模型内部嵌入维度")
    parser.add_argument('--depth', type=int, default=2, help="交叉注意力编码器深度")
    parser.add_argument('--heads', type=int, default=8, help="交叉注意力编码器头数")
    parser.add_argument('--num_neighbors', type=int, default=9, help="邻居数量")

    parser.add_argument('--data_name', type=str, required=True, 
                        choices=['cSCC', 'HER2', 'HBC', 'HEST_LUAD', 'HEST_IDC', 'HEST_PAAD', 
                                 'HEST_SKCM', 'HEST_her2st', 'HEST_kidney', 'HEST_mouse_brain', 
                                 'HEST_PRAD', 'HEST_Liver', 'HEST_Lung'],
                        help="数据集名称")
    
    parser.add_argument('--k_folds', type=int, default=5, help="K折交叉验证折数")
    parser.add_argument('--epochs', type=int, default=20, help="每折训练轮数")
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
    parser.add_argument('--select_fold', type=int, default=0, help="邻域对比损失温度")
    parser.add_argument('--ablation_mode', type=str, default='full',
                        choices=['full', 'query_no_ca_cl', 'query_with_ca', 'query_with_cl', 'query_with_ca_cl',
                                 'full_no_ca_cl', 'full_with_ca', 'full_with_cl', 'full_with_ca_cl'],
                        help="消融实验模式")
    parser.add_argument('--gene_mode', type=str, default='default', choices=['default', 'hvg'],
                        help="基因选择模式: default=top-expression, hvg=highly-variable-genes")


    return parser.parse_args()

DATA_PATHS = {
    'HBC': './data/Human_breast_cancer_in_situ_capturing_transcriptomics/',
    'cSCC': './data/GSE144240',
    'HER2': './data/HER2/',
    'HEST_BASE': './data/Hest1k_datasets/'
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
    'HEST': Text_HEST_Dataset,
}

GENE_FILES = {
    'cSCC': 'select_genes/cSCC_Selected_Genes.npy',
    'HER2': 'select_genes/HER2_Selected_Genes.npy',
    'HBC': 'select_genes/HBC_Selected_Genes.npy',
    'HEST_LUAD': 'select_genes/HEST_LUNG_gene.npy',
    'HEST_kidney': 'select_genes/HEST_KIDNEY_gene.npy',
    'HEST_mouse_brain': 'select_genes/HEST_MOUSE_BRAIN_gene.npy',
    'HEST_PRAD': 'select_genes/HEST_PRAD_gene.npy',
}

TEXT_FILES = {
    'cSCC': 'select_genes/cSCC_loki_text_encode.npy',
    'HER2': 'select_genes/HER2_loki_text_encode.npy',
    'HBC': 'select_genes/STNet_loki_text_encode.npy',
    'HEST_LUAD': 'select_genes/HEST_LUNG_loki_text_encode.npy',
    'HEST_kidney': 'select_genes/HEST_KIDNEY_loki_text_encode.npy',
    'HEST_mouse_brain': 'select_genes/HEST_MOUSE_BRAIN_loki_text_encode.npy',
    'HEST_PRAD': 'select_genes/HEST_PRAD_loki_text_encode.npy',
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
    print(f"\n✅ K-Fold 评估结果已汇总并保存到: {output_path}")


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
            preds_np = pred_exp.cpu().numpy()
            gts_np = true_expression.cpu().numpy()

            if preds_np.ndim == 1:
                preds_np = preds_np.reshape(1, -1)
            if gts_np.ndim == 1:
                gts_np = gts_np.reshape(1, -1)
            
            all_predictions.append(preds_np)
            all_true_labels.append(gts_np)
            
    print("   计算评估指标...")
    all_preds_np = np.concatenate(all_predictions, axis=0)
    all_gts_np = np.concatenate(all_true_labels, axis=0)
    
    per_gene_pcc = []
    for g in range(all_gts_np.shape[1]):
        pcc, _ = pearsonr(all_preds_np[:, g], all_gts_np[:, g])
        if not np.isnan(pcc):
            per_gene_pcc.append(pcc)
        else:
            per_gene_pcc.append(0)
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
    gene_mode_suffix = '_HVG' if args.gene_mode == 'hvg' else ''
    log_dir = f'./logs/{args.model_name}/{args.data_name}{gene_mode_suffix}/{run_timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    tee = Tee(os.path.join(log_dir, 'training_log.txt')) if args.save_logs else None
    
    print("=" * 60)
    print(f"K-Fold 训练脚本 for {args.model_name}")
    for key, value in vars(args).items(): print(f"{key:<15}: {value}")
    print(f"{'device':<15}: {device}")
    print("=" * 60)

    kfold_split_plans = [] 
    split_save_path = os.path.join(log_dir, f"{args.data_name}_kfold_splits.json")
    
    is_hest = args.data_name.startswith('HEST_')
    
    if is_hest:
        print(f"--- 检测到 HEST 数据集: {args.data_name} ---")
        hest_subset_name = args.data_name.replace('HEST_', '')
        data_path = os.path.join(DATA_PATHS['HEST_BASE'], hest_subset_name) if hest_subset_name in ['her2st', 'kidney', 'mouse_brain', 'PRAD', 'Liver', 'Lung'] else os.path.join(DATA_PATHS['HEST_BASE'], f'hest_data_{hest_subset_name.upper()}')
        print(f"--- 数据路径: {data_path} ---")

        kfold_splits_from_func = get_full_kfold_splits(data_path, n_splits=args.k_folds, random_state=args.seed)
        
        json_output = {"k_folds": args.k_folds, "seed": args.seed, "folds": []}
        for fold_data in kfold_splits_from_func:
            json_output["folds"].append({"fold_index": fold_data['fold'], "train_files": fold_data['train'], "val_files": fold_data['val']})
            plan = {'data_name': args.data_name, 'data_path': data_path, 'fold_index': fold_data['fold']}
            kfold_split_plans.append(plan)
        with open(split_save_path, 'w') as f: json.dump(json_output, f, indent=4)
        print(f"--- HEST K-Fold 划分方案已保存至: {split_save_path} ---")

    else:
        print(f"--- 检测到标准数据集: {args.data_name} ---")
        data_path = DATA_PATHS[args.data_name]
        DatasetClassForSplit = ORIGINAL_DATASET_CLASSES[args.data_name]
        _ = DatasetClassForSplit(path=data_path, mode='train', k_folds=args.k_folds, fold_index=0, seed=args.seed, split_save_path=split_save_path)
        with open(split_save_path, 'r') as f: split_data = json.load(f)
        for fold_info in split_data['folds']:
            kfold_split_plans.append({'data_name': args.data_name, 'data_path': data_path, 'fold_index': fold_info['fold_index']})
        print(f"--- 标准数据集 K-Fold 划分方案已加载/生成于: {split_save_path} ---")
    
    
    fold_results = []
    for fold, plan in enumerate(kfold_split_plans):
        print(f"\n{'='*25} FOLD {fold + 1}/{len(kfold_split_plans)} {'='*25}")
        
        num_genes = 250
        print(f"--- 检测到数据集 '{args.data_name}' 的基因数量为: {num_genes} ---")

        model = STAG(num_genes=num_genes, emb_dim=args.emb_dim, depth1=args.depth, num_heads1=args.heads,ablation_mode=args.ablation_mode).to(device)
        
        print("\n--- 正在创建数据集 (这可能需要一些时间)... ---")
        DatasetClass = TEXT_DATASET_CLASSES['HEST' if is_hest else plan['data_name']]
        if is_hest:
            with open(split_save_path, 'r') as f:
                split_data = json.load(f)
            
            current_fold_info = split_data['folds'][plan['fold_index']]
            
            gene_file = GENE_FILES.get(args.data_name)
            text_file = TEXT_FILES.get(args.data_name)
            if not gene_file or not text_file:
                raise ValueError(f"未在配置中找到 {args.data_name} 的基因或文本文件路径！")

            train_dataset = DatasetClass(
                data_root_path=plan['data_path'],
                mode='train',
                specific_subsets=current_fold_info['train_files'],
                selected_genes_file_path=gene_file,
                text_encoding_file_path=text_file,
                patch_size=224
            )
            val_dataset = DatasetClass(
                data_root_path=plan['data_path'],
                mode='val',
                specific_subsets=current_fold_info['val_files'],
                selected_genes_file_path=gene_file,
                text_encoding_file_path=text_file,
                patch_size=224
            )
        else:
            dataset_kwargs = {
                'path': plan['data_path'],
                'k_folds': args.k_folds,
                'fold_index': plan['fold_index'],
                'seed': args.seed,
                'split_save_path': split_save_path,
            }
            if args.gene_mode == 'hvg' and plan['data_name'] in HVG_GENE_FILES:
                import numpy as _np
                hvg_genes = _np.load(HVG_GENE_FILES[plan['data_name']], allow_pickle=True).tolist()
                hvg_text = HVG_TEXT_FILES[plan['data_name']]
                dataset_kwargs['selected_genes'] = hvg_genes
                dataset_kwargs['text_path'] = hvg_text
                print(f"--- [HVG模式] 使用 HVG 基因列表: {HVG_GENE_FILES[plan['data_name']]} ---")
                print(f"--- [HVG模式] 使用 HVG text encoding: {hvg_text} ---")
            train_dataset = DatasetClass(mode='train', **dataset_kwargs)
            val_dataset = DatasetClass(mode='val', **dataset_kwargs)
        
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        mse_loss_fn = torch.nn.MSELoss()
        writer = SummaryWriter(log_dir=os.path.join(log_dir, f'fold_{fold+1}')) if args.save_tensorboard else NullSummaryWriter()
        
        best_fold_pcc = -1.0
        best_fold_metrics = {}

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
                
                loss_patch = contrastive_loss(fused_target_img, fused_target_exp, args.temp1)
                loss_neighbor = contrastive_loss(fused_neighbor_img, fused_neighbor_exp, args.temp2)
                reconstruction_loss = mse_loss_fn(pred_exp, true_expression)
                
                loss = args.loss_ratio1 * loss_patch + args.loss_ratio2 * loss_neighbor + args.recon_weight * reconstruction_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"\n--- [epoch {epoch+1}/{args.epochs}] 结束 ---")
            print(f"   平均总损失: {avg_train_loss:.4f}")
            
            val_metrics = evaluate_model(model, val_loader, device)
            
            print("   --- 验证结果 (检索式评估) ---")
            for key, value in val_metrics.items():
                print(f"   {key:<12}: {value:.4f}")
                writer.add_scalar(f'Validation/{key}', value, epoch)
            print("   -----------------------------")
            
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            
            if val_metrics['pcc_all'] > best_fold_pcc:
                best_fold_pcc = val_metrics['pcc_all']
                best_fold_metrics = val_metrics
                print(f"   Best PCC updated: {best_fold_pcc:.4f}.")
                if args.save_checkpoints:
                    torch.save(model.state_dict(), os.path.join(log_dir, f'best_fold_{fold+1}.pth'))

        if best_fold_metrics:
            best_fold_metrics['fold'] = fold + 1
            fold_results.append(best_fold_metrics)
        writer.close()
            
    if fold_results:
        csv_results_path = os.path.join(log_dir, f'kfold_summary.csv')
        save_results_to_csv(fold_results, csv_results_path)
        df_summary = pd.DataFrame(fold_results)
        mean_pcc = df_summary['pcc_all'].mean()
        std_pcc = df_summary['pcc_all'].std()
        print("\n--- 最终平均性能 ---")
        print(f"平均 PCC (所有基因): {mean_pcc:.4f} ± {std_pcc:.4f}")
        print("=" * 56)
            
    if tee is not None:
        tee.close()

if __name__ == '__main__':
    main()
