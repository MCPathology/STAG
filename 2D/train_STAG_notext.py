import torch
import sys
import os
import argparse
import numpy as np
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter
import time
import json
from scipy.stats import pearsonr

from models.models.model_Hypergraph import STAG, contrastive_loss 

from dataset.NewHBCDataset import HBCDataset
from dataset.NewcSCCDataset import cSCCDataset
from dataset.NewHER2Dataset import HER2Dataset
from dataset.NewHESTDataset import HESTDataset, get_full_kfold_splits, get_all_subset_names


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="通用 K-Fold 交叉验证训练脚本 (支持 STAG)")
    
    parser.add_argument('--model_name', type=str, default='STAG', choices=['STAG'],
                        help="要使用的模型名称")
    parser.add_argument('--emb_dim', type=int, default=512, help="模型内部的嵌入维度")
    parser.add_argument('--depth', type=int, default=2, help="交叉注意力编码器的层数")
    parser.add_argument('--heads', type=int, default=8, help="交叉注意力编码器的头数")

    parser.add_argument('--data_name', type=str, required=True, 
                        choices=['HBC', 'cSCC', 'HER2', 
                                 'HEST_LUAD', 'HEST_IDC', 'HEST_PAAD', 'HEST_SKCM',
                                 'HEST_her2st', 'HEST_kidney', 'HEST_mouse_brain', 'HEST_PRAD',
                                 'HEST_Liver', 'HEST_Lung'],
                        help="要使用的数据集名称")
    parser.add_argument('--num_genes', type=int, default=250, help="数据集中预测的基因数量")

    parser.add_argument('--k_folds', type=int, default=5, help="K折交叉验证的折数")
    parser.add_argument('--epochs', type=int, default=50, help="每折训练的轮数")
    parser.add_argument('--lr', type=float, default=1e-4, help="学习率")
    parser.add_argument('--batch_size', type=int, default=16, help="批处理大小")
    parser.add_argument('--seed', type=int, default=1553, help="用于K折划分和随机操作的种子")
    parser.add_argument('--num_workers', type=int, default=4, help="数据加载器使用的工作进程数")
    parser.add_argument('--save_checkpoints', action='store_true', help="Save best-fold model weights.")
    parser.add_argument('--save_tensorboard', action='store_true', help="Save TensorBoard event files.")
    parser.add_argument('--save_logs', action='store_true', help="Save full stdout training logs.")

    parser.add_argument('--loss_ratio1', type=float, default=0.4, help="目标Spot对比损失的权重")
    parser.add_argument('--loss_ratio2', type=float, default=0.2, help="邻域对比损失的权重")
    parser.add_argument('--temp1', type=float, default=0.05, help="目标Spot对比损失的温度参数")
    parser.add_argument('--temp2', type=float, default=0.05, help="邻域对比损失的温度参数")
    parser.add_argument('--recon_weight', type=float, default=1, help="重建损失的权重")
    
    return parser.parse_args()

MODEL_CLASSES = {
    'STAG': STAG,
}

DATA_PATHS = {
    
    'HBC': './data/Human_breast_cancer_in_situ_capturing_transcriptomics/',
    'cSCC': './data/GSE144240',
    'HER2': './data/HER2/',
    'HEST_BASE': './data/' 
}

DATASET_CLASSES = {
    'HBC': HBCDataset, 'cSCC': cSCCDataset, 'HER2': HER2Dataset, 'HEST': HESTDataset
}

class Tee(object):
    def __init__(self, filename, mode="w", encoding='utf-8'):
        self.file = open(filename, mode, encoding=encoding)
        self.stdout = sys.stdout
        sys.stdout = self
        print(f"--- 日志已开始写入文件: {filename} ---")
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        sys.stdout = self.stdout
        self.file.close()
        print(f"--- 日志文件已关闭: {self.file.name} ---")

def evaluate_model(model, valloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in valloader:
            (_, img_tensor, true_expression, 
             hypergraph_x, hypergraph_x_exp) = data
            
            img_tensor = img_tensor.to(device, dtype=torch.float)
            true_expression = true_expression.to(device, dtype=torch.float)
            hypergraph_x = hypergraph_x.to(device, dtype=torch.float)
            hypergraph_x_exp = hypergraph_x_exp.to(device, dtype=torch.float)

            outputs = model(img_tensor, true_expression, hypergraph_x, hypergraph_x_exp)
            
            pred_exp = outputs[4]
            
            all_preds.append(pred_exp.cpu().numpy())
            all_labels.append(true_expression.cpu().numpy())
            
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    
    per_gene_pcc = []
    for g in range(all_labels_np.shape[1]):
        pcc, _ = pearsonr(all_preds_np[:, g], all_labels_np[:, g])
        if not np.isnan(pcc):
            per_gene_pcc.append(pcc)
    
    if not per_gene_pcc:
        return {
            'pcc_all': 0, 'pcc_top10': 0, 'pcc_top20': 0, 'pcc_top50': 0, 'pcc_top100': 0,
            'rmse': float('inf'), 'mse': float('inf'), 'mae': float('inf')
        }
    
    sorted_pcc = np.sort(np.array(per_gene_pcc))[::-1]
    
    metrics = {
        'pcc_all': np.mean(sorted_pcc),
        'pcc_top10': np.mean(sorted_pcc[:10]),
        'pcc_top20': np.mean(sorted_pcc[:20]),
        'pcc_top50': np.mean(sorted_pcc[:50]),
        'pcc_top100': np.mean(sorted_pcc[:100]),
        'rmse': np.sqrt(np.mean((all_preds_np - all_labels_np) ** 2)),
        'mse': np.mean((all_preds_np - all_labels_np) ** 2),
        'mae': np.mean(np.abs(all_preds_np - all_labels_np)),
    }
    return metrics

def save_results_to_csv(results, output_path):
    df = pd.DataFrame(results)
    mean_row = df.mean()
    std_row = df.std()
    mean_row.name = 'mean'
    std_row.name = 'std'
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])])
    df.loc[['mean', 'std'], 'fold'] = ''
    df.to_csv(output_path, index=True, float_format='%.4f')
    print(f"\n✅ K-Fold 评估结果已汇总并保存到: {output_path}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f'./logs/{args.model_name}/{args.data_name}/{run_timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    
    tee = Tee(os.path.join(log_dir, 'training_log.txt')) if args.save_logs else None
    
    print("=" * 60)
    print(f"K-Fold 交叉验证训练脚本 for {args.model_name}")
    print("-" * 60)
    for key, value in vars(args).items():
        print(f"{key:<15}: {value}")
    print(f"{'device':<15}: {device}")
    print("=" * 60)

    kfold_split_plans = [] 
    split_save_path = os.path.join(log_dir, f"{args.data_name}_kfold_splits.json")

    if args.data_name.startswith('HEST_'):
        hest_subset_name = args.data_name.replace('HEST_', '')
        if hest_subset_name in ['her2st', 'kidney', 'mouse_brain', 'PRAD', 'Liver', 'Lung']:
            data_path = os.path.join(DATA_PATHS['HEST_BASE'], 'hest1k_datasets', hest_subset_name)
        else:
            data_path = os.path.join(DATA_PATHS['HEST_BASE'], f'hest_data_{hest_subset_name.upper()}')
        
        print(f"检测到 HEST 数据集，路径: {data_path}")
        
        kfold_splits_from_func = get_full_kfold_splits(data_path, n_splits=args.k_folds, random_state=args.seed)
        
        all_files = get_all_subset_names(data_path)
        json_output = {
            "k_folds": args.k_folds,
            "seed": args.seed,
            "total_files": len(all_files),
            "folds": []
        }
        
        for fold_data in kfold_splits_from_func:
            train_files = fold_data['train']
            val_files = fold_data['val']
            
            json_output["folds"].append({
                "fold_index": fold_data['fold'],
                "train_files": train_files,
                "val_files": val_files
            })
            
            plan = {
                'type': 'HEST', 'data_path': data_path,
                'gene_file_path': f'select_genes/HEST_{hest_subset_name.upper()}_gene.npy',
                'train_subsets': train_files, 'val_subsets': val_files
            }
            kfold_split_plans.append(plan)
            
        with open(split_save_path, 'w') as f:
            json.dump(json_output, f, indent=4)
        print(f"\n✅ HEST 数据划分方案已保存到: {split_save_path}")

    else:
        print(f"--- 正在为 {args.data_name} 生成或加载数据划分方案... ---")
        DatasetClass = DATASET_CLASSES.get(args.data_name)
        if not DatasetClass:
            raise ValueError(f"未找到数据集类: {args.data_name}")
        
        _ = DatasetClass(path=DATA_PATHS[args.data_name], mode='train', k_folds=args.k_folds, fold_index=0, seed=args.seed, split_save_path=split_save_path)
        with open(split_save_path, 'r') as f: split_data = json.load(f)
        for fold_info in split_data['folds']:
            kfold_split_plans.append({'type': 'Standard', 'data_name': args.data_name, 'fold_index': fold_info['fold_index']})

    fold_results = []
    
    for fold, plan in enumerate(kfold_split_plans):
        print(f"\n{'='*25} FOLD {fold + 1}/{len(kfold_split_plans)} {'='*25}")
        
        datasets_to_close = [] 
        
        if plan['type'] == 'HEST':
            train_dataset = HESTDataset(
                data_root_path=plan['data_path'], mode='train', 
                specific_subsets=plan['train_subsets'], selected_genes_file_path=plan['gene_file_path']
            )
            val_dataset = HESTDataset(
                data_root_path=plan['data_path'], mode='val', 
                specific_subsets=plan['val_subsets'], selected_genes_file_path=plan['gene_file_path']
            )
            datasets_to_close.extend([train_dataset, val_dataset])
        
        elif plan['type'] == 'Standard':
            DatasetClass = DATASET_CLASSES[plan['data_name']]
            data_path = DATA_PATHS[plan['data_name']]
            
            train_dataset = DatasetClass(
                path=data_path, mode='train', k_folds=args.k_folds, 
                fold_index=plan['fold_index'], seed=args.seed, split_save_path=split_save_path
            )
            val_dataset = DatasetClass(
                path=data_path, mode='val', k_folds=args.k_folds, 
                fold_index=plan['fold_index'], seed=args.seed, split_save_path=split_save_path
            )
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"--- 警告: 第 {fold+1} 折的训练集或验证集为空，跳过此折。 ---")
            if plan['type'] == 'HEST':
                for ds in datasets_to_close: ds.close_caches()
            continue

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = STAG(
            num_genes=args.num_genes,
            emb_dim=args.emb_dim,
            depth1=args.depth,
            num_heads1=args.heads
        ).to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"--- 检测到 {torch.cuda.device_count()} 个 GPU，使用 DataParallel ---")
            model = torch.nn.DataParallel(model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        mse_loss_fn = torch.nn.MSELoss()
        
        writer = SummaryWriter(log_dir=os.path.join(log_dir, f'fold_{fold+1}')) if args.save_tensorboard else NullSummaryWriter()
        
        best_fold_pcc = -1.0
        best_fold_metrics = {}

        for epoch in range(args.epochs):
            model.train()
            total_loss, total_loss_patch, total_loss_neighbor, total_loss_recon = 0, 0, 0, 0
            epoch_start_time = time.time()
            
            for data in train_loader:
                (_, img_tensor, true_expression, 
                 hypergraph_x, hypergraph_x_exp) = data
                
                img_tensor = img_tensor.to(device, dtype=torch.float)
                true_expression = true_expression.to(device, dtype=torch.float)
                hypergraph_x = hypergraph_x.to(device, dtype=torch.float)
                hypergraph_x_exp = hypergraph_x_exp.to(device, dtype=torch.float)
                
                optimizer.zero_grad()
                
                outputs = model(img_tensor, true_expression, hypergraph_x, hypergraph_x_exp)
                
                fused_target_img, fused_target_exp, fused_neighbor_img, fused_neighbor_exp, pred_exp,_ = outputs
                
                loss_patch = contrastive_loss(fused_target_img, fused_target_exp, args.temp1)
                loss_neighbor = contrastive_loss(fused_neighbor_img, fused_neighbor_exp, args.temp2)
                reconstruction_loss = mse_loss_fn(pred_exp, true_expression)
                
                loss = args.loss_ratio1 * loss_patch + args.loss_ratio2 * loss_neighbor + args.recon_weight * reconstruction_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_loss_patch += loss_patch.item()
                total_loss_neighbor += loss_neighbor.item()
                total_loss_recon += reconstruction_loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            epoch_duration = time.time() - epoch_start_time
            
            print(f"\n--- [Fold {fold+1}/Epoch {epoch+1}] 结束 ---")
            print(f"   平均总损失: {avg_train_loss:.4f}")
            print(f"   训练耗时: {epoch_duration:.2f} 秒")
            
            val_metrics = evaluate_model(model, val_loader, device)
            
            print("   --- 验证结果 ---")
            for key, value in val_metrics.items():
                print(f"   {key:<12}: {value:.4f}")
                writer.add_scalar(f'Validation/{key}', value, epoch)
            print("   ----------------")
            
            writer.add_scalar('Loss/Total_Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Patch_Contrastive', total_loss_patch / len(train_loader), epoch)
            writer.add_scalar('Loss/Neighbor_Contrastive', total_loss_neighbor / len(train_loader), epoch)
            writer.add_scalar('Loss/Reconstruction', total_loss_recon / len(train_loader), epoch)
            
            if val_metrics['pcc_all'] > best_fold_pcc:
                best_fold_pcc = val_metrics['pcc_all']
                best_fold_metrics = val_metrics
                print(f"   Best PCC updated: {best_fold_pcc:.4f}.")
                if args.save_checkpoints:
                    save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                    model_save_path = os.path.join(log_dir, f'{args.model_name}_{args.data_name}_best_fold_{fold+1}.pth')
                    torch.save(save_model.state_dict(), model_save_path)

        if best_fold_metrics:
            best_fold_metrics['fold'] = fold + 1
            fold_results.append(best_fold_metrics)
            
        writer.close()
            
    print(f"\n{'='*20} K-FOLD 交叉验证总结 {'='*20}")
    if fold_results:
        csv_results_path = os.path.join(log_dir, f'{args.model_name}_{args.data_name}_kfold_summary.csv')
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
