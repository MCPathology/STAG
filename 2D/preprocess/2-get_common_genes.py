import os
import os.path as osp
import numpy as np
import pandas as pd
import anndata
import shutil
from scipy.sparse import issparse
from typing import List, Optional


def find_optimal_subset_branch_and_bound(st_dir: str, min_common_genes: int = 250) -> Optional[List[str]]:
    print("--- 正在预加载所有样本的基因数据以进行子集搜索 ---")
    sample_gene_sets = {}
    all_files = os.listdir(st_dir)
    
    for filename in all_files:
        if not filename.endswith('.h5ad'):
            continue
        sample_name = filename.split('.')[0]
        st_file = osp.join(st_dir, filename)
        try:
            adata = anndata.read_h5ad(st_file)
            sample_gene_sets[sample_name] = set(adata.var_names)
            print(f"  - 已加载样本 '{sample_name}' 用于分析")
        except Exception as e:
            print(f"  - 警告: 读取 {st_file} 失败，已跳过。错误: {e}")

    all_sample_names = sorted(list(sample_gene_sets.keys()))
    num_samples = len(all_sample_names)
    
    if num_samples < 1:
        print("\n错误: 未加载到任何有效样本。")
        return None

    print(f"\n--- 开始执行分支定界算法 (共 {num_samples} 个样本) ---")
    
    best_subset = []

    def backtrack(start_index, current_subset, current_intersection):
        nonlocal best_subset
        potential_max_size = len(current_subset) + (num_samples - start_index)
        if potential_max_size <= len(best_subset):
            return

        for i in range(start_index, num_samples):
            sample_name = all_sample_names[i]
            if len(current_subset) + (num_samples - i) <= len(best_subset):
                return
            new_intersection = current_intersection.intersection(sample_gene_sets[sample_name])
            if len(new_intersection) >= min_common_genes:
                new_subset = current_subset + [sample_name]
                if len(new_subset) > len(best_subset):
                    best_subset = new_subset
                    print(f"  -> 发现一个更大的有效子集，大小为: {len(best_subset)}")
                backtrack(i + 1, new_subset, new_intersection)

    for i in range(num_samples):
        if num_samples - i <= len(best_subset):
            break
        initial_sample = all_sample_names[i]
        initial_set = sample_gene_sets[initial_sample]
        if len(initial_set) >= min_common_genes:
            if len(best_subset) == 0:
                best_subset = [initial_sample]
                print(f"  -> 发现一个更大的有效子集，大小为: 1")
            backtrack(i + 1, [initial_sample], initial_set)

    if best_subset:
        print(f"\n搜索完成，找到的最优子集大小为: {len(best_subset)}")
        return best_subset
    else:
        return None

def remove_excluded_samples(largest_subset_names: List[str], wsi_dir: str, patches_dir: str, st_dir: str, dry_run: bool = True):
    if dry_run:
        print("\n" + "="*20 + " 演习模式 (Dry Run) 启动 " + "="*20)
        print("!!! 不会删除任何文件，只会列出将要执行的操作。")
    else:
        print("\n" + "!"*20 + " 警告：真实删除模式启动 " + "!"*20)
        print("!!! 将会永久删除文件和文件夹。请谨慎操作！")

    names_to_keep = set(largest_subset_names)
    print(f"\n将保留以下 {len(names_to_keep)} 个样本的相关文件：{sorted(list(names_to_keep))}")

    print("\n--- 正在检查 WSI 目录:", wsi_dir, "---")
    if os.path.isdir(wsi_dir):
        deleted_count = 0
        for filename in os.listdir(wsi_dir):
            if filename.endswith('.tif'):
                sample_name = os.path.splitext(filename)[0]
                if sample_name not in names_to_keep:
                    file_path = os.path.join(wsi_dir, filename)
                    print(f"  - 待删除文件: {file_path}")
                    if not dry_run:
                        try:
                            os.remove(file_path)
                            print(f"    ✅ 已删除。")
                        except OSError as e:
                            print(f"    ❌ 删除失败: {e}")
                    deleted_count += 1
        action = "计划删除" if dry_run else "已删除"
        print(f"  完成。{action} {deleted_count} 个 .tif 文件。")
    else:
        print(f"警告: 目录不存在，跳过。")

    print("\n--- 正在检查 ST 目录:", st_dir, "---")
    if os.path.isdir(st_dir):
        deleted_count = 0
        for filename in os.listdir(st_dir):
            if filename.endswith('.h5ad'):
                sample_name = os.path.splitext(filename)[0]
                if sample_name not in names_to_keep:
                    file_path = os.path.join(st_dir, filename)
                    print(f"  - 待删除文件: {file_path}")
                    if not dry_run:
                        try:
                            os.remove(file_path)
                            print(f"    ✅ 已删除。")
                        except OSError as e:
                            print(f"    ❌ 删除失败: {e}")
                    deleted_count += 1
        action = "计划删除" if dry_run else "已删除"
        print(f"  完成。{action} {deleted_count} 个 .h5ad 文件。")
    else:
        print(f"警告: 目录不存在，跳过。")
            
    print("\n--- 正在检查 Patches 目录:", patches_dir, "---")
    if not os.path.isdir(patches_dir):
        print(f"警告: 目录不存在，跳过。")
    else:
        deleted_count = 0
        for filename in os.listdir(patches_dir):
            if filename.endswith('.h5'):
                sample_name = os.path.splitext(filename)[0]
                if sample_name not in names_to_keep:
                    file_path = os.path.join(patches_dir, filename)
                    print(f"  - 待删除文件: {file_path}")
                    if not dry_run:
                        try:
                            os.remove(file_path)
                            print(f"    ✅ 已删除。")
                        except OSError as e:
                            print(f"    ❌ 删除失败: {e}")
                    deleted_count += 1
        if deleted_count == 0:
            print("  该目录中没有需要删除的文件。")
        else:
            action = "计划删除" if dry_run else "已删除"
            print(f"  完成。{action} {deleted_count} 个 .h5 文件。")

    print("\n" + "="*20 + " 清理操作全部完成 " + "="*20)

def get_genes(wsi_dir: str, st_dir: str, gene_file_path: str) -> Optional[np.ndarray]:
    files = os.listdir(wsi_dir)
    names = [i.split('.')[0] for i in files if i.endswith('.tif')]
    if not names:
        print("警告: WSI 目录中未找到任何样本 (.tif 文件)，无法生成基因列表。")
        return None
    
    all_gene_expression_sum = {}
    gene_sets_per_sample = []
    
    print(f"\n--- 开始从 {len(names)} 个样本中筛选Top基因 ---")
    
    for i in names:
        st_file = osp.join(st_dir, i + '.h5ad')
        if not osp.exists(st_file):
            print(f"跳过文件: {st_file} (ST 文件不存在)")
            continue
        print(f"处理文件: {st_file}")
        try:
            adata = anndata.read_h5ad(st_file)
        except Exception as e:
            print(f"读取 {st_file} 失败: {e}")
            continue
        gene_sets_per_sample.append(set(adata.var_names))
        X = adata.X
        if issparse(X):
            gene_sum = np.array(X.sum(axis=0)).flatten()
        else:
            gene_sum = np.sum(X, axis=0)
        gene_expression_series = pd.Series(gene_sum, index=adata.var_names)
        for gene, total_exp in gene_expression_series.items():
            all_gene_expression_sum[gene] = all_gene_expression_sum.get(gene, 0) + total_exp
        print(f"样本 {i} 已处理。")

    if not all_gene_expression_sum or not gene_sets_per_sample:
        print("警告: 未检测到有效的基因表达数据，无法进行筛选。")
        return None
        
    common_genes = set.intersection(*gene_sets_per_sample)
    print(f"\n在剩余的 {len(names)} 个样本中，共找到 {len(common_genes)} 个共同存在的基因。")
    
    if not common_genes:
        print("警告: 样本之间没有共同的基因，无法进行筛选。")
        return None

    all_gene_expression_sum_series = pd.Series(all_gene_expression_sum)
    common_gene_expression = all_gene_expression_sum_series[all_gene_expression_sum_series.index.isin(common_genes)]
    sorted_common_genes = common_gene_expression.sort_values(ascending=False)
    
    num_to_select = min(250, len(sorted_common_genes))
    if len(sorted_common_genes) < 250:
        print(f"警告: 共同基因数量 ({len(sorted_common_genes)}) 少于250个，将只选取这 {num_to_select} 个基因。")
        
    top_gene_names = sorted_common_genes.head(num_to_select).index.to_numpy()
    print(f"有 {len(top_gene_names)} 个基因： {top_gene_names}")
    
    print("\n" + "="*20 + " 基因筛选结果 " + "="*20)
    print(f"已筛选出总表达量最高的 {num_to_select} 个基因。")
    
    try:
        save_dir = osp.dirname(gene_file_path)
        if save_dir and not osp.exists(save_dir):
            os.makedirs(save_dir)
        np.save(gene_file_path, top_gene_names)
        print(f"\n✅ 已将筛选出的 {num_to_select} 个基因名称保存到: {gene_file_path}")
    except Exception as e:
        print(f"\n❌ 保存 .npy 文件失败: {e}")
        
    return top_gene_names


if __name__ == "__main__":
    BASE_DIR = './data/hest1k_datasets/Liver' 
    WSI_DIR = osp.join(BASE_DIR, 'wsis') 
    ST_DIR = osp.join(BASE_DIR, 'st') 
    PATCHES_DIR = osp.join(BASE_DIR, 'patches') 
    GENE_OUTPUT_PATH = 'select_genes/HEST_LIVER_gene.npy'

    print("="*20 + " 步骤 1: 寻找最优样本子集 " + "="*20)
    largest_subset = find_optimal_subset_branch_and_bound(st_dir=ST_DIR, min_common_genes=250)

    if largest_subset:
        print(f"\n成功找到最大子集，包含 {len(largest_subset)} 个样本。")
        
        print("\n" + "="*20 + " 步骤 2: 清理无关文件 " + "="*20)
        
        print("\n--- 首先，执行一次演习（Dry Run），检查待删除项 ---")
        remove_excluded_samples(
            largest_subset_names=largest_subset,
            wsi_dir=WSI_DIR,
            patches_dir=PATCHES_DIR,
            st_dir=ST_DIR,
            dry_run=True
        )
        
        print("\n" + "!"*60)
        print("!!! 警告：请仔细检查上面的演习输出列表。")
        print("!!! 如果确认无误，可以取消下面代码块的注释来执行真正的删除操作。")
        print("!"*60)
        
        user_confirmation = input("\n您确定要永久删除上述文件和文件夹吗？请输入 'yes' 以确认: ")
        if user_confirmation.lower() == 'yes':
            print("\n--- 用户已确认，开始执行真正的删除操作 ---")
            remove_excluded_samples(
                largest_subset_names=largest_subset,
                wsi_dir=WSI_DIR,
                patches_dir=PATCHES_DIR,
                st_dir=ST_DIR,
                dry_run=False
            )
            print("\n文件清理完成。")
        else:
            print("\n操作已取消，没有文件被删除。")

        print("\n" + "="*20 + " 步骤 3: 生成最终基因列表 " + "="*20)
        get_genes(WSI_DIR, ST_DIR, GENE_OUTPUT_PATH)

    else:
        print("\n未能找到符合条件的样本子集，程序终止，不执行任何文件清理或基因筛选。")

    print("\n所有流程执行完毕。")
