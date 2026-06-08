
import os
import glob
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import issparse


def load_cSCC(data_path):
    tsv_files = sorted(glob.glob(os.path.join(data_path, '*_stdata.tsv')))
    print(f"[cSCC] Found {len(tsv_files)} expression files")

    all_dfs = []
    for f in tsv_files:
        df = pd.read_csv(f, sep='\t', index_col=0)
        sample_name = os.path.basename(f).replace('_stdata.tsv', '')
        df.index = [f"{sample_name}_{idx}" for idx in df.index]
        all_dfs.append(df)

    combined = pd.concat(all_dfs, join='inner')
    print(f"[cSCC] Combined: {combined.shape[0]} spots x {combined.shape[1]} common genes")

    adata = ad.AnnData(X=combined.values.astype(np.float32),
                       obs=pd.DataFrame(index=combined.index),
                       var=pd.DataFrame(index=combined.columns))
    return adata


def load_HER2(data_path):
    count_dir = os.path.join(data_path, 'count-matrices')
    tsv_files = sorted(glob.glob(os.path.join(count_dir, '*.tsv')))
    print(f"[HER2] Found {len(tsv_files)} expression files")

    all_dfs = []
    for f in tsv_files:
        df = pd.read_csv(f, sep='\t', index_col=0)
        if df.shape[0] > 1000 and df.shape[1] < 1000:
            df = df.T
        sample_name = os.path.basename(f).replace('.tsv', '')
        df.index = [f"{sample_name}_{idx}" for idx in df.index]
        all_dfs.append(df)

    combined = pd.concat(all_dfs, join='inner')
    print(f"[HER2] Combined: {combined.shape[0]} spots x {combined.shape[1]} common genes")

    adata = ad.AnnData(X=combined.values.astype(np.float32),
                       obs=pd.DataFrame(index=combined.index),
                       var=pd.DataFrame(index=combined.columns))
    return adata


def load_HBC(data_path):
    tsv_files = sorted(glob.glob(os.path.join(data_path, '*_stdata.tsv')))
    print(f"[HBC] Found {len(tsv_files)} expression files")

    all_dfs = []
    for f in tsv_files:
        df = pd.read_csv(f, sep='\t', index_col=0)
        sample_name = os.path.basename(f).replace('_stdata.tsv', '')
        df.index = [f"{sample_name}_{idx}" for idx in df.index]
        all_dfs.append(df)

    combined = pd.concat(all_dfs, join='inner')
    print(f"[HBC] Combined: {combined.shape[0]} spots x {combined.shape[1]} common genes (ENSG IDs)")

    mapping_path = os.path.join(SAVE_DIR, 'STNet_mapping.csv')
    if os.path.exists(mapping_path):
        mapping_df = pd.read_csv(mapping_path)
        ensg_to_symbol = dict(zip(mapping_df['归一化ENSG'], mapping_df['Approved_symbol']))
        valid_cols = [col for col in combined.columns if col in ensg_to_symbol and pd.notna(ensg_to_symbol[col])]
        combined = combined[valid_cols]
        combined.columns = [ensg_to_symbol[col] for col in combined.columns]
        combined = combined.loc[:, ~combined.columns.duplicated()]
        print(f"[HBC] After ENSG -> Symbol mapping: {combined.shape[0]} spots x {combined.shape[1]} genes")
    else:
        print(f"[HBC] WARNING: Mapping file not found at {mapping_path}, using ENSG IDs directly")

    adata = ad.AnnData(X=combined.values.astype(np.float32),
                       obs=pd.DataFrame(index=combined.index),
                       var=pd.DataFrame(index=combined.columns))
    return adata


DATASET_LOADERS = {
    'cSCC': load_cSCC,
    'HER2': load_HER2,
    'HBC': load_HBC,
}

DATA_PATHS = {
    'cSCC': './data/GSE144240',
    'HER2': './data/HER2/',
    'HBC': './data/Human_breast_cancer_in_situ_capturing_transcriptomics/',
}

SAVE_DIR = 'select_genes'


def select_hvg(adata, n_top_genes=250, dataset_name='dataset'):
    print(f"\n[{dataset_name}] Starting HVG selection...")
    print(f"  Input: {adata.shape[0]} spots x {adata.shape[1]} genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_available = adata.shape[1]
    n_select = min(n_top_genes, n_available)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_select, flavor='seurat')

    hvg_mask = adata.var['highly_variable']
    hvg_names = adata.var_names[hvg_mask].to_numpy()

    print(f"  Selected {len(hvg_names)} HVGs")
    print(f"  Top 10 HVGs: {hvg_names[:10].tolist()}")

    return hvg_names


def process_dataset(dataset_name, n_top_genes=250):
    print(f"\n{'='*60}")
    print(f"  Processing: {dataset_name}")
    print(f"{'='*60}")

    data_path = DATA_PATHS[dataset_name]
    loader = DATASET_LOADERS[dataset_name]

    adata = loader(data_path)

    hvg_names = select_hvg(adata, n_top_genes=n_top_genes, dataset_name=dataset_name)

    os.makedirs(SAVE_DIR, exist_ok=True)
    hvg_save_path = os.path.join(SAVE_DIR, f'{dataset_name}_HVG_Genes.npy')
    np.save(hvg_save_path, hvg_names)
    print(f"\n  Saved HVG list to: {hvg_save_path}")

    orig_gene_map = {
        'cSCC': 'cSCC_Selected_Genes.npy',
        'HER2': 'HER2_Selected_Genes.npy',
        'HBC': 'HBC_Selected_Genes.npy',
    }
    orig_path = os.path.join(SAVE_DIR, orig_gene_map[dataset_name])
    if os.path.exists(orig_path):
        orig_genes = set(np.load(orig_path, allow_pickle=True).tolist())
        hvg_set = set(hvg_names.tolist())
        overlap = orig_genes & hvg_set
        print(f"\n  --- Overlap with original top-expression genes ---")
        print(f"  Original: {len(orig_genes)} genes")
        print(f"  HVG:      {len(hvg_set)} genes")
        print(f"  Overlap:  {len(overlap)} genes ({len(overlap)/len(orig_genes)*100:.1f}%)")

    return hvg_names


def main():
    parser = argparse.ArgumentParser(description="R1-Q2: Select 250 HVGs for ST datasets")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['cSCC', 'HER2', 'HBC', 'all'],
                        help="Which dataset to process")
    parser.add_argument('--n_genes', type=int, default=250,
                        help="Number of HVGs to select")
    args = parser.parse_args()

    datasets = ['cSCC', 'HER2', 'HBC'] if args.dataset == 'all' else [args.dataset]

    print("=" * 60)
    print("R1-Q2: Highly Variable Gene (HVG) Selection")
    print(f"  Datasets: {datasets}")
    print(f"  N genes:  {args.n_genes}")
    print("=" * 60)

    for ds in datasets:
        process_dataset(ds, n_top_genes=args.n_genes)

    print(f"\n{'='*60}")
    print("All done. HVG files saved to:", SAVE_DIR)
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
