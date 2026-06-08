
import os
import glob
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def build_symbol_to_ensembl(gene_symbols, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        print(f"已加载缓存的 symbol→Ensembl 映射: {len(cached)} 条")
        return cached

    try:
        import mygene
        mg = mygene.MyGeneInfo()
        results = mg.querymany(gene_symbols, scopes='symbol',
                               fields='ensembl.gene', species='human',
                               returnall=True)
        mapping = {}
        for r in results['out']:
            query = r.get('query', '')
            ensembl = r.get('ensembl', {})
            if isinstance(ensembl, list):
                ensembl = ensembl[0]
            ens_id = ensembl.get('gene', '') if isinstance(ensembl, dict) else ''
            if ens_id:
                mapping[query] = ens_id
        print(f"mygene 映射: {len(mapping)}/{len(gene_symbols)} 个基因已解析")
    except ImportError:
        print("mygene 未安装，请运行: pip install mygene")
        mapping = {}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"映射已缓存到 {cache_path}")

    return mapping


def load_local_dicts(model_dir, model_variant='v1-10m'):
    token_dict = None
    median_dict = None

    if model_variant.startswith('v1'):
        token_paths = [
            'geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl',
            'geneformer/token_dictionary.pkl',
            'token_dictionary.pkl',
        ]
        median_paths = [
            'geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl',
            'geneformer/gene_median_dictionary.pkl',
            'gene_median_dictionary.pkl',
        ]
    else:
        token_paths = [
            'geneformer/token_dictionary_gc104M.pkl',
            'geneformer/token_dictionary.pkl',
            'token_dictionary.pkl',
        ]
        median_paths = [
            'geneformer/gene_median_dictionary_gc104M.pkl',
            'geneformer/gene_median_dictionary.pkl',
            'gene_median_dictionary.pkl',
        ]

    for subpath in token_paths:
        p = os.path.join(model_dir, subpath)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                token_dict = pickle.load(f)
            print(f"已加载 token_dictionary: {p} ({len(token_dict)} 个基因)")
            break

    if token_dict is None:
        for root, dirs, files in os.walk(model_dir):
            for fname in files:
                if 'token_dictionary' in fname and fname.endswith('.pkl'):
                    p = os.path.join(root, fname)
                    with open(p, 'rb') as f:
                        token_dict = pickle.load(f)
                    print(f"已加载 token_dictionary: {p} ({len(token_dict)} 条)")
                    break
            if token_dict is not None:
                break

    if token_dict is None:
        raise FileNotFoundError(f"在 {model_dir} 中找不到 token_dictionary*.pkl")

    for subpath in median_paths:
        p = os.path.join(model_dir, subpath)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                median_dict = pickle.load(f)
            print(f"已加载 gene_median_dictionary: {p} ({len(median_dict)} 个基因)")
            break

    if median_dict is None:
        for root, dirs, files in os.walk(model_dir):
            for fname in files:
                if 'gene_median' in fname and fname.endswith('.pkl'):
                    p = os.path.join(root, fname)
                    with open(p, 'rb') as f:
                        median_dict = pickle.load(f)
                    print(f"已加载 gene_median_dictionary: {p} ({len(median_dict)} 条)")
                    break
            if median_dict is not None:
                break

    if median_dict is None:
        print("警告: 未找到 gene_median_dictionary，将使用原始计数进行排序。")

    return token_dict, median_dict


def tokenize_spot(expression_values, gene_ensembl_ids, token_dict, median_dict,
                  max_len=2048):
    gene_expr_pairs = []
    for expr_val, ens_id in zip(expression_values, gene_ensembl_ids):
        if ens_id not in token_dict:
            continue
        if expr_val <= 0:
            continue
        if median_dict and ens_id in median_dict:
            median_val = median_dict[ens_id]
            norm_val = expr_val / median_val if median_val > 0 else expr_val
        else:
            norm_val = expr_val
        gene_expr_pairs.append((norm_val, ens_id))

    if not gene_expr_pairs:
        return None

    gene_expr_pairs.sort(key=lambda x: x[0], reverse=True)
    token_ids = [token_dict[ens_id] for _, ens_id in gene_expr_pairs[:max_len]]
    return token_ids


def extract_embeddings_batch(all_token_ids, model, device, batch_size=64):
    max_seq_len = max(len(t) for t in all_token_ids)
    n = len(all_token_ids)

    padded_ids = np.zeros((n, max_seq_len), dtype=np.int64)
    attention_masks = np.zeros((n, max_seq_len), dtype=np.int64)
    for i, tokens in enumerate(all_token_ids):
        padded_ids[i, :len(tokens)] = tokens
        attention_masks[i, :len(tokens)] = 1

    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="    推理", leave=False):
            end = min(start + batch_size, n)
            input_ids = torch.tensor(padded_ids[start:end], dtype=torch.long).to(device)
            attn_mask = torch.tensor(attention_masks[start:end], dtype=torch.long).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = outputs.last_hidden_state

            mask_exp = attn_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            all_embeddings.append(pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def process_wsi(stdata_path, symbol_to_ensembl, token_dict, median_dict,
                model, device, batch_size, max_len, selected_genes=None):
    feats_all = pd.read_csv(stdata_path, sep='\t', index_col=0, header=0)

    if selected_genes is not None:
        available = [g for g in selected_genes if g in feats_all.columns]
        if not available:
            return None, []
        feats_all = feats_all[available]

    gene_symbols = list(feats_all.columns)

    gene_ensembl_ids = [symbol_to_ensembl.get(s, '') for s in gene_symbols]
    valid_gene_count = sum(1 for eid in gene_ensembl_ids if eid in token_dict)
    if valid_gene_count == 0:
        return None, []

    spot_ids = list(feats_all.index)
    all_token_ids = []
    valid_spot_ids = []

    for spot_id in spot_ids:
        expr = feats_all.loc[spot_id].values.astype(np.float32)
        if np.sum(expr) == 0:
            continue
        tokens = tokenize_spot(expr, gene_ensembl_ids, token_dict, median_dict, max_len)
        if tokens and len(tokens) > 0:
            all_token_ids.append(tokens)
            valid_spot_ids.append(str(spot_id))

    if not all_token_ids:
        return None, []

    embeddings = extract_embeddings_batch(all_token_ids, model, device, batch_size)
    return embeddings, valid_spot_ids


def main():
    parser = argparse.ArgumentParser(
        description="离线预计算冻结 Geneformer 的 per-spot embeddings (cSCC)")
    parser.add_argument('--data_path', type=str, required=True,
                        help='GSE144240 目录路径，包含 *_stdata.tsv 文件')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='本地 clone 的 Geneformer 根目录路径')
    parser.add_argument('--model_variant', type=str, default='v1-10m',
                        choices=['v1-10m', 'v2-104m', 'v2-316m'],
                        help='Geneformer 模型版本 (默认: v1-10m, hidden_dim=256)')
    parser.add_argument('--selected_genes', type=str, default=None,
                        help='可选: 指定基因列表 .npy 文件路径，仅用这些基因做 tokenization'
                             '（默认使用全转录组）')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    variant_subdirs = {
        'v1-10m': 'Geneformer-V1-10M',
        'v2-104m': 'Geneformer-V2-104M',
        'v2-316m': 'Geneformer-V2-316M',
    }
    model_weights_dir = os.path.join(args.model_dir, variant_subdirs[args.model_variant])
    if not os.path.exists(model_weights_dir):
        print(f"子目录不存在: {model_weights_dir}，尝试使用根目录加载")
        model_weights_dir = args.model_dir

    print(f"正在从本地加载 Geneformer ({args.model_variant}): {model_weights_dir}")
    from transformers import AutoModel, BertModel
    try:
        model = AutoModel.from_pretrained(model_weights_dir, trust_remote_code=True)
    except Exception:
        model = BertModel.from_pretrained(model_weights_dir, trust_remote_code=True)
    model = model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.hidden_size
    print(f"Geneformer hidden_dim: {hidden_dim}")

    token_dict, median_dict = load_local_dicts(args.model_dir, args.model_variant)

    selected_genes = None
    if args.selected_genes is not None:
        selected_genes = np.load(args.selected_genes, allow_pickle=True).tolist()
        print(f"已加载基因子集: {len(selected_genes)} 个基因 (来自 {args.selected_genes})")

    stdata_files = sorted(glob.glob(os.path.join(args.data_path, '*_stdata.tsv')))
    print(f"找到 {len(stdata_files)} 个 stdata 文件")
    if not stdata_files:
        print("错误: 未找到 *_stdata.tsv 文件。")
        return

    if selected_genes is not None:
        all_gene_symbols = sorted(selected_genes)
        print(f"仅使用指定的 {len(all_gene_symbols)} 个基因做 tokenization")
    else:
        all_gene_symbols = set()
        for f in stdata_files:
            df = pd.read_csv(f, sep='\t', index_col=0, header=0, nrows=0)
            all_gene_symbols.update(df.columns.tolist())
        all_gene_symbols = sorted(all_gene_symbols)
        print(f"使用全转录组，唯一基因符号总数: {len(all_gene_symbols)}")

    cache_path = os.path.join(args.output_dir, 'gene_symbol_to_ensembl.json')
    symbol_to_ensembl = build_symbol_to_ensembl(all_gene_symbols, cache_path)

    in_vocab = sum(1 for s in all_gene_symbols
                   if symbol_to_ensembl.get(s, '') in token_dict)
    print(f"在 Geneformer 词表中的基因数: {in_vocab}/{len(all_gene_symbols)}")

    summary = {}
    for stdata_path in tqdm(stdata_files, desc="处理 WSI"):
        basename = os.path.basename(stdata_path).replace('_stdata.tsv', '')

        embeddings, spot_ids = process_wsi(
            stdata_path, symbol_to_ensembl, token_dict, median_dict,
            model, args.device, args.batch_size, args.max_len,
            selected_genes=selected_genes
        )

        if embeddings is None:
            print(f"  [跳过] {basename}: 无有效 spots")
            continue

        emb_path = os.path.join(args.output_dir, f"{basename}_geneformer_emb.npy")
        np.save(emb_path, embeddings.astype(np.float32))

        ids_path = os.path.join(args.output_dir, f"{basename}_geneformer_spotids.json")
        with open(ids_path, 'w') as f:
            json.dump(spot_ids, f)

        summary[basename] = {
            'num_spots': len(spot_ids),
            'emb_dim': int(embeddings.shape[1]),
        }
        print(f"  {basename}: {len(spot_ids)} 个 spots, dim={embeddings.shape[1]}")

    config = {
        'model_dir': args.model_dir,
        'model_variant': args.model_variant,
        'hidden_dim': hidden_dim,
        'max_len': args.max_len,
        'selected_genes': args.selected_genes if args.selected_genes else 'full_transcriptome',
        'num_input_genes': len(all_gene_symbols),
        'num_wsis': len(summary),
        'total_spots': sum(v['num_spots'] for v in summary.values()),
        'wsi_details': summary,
    }
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n完成! 已保存到: {args.output_dir}")
    print(f"  WSI 总数: {config['num_wsis']}, Spot 总数: {config['total_spots']}")
    print(f"  Embedding 维度: {hidden_dim}")


if __name__ == '__main__':
    main()
