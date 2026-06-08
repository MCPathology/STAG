
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer

def load_model(model_path, device):
    from open_clip import create_model

    model = create_model("coca_ViT-L-14", device=device)

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    tokenizer = get_tokenizer("coca_ViT-L-14")
    model.to(device).eval()
    return model, None, tokenizer

def encode_texts(model, tokenizer, texts, device):
    text_inputs = tokenizer(texts)
    with torch.no_grad():
        feats = model.encode_text(text_inputs)
    return F.normalize(feats, p=2, dim=-1)

MODEL_PATH = './loki_model/checkpoint.pt'
GENE_DIR = 'select_genes'

HVG_GENE_FILES = {
    'cSCC': os.path.join(GENE_DIR, 'cSCC_HVG_Genes.npy'),
    'HER2': os.path.join(GENE_DIR, 'HER2_HVG_Genes.npy'),
    'HBC':  os.path.join(GENE_DIR, 'HBC_HVG_Genes.npy'),
}

OUTPUT_FILES = {
    'cSCC': os.path.join(GENE_DIR, 'cSCC_HVG_loki_text_encode.npy'),
    'HER2': os.path.join(GENE_DIR, 'HER2_HVG_loki_text_encode.npy'),
    'HBC':  os.path.join(GENE_DIR, 'HBC_HVG_loki_text_encode.npy'),
}

def encode_genes(gene_names, model, tokenizer, device):
    all_embeddings = []
    with torch.no_grad():
        for gene_name in tqdm(gene_names, desc="Encoding genes"):
            embedding = encode_texts(model, tokenizer, [gene_name], device)
            all_embeddings.append(embedding.cpu().numpy())
    return np.vstack(all_embeddings)


def process_dataset(dataset_name, model, tokenizer, device):
    gene_path = HVG_GENE_FILES[dataset_name]
    output_path = OUTPUT_FILES[dataset_name]

    if not os.path.exists(gene_path):
        print(f"[{dataset_name}] HVG gene file not found: {gene_path}")
        print(f"  Run select_hvg.py first!")
        return

    gene_names = np.load(gene_path, allow_pickle=True).tolist()
    print(f"\n[{dataset_name}] Encoding {len(gene_names)} HVG genes...")

    embeddings = encode_genes(gene_names, model, tokenizer, device)
    print(f"[{dataset_name}] Embedding shape: {embeddings.shape}")

    np.save(output_path, embeddings)
    print(f"[{dataset_name}] Saved to: {output_path}")

    loaded = np.load(output_path)
    assert np.array_equal(embeddings, loaded), "Verification failed!"
    print(f"[{dataset_name}] Verification passed.")


def main():
    parser = argparse.ArgumentParser(description="Encode HVG genes with loki (OmiCLIP)")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['cSCC', 'HER2', 'HBC', 'all'])
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help="Path to OmiCLIP checkpoint")
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    args = parser.parse_args()

    datasets = ['cSCC', 'HER2', 'HBC'] if args.dataset == 'all' else [args.dataset]

    print("=" * 60)
    print("Encoding HVG genes with loki (OmiCLIP COCA ViT-L-14)")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Datasets: {datasets}")
    print("=" * 60)

    print("\nLoading OmiCLIP model...")
    model, preprocess, tokenizer = load_model(args.model_path, args.device)
    model.eval()
    print("Model loaded.")

    for ds in datasets:
        process_dataset(ds, model, tokenizer, args.device)

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
