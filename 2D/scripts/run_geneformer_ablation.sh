#!/bin/bash
# Geneformer 消融实验：全转录组 vs HEG-250
# 用法: bash run_geneformer_ablation.sh

set -e

DATA_PATH="./data/GSE144240"
GF_MODEL_DIR="./Geneformer"
MODEL_VARIANT="v1-10m"
GF_DIM=256
SELECTED_GENES="select_genes/cSCC_Selected_Genes.npy"

# ========== 实验1: Geneformer + 全转录组 ==========
GF_EMB_FULL="${DATA_PATH}/geneformer_emb"

echo "===== 实验1: Geneformer + 全转录组 ====="
# 如果 embeddings 已存在则跳过预计算
if [ -f "${GF_EMB_FULL}/config.json" ]; then
    echo "全转录组 embeddings 已存在，跳过预计算"
else
    python preprocess/extract_geneformer_cell_embeddings.py \
        --data_path ${DATA_PATH} \
        --output_dir ${GF_EMB_FULL} \
        --model_dir ${GF_MODEL_DIR} \
        --model_variant ${MODEL_VARIANT} \
        --batch_size 64
fi

echo "--- 训练: Geneformer (全转录组) ---"
python ablation_encoder_cSCC.py \
    --img_encoder resnet18 \
    --exp_encoder geneformer \
    --geneformer_emb_dir ${GF_EMB_FULL} \
    --geneformer_dim ${GF_DIM}

# ========== 实验2: Geneformer + HEG-250 ==========
GF_EMB_HEG="${DATA_PATH}/geneformer_emb_heg250"

echo ""
echo "===== 实验2: Geneformer + HEG-250 ====="
python preprocess/extract_geneformer_cell_embeddings.py \
    --data_path ${DATA_PATH} \
    --output_dir ${GF_EMB_HEG} \
    --model_dir ${GF_MODEL_DIR} \
    --model_variant ${MODEL_VARIANT} \
    --selected_genes ${SELECTED_GENES} \
    --batch_size 64

echo "--- 训练: Geneformer (HEG-250) ---"
python ablation_encoder_cSCC.py \
    --img_encoder resnet18 \
    --exp_encoder geneformer \
    --geneformer_emb_dir ${GF_EMB_HEG} \
    --geneformer_dim ${GF_DIM}

echo ""
echo "===== 全部完成 ====="
