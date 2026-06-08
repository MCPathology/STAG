#!/bin/bash
# =================================================================
# Run STAG (Ours) HVG experiments on cSCC, HER2, HBC
# 三个数据集分别跑在不同 GPU 上 (并行)
# =================================================================

echo "=========================================="
echo "Starting HVG experiments for STAG (Ours)"
echo "=========================================="

# --- cSCC: 4-fold ---
CUDA_VISIBLE_DEVICES=0 python train_STAG_hvg.py \
    --data_name cSCC \
    --k_folds 4 \
    --epochs 80 \
    --lr 1e-4 \
    --batch_size 8 \
    --gene_mode hvg \
    --warmup_epochs 5 \
    --patience 20 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    > logs_hvg_cSCC.log 2>&1 &

PID_CSCC=$!
echo "[cSCC] Started on GPU 0, PID=$PID_CSCC"

# --- HER2: 6-fold ---
CUDA_VISIBLE_DEVICES=1 python train_STAG_hvg.py \
    --data_name HER2 \
    --k_folds 6 \
    --epochs 80 \
    --lr 1e-4 \
    --batch_size 8 \
    --gene_mode hvg \
    --warmup_epochs 5 \
    --patience 20 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    > logs_hvg_HER2.log 2>&1 &

PID_HER2=$!
echo "[HER2] Started on GPU 1, PID=$PID_HER2"

# --- HBC: 9-fold ---
CUDA_VISIBLE_DEVICES=2 python train_STAG_hvg.py \
    --data_name HBC \
    --k_folds 9 \
    --epochs 80 \
    --lr 1e-4 \
    --batch_size 8 \
    --gene_mode hvg \
    --warmup_epochs 5 \
    --patience 20 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    > logs_hvg_HBC.log 2>&1 &

PID_HBC=$!
echo "[HBC]  Started on GPU 2, PID=$PID_HBC"

echo ""
echo "All 3 experiments launched in parallel."
echo "Monitor logs:"
echo "  tail -f logs_hvg_cSCC.log"
echo "  tail -f logs_hvg_HER2.log"
echo "  tail -f logs_hvg_HBC.log"
echo ""

# 等待全部完成
wait $PID_CSCC
echo "[cSCC] Done (exit code: $?)"
wait $PID_HER2
echo "[HER2] Done (exit code: $?)"
wait $PID_HBC
echo "[HBC]  Done (exit code: $?)"

echo ""
echo "=========================================="
echo "All HVG experiments completed!"
echo "=========================================="
