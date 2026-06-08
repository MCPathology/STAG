#!/bin/bash
# Encoder Ablation on cSCC — 3 single-replacement experiments, sequential on GPU 7
# Baseline (ResNet18 + MLP) results already available from main experiments
# Text Encoder (OmiCLIP/Loki) unchanged throughout

UNI_CKPT="./pretrained/uni/pytorch_model.bin"
CONCH_CKPT="./pretrained/conch/pytorch_model.bin"

# Image encoder ablation: ResNet18 → UNI
CUDA_VISIBLE_DEVICES=7 python ablation_encoder_cSCC.py --img_encoder uni --exp_encoder mlp --uni_ckpt $UNI_CKPT --epochs 50 --batch_size 8 --k_folds 5

# Image encoder ablation: ResNet18 → Conch
CUDA_VISIBLE_DEVICES=7 python ablation_encoder_cSCC.py --img_encoder conch --exp_encoder mlp --conch_ckpt $CONCH_CKPT --epochs 50 --batch_size 8 --k_folds 5

# Gene encoder ablation: MLP → SNN
CUDA_VISIBLE_DEVICES=7 python ablation_encoder_cSCC.py --img_encoder resnet18 --exp_encoder snn --epochs 50 --batch_size 8 --k_folds 5

echo "All encoder ablation experiments finished."
echo "Results in ./ablation_encoder_results/:"
echo "  encoder_ablation_uni_mlp_summary.csv       (UNI replaces ResNet18)"
echo "  encoder_ablation_conch_mlp_summary.csv     (CONCH replaces ResNet18)"
echo "  encoder_ablation_resnet18_snn_summary.csv  (SNN replaces MLP)"
