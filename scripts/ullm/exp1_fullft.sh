#!/bin/bash
# EXP1: Full Fine-Tuning (No LoRA)
# Baseline: MSE=0.5315 @ LoRA (6 distributed layers)
# Change: Remove LoRA, tune all layers

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  EXP1: Full Fine-Tuning (No LoRA)                         ║"
echo "║  Baseline MSE: 0.5315 | Tuning ALL layers (no LoRA)      ║"
echo "╚════════════════════════════════════════════════════════════╝"

export CUDA_VISIBLE_DEVICES=0,1,2
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python3 scripts/ullm/train_forecasting.py \
    --dataset etth1 \
    --n_pred_steps 96 \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.00005 \
    --d_model 512 \
    --d_llm 768 \
    --n_heads 16 \
    --patch_size 16 \
    --llm_model gpt2 \
    --n_llm_layers_to_tune 12 \
    --llm_layer_selection last \
    --lora_r 0 \
    --patience 30 \
    --saving_path results/unifiedllm/exp1_fullft

echo "✅ EXP1 completed!"
