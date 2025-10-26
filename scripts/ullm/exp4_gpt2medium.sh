#!/bin/bash
# EXP4: GPT-2 Medium
# Baseline: MSE=0.5315 @ gpt2 (124M)
# Change: gpt2-medium (355M)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  EXP4: GPT-2 Medium (Larger Model)                        ║"
echo "║  Baseline MSE: 0.5315 @ gpt2 (124M params)                ║"
echo "╚════════════════════════════════════════════════════════════╝"

export CUDA_VISIBLE_DEVICES=0,1,2
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python3 scripts/ullm/train_forecasting.py \
    --dataset etth1 \
    --n_pred_steps 96 \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --d_model 512 \
    --d_llm 1024 \
    --n_heads 16 \
    --patch_size 16 \
    --llm_model gpt2-medium \
    --n_llm_layers_to_tune 6 \
    --llm_layer_selection distributed \
    --lora_r 32 \
    --lora_alpha 64 \
    --patience 30 \
    --saving_path results/unifiedllm/exp4_gpt2medium

echo "✅ EXP4 completed!"
