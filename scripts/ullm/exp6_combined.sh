#!/bin/bash
# EXP6: Combined Best Approaches
# Combine: Higher LR + More layers + Larger patches

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  EXP6: Combined Best (LR+Layers+Patches)                  ║"
echo "║  Baseline MSE: 0.5315                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"

export CUDA_VISIBLE_DEVICES=0,1,2
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python3 scripts/ullm/train_forecasting.py \
    --dataset etth1 \
    --n_pred_steps 96 \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --d_model 512 \
    --d_llm 768 \
    --n_heads 16 \
    --patch_size 32 \
    --llm_model gpt2 \
    --n_llm_layers_to_tune 12 \
    --llm_layer_selection distributed \
    --lora_r 32 \
    --lora_alpha 64 \
    --patience 30 \
    --saving_path results/unifiedllm/exp6_combined

echo "✅ EXP6 completed!"
