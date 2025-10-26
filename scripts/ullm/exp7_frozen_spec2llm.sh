#!/bin/bash
# EXP7: Frozen LLM (Spec2LLM Approach)
# Baseline: MSE=0.5315 @ LoRA (29.5% trainable)
# Change: Freeze entire LLM backbone (0% LLM params trainable)
# This implements Spec2LLM's core idea: frozen LLM with only adapter training

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  EXP7: Frozen LLM (Spec2LLM Approach)                     ║"
echo "║  Testing: Can frozen LLM compete with LoRA fine-tuning?   ║"
echo "╚════════════════════════════════════════════════════════════╝"

export CUDA_VISIBLE_DEVICES=0,1,2
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python3 scripts/ullm/train_forecasting.py \
    --dataset etth1 \
    --n_pred_steps 96 \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --d_model 512 \
    --d_llm 768 \
    --n_heads 16 \
    --patch_size 16 \
    --llm_model gpt2 \
    --n_llm_layers_to_tune 0 \
    --llm_layer_selection last \
    --use_lora false \
    --patience 10 \
    --saving_path outputs/unifiedllm/exp7_frozen_spec2llm \
    2>&1 | tee /tmp/exp7_frozen_spec2llm.log

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ EXP7 (Frozen LLM - Spec2LLM Approach) Complete!"
echo ""
echo "Results:"
grep -E "MSE:|MAE:|RMSE:" /tmp/exp7_frozen_spec2llm.log | tail -3
echo ""
echo "Compare to:"
echo "  - Spec2LLM (prior work): Frozen LLM with spectral features"
echo "  - Baseline (this work):  LoRA fine-tuning, MSE=0.5315"
echo "  - Full FT (this work):   Full fine-tuning, MSE=0.7398"
echo ""
echo "This tests if freezing LLM (Spec2LLM's approach) is competitive"
echo "════════════════════════════════════════════════════════════"
