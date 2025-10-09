#!/bin/bash

# Single GPU training script for Dense 180M LLM

python train.py \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --total_steps 30000 \
  --learning_rate 3e-3 \
  --warmup_steps 2000 \
  --weight_decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --grad_clip 1.0 \
  --max_seq_len 2048 \
  --num_workers 8 \
  --buffer_size 10000 \
  --log_interval 10 \
  --eval_interval 500 \
  --eval_batches 50 \
  --save_interval 5000 \
  --checkpoint_dir ./checkpoints \
  --wandb_project "dense-llm-pretraining" \
  --run_name "dense-180m-$(date +%Y%m%d_%H%M%S)" \
  "$@"
