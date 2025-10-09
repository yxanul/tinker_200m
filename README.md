# Dense 180M LLM Pretraining

A modern dense transformer model (~180-200M parameters) with state-of-the-art optimizations for efficient pretraining.

## 🚀 NEW: FP8 Training Support

**FP8 training** now available for H100+ GPUs! Get **1.5-2x speedup** with minimal accuracy loss.

```bash
# Enable FP8 training (requires H100+ GPU)
python train.py --use_fp8 --batch_size 16

# Multi-GPU with FP8
torchrun --nproc_per_node=8 train.py --use_fp8
```

See [FP8_UPGRADE.md](FP8_UPGRADE.md) for details.

## ⚠️ CRITICAL FIX: Label Shifting

**If you cloned before this fix**: The model was learning to copy tokens instead of predicting next tokens, causing artificially fast loss drop (10→1.8 in 120 steps).

**Fixed**: Model now properly shifts logits and labels for true next-token prediction. See [CRITICAL_BUG_LABEL_SHIFT.md](CRITICAL_BUG_LABEL_SHIFT.md) for details.

## Model Architecture

- **Parameters**: ~180-200M (dense, all active)
- **Layers**: 32 deep layers
- **Hidden dim**: 768
- **Attention**: 12 query heads, 4 KV heads (Grouped Query Attention)
- **FFN**: 2048 hidden (SwiGLU activation)
- **Context**: 2048 tokens
- **Vocabulary**: 32,768 (Mistral tokenizer)

### Modern Optimizations

- ✅ RMSNorm (15% faster than LayerNorm)
- ✅ RoPE (Rotary Position Embeddings)
- ✅ Grouped Query Attention (3:1 ratio)
- ✅ QK Normalization for training stability
- ✅ Flash Attention 2 (2-3x faster)
- ✅ SwiGLU activation
- ✅ No bias terms (Llama-style)
- ✅ Weight tying (input/output embeddings)
- ✅ BF16 mixed precision
- ✅ **FP8 training** (NEW! 1.5-2x faster on H100+)

## Training Configuration

- **Target**: ~6B tokens in 8-10 hours on single GPU
- **Optimizer**: AdamW (lr=3e-3, β=(0.9, 0.95), wd=0.1)
- **Schedule**: Cosine decay with 2K warmup steps
- **Batch size**: 16 per GPU, 4 grad accumulation steps
- **Effective batch**: 64 samples (131K tokens)
- **Dataset**: FineWeb-EDU (streaming)
- **Precision**: BF16

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: For FP8 training on H100+ GPUs (1.5-2x speedup)
pip install transformer-engine
```

**Note**: FP8 training requires:
- NVIDIA H100 or newer GPU
- CUDA >= 11.8
- transformer-engine >= 1.0.0

## Usage

### Single GPU Training

```bash
python train.py \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --total_steps 30000 \
  --learning_rate 3e-3 \
  --warmup_steps 2000 \
  --num_workers 8 \
  --wandb_project "my-dense-llm" \
  --run_name "dense-180m-run1"
```

### Multi-GPU Training (DDP)

```bash
# 2 GPUs
torchrun --nproc_per_node=2 train.py \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --total_steps 30000 \
  --learning_rate 3e-3 \
  --warmup_steps 2000

# 8 GPUs
torchrun --nproc_per_node=8 train.py \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --total_steps 30000 \
  --learning_rate 3e-3 \
  --warmup_steps 2000
```

### FP8 Training (H100+ GPUs)

```bash
# Single GPU with FP8
python train.py \
  --use_fp8 \
  --batch_size 16 \
  --grad_accum_steps 4

# Multi-GPU with FP8 (8x H100)
torchrun --nproc_per_node=8 train.py \
  --use_fp8 \
  --batch_size 16 \
  --grad_accum_steps 4
```

**Benefits**:
- 1.5-2x faster training
- ~50% memory reduction for activations
- Minimal accuracy impact (<1%)

See [FP8_UPGRADE.md](FP8_UPGRADE.md) for full documentation.

### Test Model

```bash
# Test model architecture (BF16)
python model.py

# Test model with FP8
python model.py --fp8

# Test dataloader
python data.py
```

## Project Structure

```
.
├── model.py          # Dense transformer implementation
├── data.py           # Streaming dataset with distributed support
├── train.py          # Training loop with DDP and WandB
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Key Features

### Model (model.py)
- Dense transformer with all modern optimizations
- ~180-200M parameters
- Efficient GQA implementation
- Flash Attention support
- Proper initialization

### Data Pipeline (data.py)
- **Streaming dataset** (no disk space needed, infinite iteration)
- **On-the-fly tokenization** with continuous token buffering
- **Offset-based train/eval separation** (minimal overlap probability)
  - Train: starts from document 0, seed 42
  - Eval: starts from document 100K, seed 1041
  - Both are infinite streams (no iterator exhaustion)
- **Proper worker sharding** (manual filtering per worker, no overlap)
- **Distributed data splitting** (multi-GPU support)
- **Shuffle buffer** to avoid temporal bias
- **8 persistent workers** for throughput
- **Non-blocking GPU transfers**
- **Fresh dataset loading** per `__iter__()` call (prevents data repetition)

### Training (train.py)
- Multi-GPU support (DDP)
- BF16 mixed precision
- Gradient accumulation
- Cosine LR schedule with warmup
- WandB integration
- Checkpointing
- Evaluation loop

## Monitoring

The training script logs:
- **Loss** (train & eval)
- **Perplexity** (train & eval)
- **Learning rate**
- **Gradient norm**
- **Tokens/second**
- **Total tokens seen**

## Expected Performance

- **Single GPU**: ~8-10 hours for 6B tokens
- **Throughput**: ~600K-800K tokens/sec on A100
- **Memory**: ~12-16GB GPU memory with BF16
- **Final loss**: ~2.5-3.0 (depending on data quality)

## Memory Optimization

If you encounter OOM errors, try these in order:

1. **Reduce batch size** (adjust grad_accum to maintain effective batch):
```bash
# Example: 32GB GPU
python train.py --batch_size 8 --grad_accum_steps 8  # Effective batch = 64

# Example: 24GB GPU
python train.py --batch_size 4 --grad_accum_steps 16  # Effective batch = 64

# Example: 16GB GPU
python train.py --batch_size 2 --grad_accum_steps 32  # Effective batch = 64
```

2. **Reduce sequence length**:
```bash
python train.py --batch_size 16 --max_seq_len 1024
```

3. **Reduce num_workers** (saves CPU memory):
```bash
python train.py --num_workers 4
```

4. **Enable gradient checkpointing** (trade compute for memory - requires code modification)

## Checkpoints

Checkpoints are saved to `./checkpoints/` by default:
- **Every 500 steps**: `checkpoint_step_500.pt`, `checkpoint_step_1000.pt`, etc.
- **Best model** (lowest eval loss): `best_model.pt` (updated whenever eval loss improves)
- **Final model**: `final_model.pt` (saved at end of training)

Each checkpoint contains:
- Model weights
- Optimizer state
- Step count
- Tokens seen
- Best eval loss
- Training args

### Loading Checkpoints

```python
import torch
from model import create_model

# Load best model
checkpoint = torch.load("checkpoints/best_model.pt")
model = create_model()
model.load_state_dict(checkpoint["model"])

print(f"Loaded model from step {checkpoint['step']}")
print(f"Best eval loss: {checkpoint['best_eval_loss']:.4f}")
```

## Configuration Options

```bash
# Model
--max_seq_len 2048
--use_fp8  # Enable FP8 training (H100+ GPU, 1.5-2x speedup)

# Training
--batch_size 16
--grad_accum_steps 4
--total_steps 30000
--learning_rate 3e-3
--warmup_steps 2000
--weight_decay 0.1
--grad_clip 1.0

# Data
--num_workers 8
--buffer_size 10000
--seed 42

# Logging
--log_interval 10
--eval_interval 500
--eval_batches 50
--save_interval 500  # Save checkpoint every 500 steps + best model
--checkpoint_dir ./checkpoints
--wandb_project "dense-llm-pretraining"
--run_name "dense-180m"
--no_wandb  # Disable WandB
```

## Training Dynamics

### Gradient Norms
The logged `GradNorm` is the **pre-clipping** gradient norm (standard PyTorch behavior). High values (>1.0) are expected, especially early in training:
- Steps 1-100: Often 5-30 (normal during warmup)
- Steps 100-1000: Gradually decreases to 2-5
- Steps 1000+: Stabilizes around 1-3

**Important**: Gradients ARE being clipped to 1.0 despite high reported norms - this is how `clip_grad_norm_` works.

### Loss Dynamics

**After the label shift fix**, expect realistic, gradual learning:

- **Step 1-100**: Loss ~10 → ~8 (learning token distributions)
- **Step 100-500**: Loss ~8 → ~5 (basic patterns emerge)
- **Step 500-2000**: Loss ~5 → ~4 (understanding sequences)
- **Step 2000-10000**: Loss ~4 → ~3 (language understanding)
- **Step 10000+**: Loss ~2.5-3.0 (converging)

**Warning**: If you see loss drop to <3.0 before step 1000, something is wrong:
- Check that model.py has label shifting (`shift_logits = logits[..., :-1, :]`)
- Verify data diversity with `verify_data_diversity.py`
- Loss ~1.5-2.0 in <500 steps = model is cheating (copying tokens)

### Data Diversity

**Critical Fix Applied**: The data pipeline uses **infinite streams** to prevent iterator exhaustion and data repetition:

1. **Infinite streaming** (no `take()` operation)
   - Both train and eval are infinite streams
   - Workers never exhaust their data source
   - Fresh dataset loaded in each `__iter__()` call
   - Prevents the bug: `take(N)` + `persistent_workers` = data loop

2. **Manual worker filtering**: Each worker processes every Nth item (where N = num_workers)
   - Worker 0: items 0, 8, 16, 24, ...
   - Worker 1: items 1, 9, 17, 25, ...
   - Worker 7: items 7, 15, 23, 31, ...

3. **Token buffering**: Documents are tokenized continuously without padding truncation
   - Prevents wasteful padding within documents
   - EOS tokens mark document boundaries
   - Chunks are exactly `max_length` tokens

4. **Offset-based train/eval separation**: Different starting positions
   - Train starts from document 0 (seed 42)
   - Eval starts from document 100K (seed 1041)
   - Minimal overlap probability with shuffle buffers

**Verification**:
```bash
python verify_data_diversity.py
```
This checks for duplicate batches and ensures workers are producing diverse data.

**Expected**: 0% duplicate rate across 100+ batches with 4-8 workers.

**Root Cause**: Previously used `take()` which created finite iterators that exhausted and restarted from the same documents. Now using infinite streams that never repeat.

## Notes

- The model uses BF16 by default (requires Ampere+ GPUs)
- Flash Attention requires PyTorch 2.0+
- Dataset streams from HuggingFace (no local download needed)
- Eval set uses different seed from training set
- QK normalization allows 5x higher learning rate (3e-3)
- Each worker gets a unique shard of data (no repetition)

## License

MIT
