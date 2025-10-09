# Dense 180M LLM Pretraining

A modern dense transformer model (~180-200M parameters) with state-of-the-art optimizations for efficient pretraining.

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
pip install -r requirements.txt
```

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

### Test Model

```bash
# Test model architecture
python model.py

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
- Streaming dataset (no disk space needed)
- On-the-fly tokenization
- Distributed data splitting
- Shuffle buffer to avoid temporal bias
- 8 persistent workers for throughput
- Non-blocking GPU transfers

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

## Checkpoints

Checkpoints are saved to `./checkpoints/` by default:
- Every 5000 steps: `checkpoint_step_5000.pt`
- Final model: `final_model.pt`

Each checkpoint contains:
- Model weights
- Optimizer state
- Step count
- Tokens seen
- Training args

## Configuration Options

```bash
# Model
--max_seq_len 2048

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
--save_interval 5000
--checkpoint_dir ./checkpoints
--wandb_project "dense-llm-pretraining"
--run_name "dense-180m"
--no_wandb  # Disable WandB
```

## Notes

- The model uses BF16 by default (requires Ampere+ GPUs)
- Flash Attention requires PyTorch 2.0+
- Dataset streams from HuggingFace (no local download needed)
- Eval set uses different seed from training set
- QK normalization allows 5x higher learning rate (3e-3)

## License

MIT
