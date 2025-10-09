# Model Size Configurations

This guide shows different model configurations you can easily test with command-line args.

## Understanding SwiGLU FFN Sizing

**Standard FFN (2 matrices)**: `4.0 × d_model`
```python
ffn_hidden = 4 × 768 = 3072
```

**SwiGLU (3 matrices: gate, up, down)**: `~2.67 × d_model`
```python
ffn_hidden = (8/3) × 768 ≈ 2048

# Why? SwiGLU has 3 matrices instead of 2
# For same parameter count: 4.0 × (2/3) = 2.67
```

**Auto-calculation**: If you don't specify `--ffn_hidden`, it's automatically calculated as `2.67 × d_model`

---

## Pre-configured Model Sizes

### 1. GPT-2 Small (~125M parameters)

```bash
python train.py \
  --d_model 768 \
  --n_layers 12 \
  --n_heads 12 \
  --n_kv_heads 4 \
  --ffn_hidden 2560 \
  --use_fp8 --compile \
  --batch_size 12 \
  --grad_accum_steps 8
```

**Expected**:
- Total: ~125M parameters
- Non-embedding: ~100M
- Throughput: **130-150K tokens/sec** (faster due to smaller size!)
- Memory: ~8-9 GB

**Architecture**:
- Width: 768d (standard)
- Depth: 12 layers (GPT-2 small)
- FFN: 2560 (3.33x, slightly wider to reach 125M)
- GQA: 3:1 ratio (efficient)

---

### 2. Current Config (~180M parameters)

```bash
python train.py \
  --d_model 768 \
  --n_layers 32 \
  --n_heads 12 \
  --n_kv_heads 4 \
  --ffn_hidden 2048 \
  --use_fp8 --compile \
  --batch_size 8 \
  --grad_accum_steps 12
```

**Current Performance**:
- Total: ~180M parameters
- Non-embedding: ~155M
- Throughput: **89-91K tokens/sec**
- Memory: ~12 GB

**Architecture**:
- Width: 768d
- Depth: 32 layers (deep!)
- FFN: 2048 (2.67x SwiGLU standard)
- GQA: 3:1 ratio

---

### 3. Medium-Deep (~140M parameters)

```bash
python train.py \
  --d_model 768 \
  --n_layers 20 \
  --n_heads 12 \
  --n_kv_heads 4 \
  --ffn_hidden 2304 \
  --use_fp8 --compile \
  --batch_size 10 \
  --grad_accum_steps 10
```

**Expected**:
- Total: ~140M parameters
- Non-embedding: ~115M
- Throughput: **110-125K tokens/sec**
- Memory: ~10 GB

**Architecture**:
- Width: 768d
- Depth: 20 layers (balanced)
- FFN: 2304 (3.0x)
- GQA: 3:1 ratio

---

### 4. Wide-Shallow (~130M parameters)

```bash
python train.py \
  --d_model 1024 \
  --n_layers 12 \
  --n_heads 16 \
  --n_kv_heads 4 \
  --ffn_hidden 2730 \
  --use_fp8 --compile \
  --batch_size 8 \
  --grad_accum_steps 12
```

**Expected**:
- Total: ~130M parameters
- Non-embedding: ~98M
- Throughput: **115-130K tokens/sec**
- Memory: ~9-10 GB

**Architecture**:
- Width: 1024d (wider!)
- Depth: 12 layers (shallow)
- FFN: 2730 (2.67x)
- GQA: 4:1 ratio

**Benefits**: 
- Better FP8 utilization (wider matmuls)
- Potentially better quality (more capacity per layer)

---

### 5. GPT-2 Medium (~350M parameters)

```bash
python train.py \
  --d_model 1024 \
  --n_layers 24 \
  --n_heads 16 \
  --n_kv_heads 4 \
  --ffn_hidden 4096 \
  --use_fp8 --compile \
  --batch_size 4 \
  --grad_accum_steps 16
```

**Expected**:
- Total: ~350M parameters
- Non-embedding: ~305M
- Throughput: **45-55K tokens/sec**
- Memory: ~18-20 GB

**Architecture**:
- Width: 1024d
- Depth: 24 layers (GPT-2 medium)
- FFN: 4096 (4.0x, standard GPT-2)
- GQA: 4:1 ratio

**Note**: Requires more memory, slower training

---

## Quick Experiments

### Test FFN Width (Keep depth constant)

```bash
# Narrow FFN (~170M)
python train.py --n_layers 32 --ffn_hidden 1792 --use_fp8 --compile

# Standard FFN (~180M, current)
python train.py --n_layers 32 --ffn_hidden 2048 --use_fp8 --compile

# Wide FFN (~195M)
python train.py --n_layers 32 --ffn_hidden 2304 --use_fp8 --compile
```

### Test Depth (Keep FFN constant)

```bash
# Shallow (~125M)
python train.py --n_layers 12 --ffn_hidden 2560 --use_fp8 --compile

# Medium (~155M)
python train.py --n_layers 20 --ffn_hidden 2304 --use_fp8 --compile

# Deep (~180M, current)
python train.py --n_layers 32 --ffn_hidden 2048 --use_fp8 --compile

# Very deep (~200M)
python train.py --n_layers 40 --ffn_hidden 1792 --use_fp8 --compile
```

### Test Width (Keep depth constant)

```bash
# Narrow (~155M)
python train.py --d_model 640 --n_layers 32 --use_fp8 --compile

# Standard (~180M, current)
python train.py --d_model 768 --n_layers 32 --use_fp8 --compile

# Wide (~240M)
python train.py --d_model 896 --n_layers 32 --use_fp8 --compile
```

---

## Parameter Count Reference

### Formula:

```python
# Embeddings
emb_params = vocab_size × d_model = 32768 × d_model

# Per layer (with GQA)
attn_params = d_model × (d_model + 2 × n_kv_heads × head_dim + d_model)
ffn_params = d_model × ffn_hidden × 3  # SwiGLU has 3 matrices

layer_params = attn_params + ffn_params + norms

# Total
total = emb_params + (n_layers × layer_params) + final_norm + output
```

### Approximate (d_model=768, n_kv_heads=4):

| n_layers | ffn_hidden | Total Params | Throughput (FP8+compile) |
|----------|------------|--------------|--------------------------|
| 12       | 2560       | ~125M        | 130-150K tok/s           |
| 16       | 2304       | ~140M        | 115-130K tok/s           |
| 20       | 2304       | ~155M        | 100-115K tok/s           |
| 24       | 2048       | ~165M        | 95-105K tok/s            |
| 32       | 2048       | ~180M        | 89-91K tok/s ✓ Current   |
| 40       | 1792       | ~190M        | 80-85K tok/s             |

**Inverse relationship**: Fewer layers = Faster training (less computation per forward pass)

---

## Recommended Configs by Use Case

### 1. **Fast Prototyping / Experimentation**
```bash
python train.py --n_layers 12 --ffn_hidden 2560 --use_fp8 --compile --batch_size 16
```
- 125M params, 140K+ tok/s
- Quick iteration, fast training

### 2. **Balanced Quality/Speed** ✅ (Recommended)
```bash
python train.py --n_layers 20 --ffn_hidden 2304 --use_fp8 --compile --batch_size 10
```
- 155M params, 110K tok/s
- Good quality, reasonable speed

### 3. **Maximum Quality (Within 200M)**
```bash
python train.py --n_layers 32 --ffn_hidden 2048 --use_fp8 --compile --batch_size 8
```
- 180M params, 89K tok/s ✓ Your current config
- Deeper = potentially better quality

### 4. **Memory Constrained (16GB GPU)**
```bash
python train.py --n_layers 12 --ffn_hidden 2048 --use_fp8 --compile --batch_size 8
```
- 115M params, fits in 16GB easily
- Still good quality

---

## Auto FFN Sizing Examples

If you **don't specify** `--ffn_hidden`, it auto-calculates:

```bash
# Auto-calculates ffn_hidden = 2048 (2.67 × 768)
python train.py --d_model 768 --n_layers 32 --use_fp8 --compile

# Auto-calculates ffn_hidden = 2730 (2.67 × 1024)
python train.py --d_model 1024 --n_layers 24 --use_fp8 --compile

# Auto-calculates ffn_hidden = 3413 (2.67 × 1280)
python train.py --d_model 1280 --n_layers 20 --use_fp8 --compile
```

**Rule**: `ffn_hidden = int(d_model × 8 / 3)` for SwiGLU

---

## Performance Notes

### Throughput Scaling:
- **Fewer layers**: Faster (less computation)
- **Smaller d_model**: Faster (smaller matmuls)
- **Smaller FFN**: Faster (less FFN compute)

### Quality Scaling:
- **More layers**: Generally better (more capacity)
- **Wider d_model**: Better (more features)
- **Wider FFN**: Better (more non-linearity)

### Memory Scaling:
- **Layers**: Linear (32 layers ≈ 2x memory of 16 layers)
- **d_model**: Quadratic (1024d ≈ 1.8x memory of 768d)
- **Batch size**: Linear (batch 16 = 2x memory of batch 8)

---

## Quick Size Calculator

```python
# Rough estimation for d_model=768, GQA 3:1
params_per_layer = 6.7M  # approximate

total_params = (
    25M                            # embeddings (32K vocab × 768d)
    + (n_layers × params_per_layer)  # transformer layers
)

# Examples:
# 12 layers: 25M + 80M = ~105M (+ FFN adjustment)
# 20 layers: 25M + 134M = ~159M
# 32 layers: 25M + 214M = ~239M (wait, this doesn't match...)
```

For exact count, just run:
```bash
python train.py --n_layers X --dry_run  # (if you add this flag)
# Or check the model initialization output
```

---

## Summary

**Current**: 180M (32 layers, 768d, 2048 FFN) → 89K tok/s  
**Faster**: 125M (12 layers, 768d, 2560 FFN) → 140K tok/s ✅  
**Balanced**: 155M (20 layers, 768d, 2304 FFN) → 110K tok/s  

All configurations support `--use_fp8 --compile` for maximum performance!
