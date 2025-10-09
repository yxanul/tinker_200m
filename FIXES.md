# Critical Data Pipeline Fixes

## Problem Summary

The initial implementation had **data repetition** issues causing artificially fast loss convergence:
- Loss dropped from ~10 to ~3 in just 100 steps
- Workers were processing the same data
- Train and eval sets had overlapping documents

## Root Causes

### 1. Worker Sharding Bug
**Problem**: Using `split_dataset_by_node()` for worker sharding didn't work correctly - workers were seeing overlapping or identical data.

**Solution**: Implemented manual worker filtering in `__iter__()`:
```python
# Each worker processes every num_workers-th item
if num_workers > 1 and items_processed % num_workers != worker_id:
    items_processed += 1
    continue
```

**Result**: Each worker now gets a unique, non-overlapping shard of data.

### 2. Dataset Initialization Timing
**Problem**: Dataset was initialized in `__iter__()` or `_initialize_dataset()`, causing:
- Multiple HuggingFace API calls per worker (rate limit issues)
- Potential race conditions

**Solution**: Initialize dataset ONCE in `__init__()`:
```python
def __init__(self, ...):
    # Initialize dataset ONCE here
    self._base_dataset = load_dataset(..., streaming=True)
```

**Result**: Single API call, no rate limits, cleaner code.

### 3. Train/Eval Data Leakage
**Problem**: Both train and eval loaders started at offset 0 with different seeds, but could still sample overlapping documents.

**Solution**: Implemented disjoint split with `take_first` parameter:
```python
# Eval: First N documents
eval_dataset = StreamingTextDataset(skip_first=0, take_first=eval_take)

# Train: Everything after first N documents  
train_dataset = StreamingTextDataset(skip_first=eval_take, take_first=None)
```

**Result**: Zero overlap between train and eval sets.

### 4. Token Wastage
**Problem**: Using `truncation=True, padding="max_length"` wasted tokens and created artificial document boundaries.

**Solution**: Continuous tokenization with buffer:
```python
# Tokenize without truncation
tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]
tokens.append(tokenizer.eos_token_id)  # Mark document boundary
token_buffer.extend(tokens)

# Yield complete chunks
while len(token_buffer) >= max_length:
    chunk = token_buffer[:max_length]
    token_buffer = token_buffer[max_length:]
    yield {"input_ids": torch.tensor(chunk), "labels": ...}
```

**Result**: 
- No wasted padding tokens within documents
- Natural document boundaries with EOS tokens
- Maximum token utilization

## Changes Made

### data.py
1. ✅ Added `take_first` parameter for disjoint splits
2. ✅ Moved dataset initialization to `__init__()`
3. ✅ Implemented manual worker sharding (replaced buggy `split_dataset_by_node`)
4. ✅ Added token buffer for continuous tokenization
5. ✅ Removed padding/truncation in tokenization
6. ✅ Added EOS tokens for document boundaries
7. ✅ Updated `create_dataloaders()` for disjoint train/eval

### train.py
1. ✅ Added `--eval_take` argument (default: 10,000 documents)
2. ✅ Updated `setup_data()` to pass `eval_take` parameter
3. ✅ Added logging for train/eval split configuration
4. ✅ Added comment clarifying gradient clipping behavior

### verify_data_diversity.py
1. ✅ Updated to use new `eval_take` parameter

### README.md
1. ✅ Added detailed data pipeline documentation
2. ✅ Explained worker sharding mechanism
3. ✅ Documented train/eval split strategy
4. ✅ Added training dynamics section
5. ✅ Explained gradient norm behavior

## Expected Training Behavior (After Fixes)

### Loss Curve
```
Step 1-50:    Loss ~10 → ~6   (basic statistics)
Step 50-500:  Loss ~6 → ~4    (common patterns)
Step 500-5K:  Loss ~4 → ~2.5  (gradual learning)
Step 5K+:     Loss ~2.5+      (slow convergence)
```

### Gradient Norms
```
Step 1-100:   GradNorm 5-30   (normal during warmup)
Step 100-1K:  GradNorm 2-5    (decreasing)
Step 1K+:     GradNorm 1-3    (stabilized)
```

**Note**: Reported grad norm is **pre-clipping** (PyTorch behavior). Gradients ARE clipped to 1.0.

## Verification

Run this to verify data diversity:
```bash
python verify_data_diversity.py
```

**Expected output**:
- 0% duplicate rate across 100 batches
- Unique tokens across samples
- No repeated sequences

## Configuration

### Recommended Settings

**Single GPU (32GB)**:
```bash
python train.py \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --num_workers 8 \
  --eval_take 10000
```

**Single GPU (24GB)** (if OOM):
```bash
python train.py \
  --batch_size 4 \
  --grad_accum_steps 16 \
  --num_workers 8 \
  --eval_take 10000
```

**Multi-GPU (8x GPUs)**:
```bash
torchrun --nproc_per_node=8 train.py \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --num_workers 8 \
  --eval_take 10000
```

### Key Parameters

- `--eval_take`: Number of documents for eval set (default: 10,000)
  - Eval uses first N documents
  - Train uses everything after N
  - Increase if you want larger eval set
  
- `--num_workers`: Number of data loading workers (default: 8)
  - Each worker gets unique data shard
  - More workers = better CPU utilization
  - 8 workers recommended for GPU training

- `--buffer_size`: Shuffle buffer size (default: 10,000)
  - Larger = more randomness
  - Memory overhead is minimal (just document indices)

## Testing

### 1. Test Data Pipeline
```bash
python data.py
```
Should show diverse tokens across train and eval batches.

### 2. Test Data Diversity
```bash
python verify_data_diversity.py
```
Should show 0% duplicate rate.

### 3. Test Model
```bash
python model.py
```
Should show ~180-200M parameters.

### 4. Start Training
```bash
python train.py --batch_size 4 --grad_accum_steps 12
```
Should see gradual loss decrease, not instant drop.

## Summary

These fixes ensure:
1. ✅ **No data repetition** - each worker gets unique data
2. ✅ **No train/eval leakage** - disjoint document sets
3. ✅ **Proper tokenization** - continuous chunks, no padding waste
4. ✅ **Correct gradient clipping** - norms are pre-clip values
5. ✅ **Realistic training** - gradual loss convergence

The previous fast loss drop was **not** because the model was good, but because it was overfitting to repeated batches. With these fixes, you'll see realistic training dynamics.
