# Checkpoint Improvements

## Changes Made

### 1. More Frequent Checkpoints
**Before**: Saved every 5000 steps  
**After**: Saved every 500 steps (10x more frequent)

**Why**: 
- 5000 steps = ~4.9M tokens per checkpoint
- If training crashes, you lose up to 5000 steps of progress
- 500 steps = ~491K tokens, much less loss if crash occurs

### 2. Best Model Tracking
**New feature**: Automatically saves `best_model.pt` when eval loss improves

**How it works**:
- Tracks `best_eval_loss` (initialized to infinity)
- After each evaluation (every 500 steps), checks if current eval loss < best
- If improved, saves checkpoint as `best_model.pt`
- Logs improvement with 🎯 emoji

**Benefits**:
- Always have the best performing model saved
- Don't need to manually track which checkpoint is best
- Can continue training and revert to best if overfitting occurs

### 3. Better Logging
**Console output now shows**:
```
============================================================
Evaluation at step 500
  Eval Loss: 4.7867
  Eval PPL: 119.90
  🎯 New best eval loss!
============================================================

💾 Best model checkpoint saved: ./checkpoints/best_model.pt
```

**WandB tracking includes**:
- `eval/loss`: Current eval loss
- `eval/perplexity`: Current eval perplexity  
- `eval/best_loss`: Best eval loss seen so far (NEW)

### 4. Enhanced Checkpoint Contents
Each checkpoint now includes:
- `model`: Model state dict
- `optimizer`: Optimizer state dict
- `step`: Global step count
- `tokens_seen`: Total tokens processed
- `best_eval_loss`: Best eval loss so far (NEW)
- `args`: Training arguments

## Checkpoint Files

After training, you'll have:

```
checkpoints/
├── checkpoint_step_500.pt      # Regular checkpoint
├── checkpoint_step_1000.pt     # Regular checkpoint
├── checkpoint_step_1500.pt     # Regular checkpoint
├── ...
├── best_model.pt              # Best model (lowest eval loss)
└── final_model.pt             # Model at end of training
```

**Note**: `best_model.pt` overwrites itself whenever eval loss improves, so you only keep one best model.

## Usage Examples

### Load Best Model

```python
import torch
from model import create_model

# Load the best checkpoint
checkpoint = torch.load("checkpoints/best_model.pt")
model = create_model()
model.load_state_dict(checkpoint["model"])

print(f"Loaded best model from step {checkpoint['step']}")
print(f"Best eval loss: {checkpoint['best_eval_loss']:.4f}")
print(f"Tokens seen: {checkpoint['tokens_seen'] / 1e9:.2f}B")
```

### Resume Training from Checkpoint

```python
import torch
from model import create_model

# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint_step_1000.pt")

# Restore model
model = create_model()
model.load_state_dict(checkpoint["model"])

# Restore optimizer
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(checkpoint["optimizer"])

# Resume from correct step
start_step = checkpoint["step"]
tokens_seen = checkpoint["tokens_seen"]
best_eval_loss = checkpoint["best_eval_loss"]

print(f"Resuming from step {start_step}")
print(f"Tokens processed: {tokens_seen / 1e9:.2f}B")
print(f"Best eval loss so far: {best_eval_loss:.4f}")

# Continue training...
```

### Compare Checkpoints

```python
import torch
import glob

# Load all checkpoints
checkpoint_files = glob.glob("checkpoints/checkpoint_step_*.pt")

results = []
for ckpt_path in sorted(checkpoint_files):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    results.append({
        "step": ckpt["step"],
        "tokens": ckpt["tokens_seen"] / 1e9,
        "best_eval_loss": ckpt.get("best_eval_loss", float('inf'))
    })

# Print progression
print("Training progression:")
print("Step    | Tokens (B) | Best Eval Loss")
print("--------|------------|---------------")
for r in results:
    print(f"{r['step']:7d} | {r['tokens']:10.2f} | {r['best_eval_loss']:14.4f}")
```

## Training Output Example

```
Step   500/30000 | Loss: 4.7748 | PPL:  118.49 | ...

============================================================
Evaluation at step 500
  Eval Loss: 4.7867
  Eval PPL: 119.90
  🎯 New best eval loss!
============================================================

💾 Best model checkpoint saved: ./checkpoints/best_model.pt
💾 Checkpoint saved: ./checkpoints/checkpoint_step_500.pt

Step   510/30000 | Loss: 4.7067 | PPL:  110.69 | ...
...
Step  1000/30000 | Loss: 4.2345 | PPL:   69.01 | ...

============================================================
Evaluation at step 1000
  Eval Loss: 4.3234
  Eval PPL: 75.45
  Best eval loss: 4.7867
============================================================

💾 Checkpoint saved: ./checkpoints/checkpoint_step_1000.pt

Step  1010/30000 | Loss: 4.2103 | PPL:   67.33 | ...
```

## Configuration

Control checkpointing with these flags:

```bash
python train.py \
  --save_interval 500 \      # Save regular checkpoints every N steps
  --eval_interval 500 \       # Evaluate (and check for best) every N steps
  --checkpoint_dir ./checkpoints  # Directory to save checkpoints
```

**Notes**:
- `save_interval` and `eval_interval` are independent
- Best model is saved whenever eval occurs AND loss improves
- Regular checkpoints saved at `save_interval` regardless of eval

## Best Practices

1. **Keep save_interval = eval_interval**: Ensures you have a checkpoint at each evaluation point

2. **Always use best_model.pt for inference**: It's the most generalizable model

3. **Monitor WandB `eval/best_loss`**: Shows your best performance throughout training

4. **Disk space management**: 
   - Each checkpoint ~700MB for 180M model
   - 60 checkpoints (30K steps / 500) = ~42GB
   - Delete old checkpoints if needed: `rm checkpoints/checkpoint_step_{1000..10000}.pt`
   - Keep `best_model.pt` and recent checkpoints

5. **Resume from best, not final**: If training diverged or overfit, load `best_model.pt`

## Why This Matters

### Before Improvements:
- Checkpoint every 5000 steps → lose up to 5000 steps if crash
- No tracking of best model → manual comparison needed
- Risk of continuing training past optimal point

### After Improvements:
- Checkpoint every 500 steps → lose at most 500 steps
- Automatic best model tracking → always have optimal checkpoint
- Can experiment freely knowing best model is saved

## Example Training Run

```
Step   500: Loss 4.78, Eval 4.79 → best_model.pt (first eval)
Step  1000: Loss 4.23, Eval 4.32 → no save (worse than 4.79)
Step  1500: Loss 3.98, Eval 4.05 → best_model.pt (improved!)
Step  2000: Loss 3.76, Eval 3.89 → best_model.pt (improved!)
Step  2500: Loss 3.61, Eval 3.78 → best_model.pt (improved!)
Step  3000: Loss 3.45, Eval 3.82 → no save (worse than 3.78)
Step  3500: Loss 3.32, Eval 3.71 → best_model.pt (improved!)
...
Final:       Loss 2.54, Eval 2.89 → no save (worse than 2.67)
```

Best model is at eval loss 2.67, even though training continued to lower loss. This is the power of best model tracking - prevents overfitting!

## Summary

Three key improvements:
1. ✅ **10x more frequent checkpoints** (500 vs 5000 steps)
2. ✅ **Automatic best model saving** (based on eval loss)
3. ✅ **Better tracking and logging** (visual indicators, WandB metrics)

Result: More robust training with guaranteed best model preservation.
