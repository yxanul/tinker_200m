# NVFP4 Training Guide (E2M1 - 4-bit Floating Point)

## What is NVFP4?

**NVFP4** (E2M1) is NVIDIA's **4-bit floating point format** introduced in Transformer Engine 2.8 for **Blackwell GPUs**.

### Format Breakdown:
- **E2M1**: 2-bit exponent, 1-bit mantissa, 1-bit sign = **4 bits total**
- **Half the bits of FP8** (E4M3/E5M2 = 8 bits)
- **Potential 2-3x speedup** over BF16 (experimental)

### Comparison:

| Format | Bits | Exponent | Mantissa | Use Case |
|--------|------|----------|----------|----------|
| **BF16** | 16 | 8 | 7 | Standard mixed precision |
| **FP8 (E4M3)** | 8 | 4 | 3 | Forward pass (H100+) |
| **FP8 (E5M2)** | 8 | 5 | 2 | Gradients (H100+) |
| **NVFP4 (E2M1)** | 4 | 2 | 1 | Bleeding edge (RTX 5090/B200+) |

---

## Requirements

### Hardware:
✅ **NVIDIA RTX 5090** (Consumer Blackwell, SM 12.0)  
✅ **NVIDIA B200/GB200** (Datacenter Blackwell, SM 10.0)  
❌ **NVIDIA H100** (Hopper, SM 9.0) - FP8 only, no NVFP4

### Software:
- **Transformer Engine 2.8+**
- **CUDA 12.0+**
- **PyTorch 2.1+**

### Check Your System:

```bash
# Check GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Expected output for RTX 5090:
# NVIDIA GeForce RTX 5090, 12.0

# Check TE version
python -c "import transformer_engine as te; print(f'TE: {te.__version__}')"
# Should show: 2.8.0 or higher

# Check E2M1 availability
python -c "
from transformer_engine.common.recipe import Format
formats = [attr for attr in dir(Format) if not attr.startswith('_')]
print('Available formats:', formats)
print('E2M1 available:', hasattr(Format, 'E2M1'))
"
# Should show: E2M1 available: True
```

---

## Usage

### Basic NVFP4 Training:

```bash
# 100M model with NVFP4
python train.py \
  --d_model 768 \
  --n_layers 12 \
  --use_nvfp4 \
  --batch_size 16 \
  --grad_accum_steps 6
```

### NVFP4 + torch.compile:

```bash
# Maximum performance (experimental)
python train.py \
  --d_model 768 \
  --n_layers 12 \
  --use_nvfp4 \
  --compile \
  --batch_size 16 \
  --grad_accum_steps 6
```

**Note**: With `--compile`, TE attention is disabled (uses PyTorch Flash Attention instead)

---

## Expected Performance

### Baseline (100M model, RTX 5090):

| Mode | Tokens/sec | Speedup | Memory |
|------|------------|---------|--------|
| **BF16** | ~90-100K | 1.0x | 100% |
| **FP8 (E4M3/E5M2)** | ~160-180K | 1.7-1.9x | ~70% |
| **NVFP4 (E2M1)** | **~200-270K?** | **2.2-3.0x?** | **~50%?** |

**Note**: NVFP4 numbers are **theoretical/estimated**. It's Day 0!

### Your Current Results:

**With FP8 (E4M3/E5M2)**:
```
100M model: 181K tokens/sec
```

**Expected with NVFP4 (E2M1)**:
```
100M model: ~220-250K tokens/sec? (+20-35%)
```

---

## Trade-offs

### Pros:
✅ **Fastest training** (potential 2-3x over BF16)  
✅ **Lowest memory** (~50% reduction in activations)  
✅ **Cutting edge** (be among first to test!)  
✅ **Works on consumer GPU** (RTX 5090)

### Cons:
⚠️ **Experimental** (Day 0 release)  
⚠️ **Potential accuracy loss** (4-bit is very low precision)  
⚠️ **Stability unknown** (may see NaN losses, convergence issues)  
⚠️ **Limited documentation** (community still testing)  
⚠️ **May need tuning** (learning rate, grad clipping, etc.)

---

## Safety Recommendations

### Start Small:
```bash
# Test NVFP4 on short run first (1K steps)
python train.py \
  --use_nvfp4 \
  --d_model 768 \
  --n_layers 12 \
  --total_steps 1000 \
  --batch_size 8
```

### Monitor for Issues:
- **NaN losses**: Very likely with 4-bit precision
- **Training instability**: Loss spikes, gradient explosions
- **Convergence**: May not converge as well as FP8
- **Quality degradation**: Check perplexity vs FP8 baseline

### Fallback Plan:
If NVFP4 fails, fall back to FP8:
```bash
# Proven stable (your current 181K tok/s)
python train.py --use_fp8 --d_model 768 --n_layers 12
```

---

## Comparison: FP8 vs NVFP4

### When to Use FP8 (E4M3/E5M2):
✅ Production training runs  
✅ Need stability and proven results  
✅ H100 datacenter GPUs  
✅ Already excellent performance (1.8-2.2x)

### When to Use NVFP4 (E2M1):
🔬 Research and experimentation  
🔬 Have RTX 5090 or B200 (only GPUs that support it)  
🔬 Want bleeding edge performance  
🔬 Can tolerate instability/debugging  
🔬 Testing limits of low-precision training

---

## Troubleshooting

### Error: "E2M1 format not available"
```bash
# Check TE version
pip install --upgrade transformer-engine>=2.8.0

# Verify E2M1 is present
python -c "from transformer_engine.common.recipe import Format; print(hasattr(Format, 'E2M1'))"
```

### NaN Losses with NVFP4:
Try these mitigations:
1. **Lower learning rate**: Try 50-70% of FP8 LR
2. **Increase gradient clipping**: `--grad_clip 0.5` (instead of 1.0)
3. **Use mixed precision**: Keep some layers in FP8/BF16
4. **Add loss scaling**: Manual loss scaling before backward

### Slower Than Expected:
1. **Check GPU utilization**: `nvidia-smi dmon`
2. **Verify E2M1 is active**: Look for "✓ NVFP4 training enabled" in logs
3. **Try without compile**: Some ops may not compile well
4. **Increase batch size**: NVFP4 should allow larger batches

---

## Command Reference

### FP8 (Proven, Stable):
```bash
python train.py --use_fp8 --compile
```

### NVFP4 (Experimental):
```bash
python train.py --use_nvfp4 --compile
```

### Both Specified (NVFP4 takes precedence):
```bash
python train.py --use_fp8 --use_nvfp4 --compile
# Warning: Both flags specified, using NVFP4
```

### Test on Short Run:
```bash
python train.py --use_nvfp4 --total_steps 1000
```

---

## Expected Timeline

**Day 0-1** (Now): Testing, stability unknown  
**Week 1-2**: Community reports performance/issues  
**Month 1**: Documentation improves, stability patches  
**Month 2-3**: Production-ready (maybe)

**Your position**: **Day 0 pioneer!** 🚀

---

## Reporting Results

If you test NVFP4, track:
1. **Throughput**: tokens/sec vs FP8
2. **Stability**: Any NaN losses, how often?
3. **Quality**: Final perplexity vs FP8 baseline
4. **Memory**: Peak memory usage
5. **GPU**: RTX 5090 specs

Share results with:
- NVIDIA Transformer Engine team
- Community (Reddit r/LocalLLaMA, Twitter)
- This project (issue tracker)

---

## Rollback Commands

### If NVFP4 Doesn't Work:

**Option 1: Go back to proven FP8**
```bash
python train.py --use_fp8 --d_model 768 --n_layers 12 --compile
# 181K tokens/sec (proven stable)
```

**Option 2: Try FP8 without compile (most stable)**
```bash
python train.py --use_fp8 --d_model 768 --n_layers 12
# ~87-89K tokens/sec with TE attention
```

**Option 3: Pure BF16 (maximum stability)**
```bash
python train.py --d_model 768 --n_layers 12
# ~60K tokens/sec, no precision issues
```

---

## Summary

**NVFP4 (E2M1)** is the **newest, fastest, most experimental** format:
- ✅ Available on **RTX 5090** (SM 12.0)
- ✅ Easy to enable: `--use_nvfp4`
- ⚠️ **Experimental** (Day 0 release)
- 🎯 Potential: **200-270K tokens/sec** on 100M model

**Recommendation**: 
1. **Test NVFP4 on 1K steps** to verify it works
2. **Compare to FP8 baseline** (your current 181K tok/s)
3. **If stable → full run**, if unstable → stick with FP8

You're a **bleeding edge pioneer** now! Good luck! 🔥
