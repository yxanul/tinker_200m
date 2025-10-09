import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    HAS_TE = True
except ImportError:
    HAS_TE = False
    print("Warning: transformer_engine not available. FP8 training will be disabled.")


class RMSNorm(nn.Module):
    """Fallback RMSNorm for when TE is not available"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # (seq_len, 1, dim)
    sin = sin.unsqueeze(1)  # (seq_len, 1, dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        use_qk_norm: bool = True,
        use_flash: bool = True,
        use_fp8: bool = False,
        use_te_attention: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        self.use_fp8 = use_fp8 and HAS_TE
        self.use_te_attention = use_te_attention and self.use_fp8 and HAS_TE

        if self.use_fp8 and not HAS_TE:
            raise RuntimeError("FP8 requested but transformer_engine not available. Install with: pip install transformer-engine")

        # Use fused QKV projection for FP8 (single kernel, much faster)
        if self.use_fp8:
            # Fused QKV: Q (n_heads * head_dim) + K (n_kv_heads * head_dim) + V (n_kv_heads * head_dim)
            self.qkv_proj = te.Linear(d_model, (n_heads + 2 * n_kv_heads) * head_dim, bias=False)
            self.use_fused_qkv = True
            
            # TE FP8 DotProductAttention (only if not using torch.compile)
            if self.use_te_attention:
                self.te_attn = te.DotProductAttention(
                    num_attention_heads=n_heads,
                    kv_channels=head_dim,
                    attention_dropout=0.0,
                )
        else:
            # Separate projections for non-FP8 mode
            self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
            self.use_fused_qkv = False
        
        # Output projection
        LinearLayer = te.Linear if self.use_fp8 else nn.Linear
        self.o_proj = LinearLayer(n_heads * head_dim, d_model, bias=False)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            # Use TE RMSNorm for FP8 if available, else fallback
            NormLayer = te.RMSNorm if (self.use_fp8 and HAS_TE) else RMSNorm
            self.q_norm = NormLayer(head_dim, eps=1e-6) if self.use_fp8 else RMSNorm(head_dim)
            self.k_norm = NormLayer(head_dim, eps=1e-6) if self.use_fp8 else RMSNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if self.use_fused_qkv:
            # Fused QKV projection (FP8 mode)
            qkv = self.qkv_proj(x)  # [batch, seq, (n_heads + 2*n_kv_heads) * head_dim]
            
            # Split into Q, K, V
            q_dim = self.n_heads * self.head_dim
            k_dim = self.n_kv_heads * self.head_dim
            v_dim = self.n_kv_heads * self.head_dim
            
            q = qkv[:, :, :q_dim].view(batch_size, seq_len, self.n_heads, self.head_dim)
            k = qkv[:, :, q_dim:q_dim+k_dim].view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            v = qkv[:, :, q_dim+k_dim:].view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        else:
            # Separate projections (non-FP8 mode)
            q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Repeat k, v for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose to (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_te_attention:
            # TE FP8 attention expects (B, S, H, D) layout
            q_te = q.transpose(1, 2).contiguous()  # (B, S, H, D)
            k_te = k.transpose(1, 2).contiguous()
            v_te = v.transpose(1, 2).contiguous()
            
            # Ensure same dtype (QK norm might change Q/K dtype, but V is unchanged)
            # TE DotProductAttention requires all inputs to have same dtype
            target_dtype = q_te.dtype
            if k_te.dtype != target_dtype:
                k_te = k_te.to(target_dtype)
            if v_te.dtype != target_dtype:
                v_te = v_te.to(target_dtype)
            
            # TE DotProductAttention (FP8-optimized)
            output = self.te_attn(q_te, k_te, v_te, attention_mask=None)
            
            # Back to (B, H, S, D) for consistency
            output = output.transpose(1, 2).contiguous()
        elif self.use_flash:
            # PyTorch Flash Attention (BF16)
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        else:
            # Manual attention (fallback)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, use_fp8: bool = False):
        super().__init__()
        self.use_fp8 = use_fp8 and HAS_TE
        
        if self.use_fp8 and not HAS_TE:
            raise RuntimeError("FP8 requested but transformer_engine not available")
        
        # Use TE Linear for FP8 support
        LinearLayer = te.Linear if self.use_fp8 else nn.Linear
        
        self.w1 = LinearLayer(d_model, ffn_hidden, bias=False)
        self.w2 = LinearLayer(ffn_hidden, d_model, bias=False)
        self.w3 = LinearLayer(d_model, ffn_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ffn_hidden: int,
        norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
        use_flash: bool = True,
        use_fp8: bool = False,
        use_te_attention: bool = True,
    ):
        super().__init__()
        self.use_fp8 = use_fp8 and HAS_TE
        
        if self.use_fp8 and not HAS_TE:
            raise RuntimeError("FP8 requested but transformer_engine not available")
        
        # Use TE RMSNorm for FP8 if available
        NormLayer = te.RMSNorm if self.use_fp8 else RMSNorm
        
        self.attn_norm = NormLayer(d_model, eps=norm_eps) if self.use_fp8 else RMSNorm(d_model, norm_eps)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, use_qk_norm, use_flash, use_fp8, use_te_attention)
        self.ffn_norm = NormLayer(d_model, eps=norm_eps) if self.use_fp8 else RMSNorm(d_model, norm_eps)
        self.ffn = SwiGLU(d_model, ffn_hidden, use_fp8)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DenseTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 768,
        n_layers: int = 32,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        head_dim: int = 64,
        ffn_hidden: int = 2048,
        max_seq_len: int = 2048,
        norm_eps: float = 1e-6,
        rope_theta: int = 10000,
        use_qk_norm: bool = True,
        use_flash: bool = True,
        tie_weights: bool = True,
        use_fp8: bool = False,
        use_te_attention: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights
        self.use_fp8 = use_fp8 and HAS_TE
        
        if self.use_fp8 and not HAS_TE:
            raise RuntimeError(
                "FP8 training requested but transformer_engine not available. "
                "Install with: pip install transformer-engine"
            )
        
        # FP8 recipe: E4M3 forward, E5M2 backward (HYBRID mode)
        if self.use_fp8:
            self.fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
                amax_history_len=16,
                amax_compute_algo="max",
                fp8_dpa=True,  # Enable FP8 dot-product attention
            )
            print("✓ FP8 training enabled (E4M3 forward, E5M2 backward)")
            print("✓ Fused QKV enabled (single kernel for Q/K/V projections)")
            if use_te_attention:
                print("✓ FP8 DotProductAttention enabled (FP8 QK^T + softmax)")
            else:
                print("✓ PyTorch Flash Attention (FP8 incompatible with torch.compile)")
        else:
            self.fp8_recipe = None

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_theta)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, n_kv_heads, head_dim, ffn_hidden,
                norm_eps, use_qk_norm, use_flash, use_fp8, use_te_attention
            )
            for _ in range(n_layers)
        ])

        # Use TE RMSNorm for final norm if FP8 enabled
        if self.use_fp8:
            self.norm = te.RMSNorm(d_model, eps=norm_eps)
        else:
            self.norm = RMSNorm(d_model, norm_eps)
        
        # Output projection (TE Linear if FP8)
        if self.use_fp8:
            self.output = te.Linear(d_model, vocab_size, bias=False)
        else:
            self.output = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.output.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        
        # Use FP8 autocast if enabled
        if self.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                h = self._forward_impl(input_ids, seq_len)
        else:
            h = self._forward_impl(input_ids, seq_len)
        
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            # logits[:, :-1] predicts labels[:, 1:]
            # This ensures position i predicts token i+1, not token i
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss
    
    def _forward_impl(self, input_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Core forward pass that can be wrapped with FP8 autocast"""
        h = self.tok_embeddings(input_ids)
        rope_cos, rope_sin = self.rope(h, seq_len)

        for layer in self.layers:
            h = layer(h, rope_cos, rope_sin)
        
        return h

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
            if not self.tie_weights:
                n_params -= self.output.weight.numel()
        return n_params


def create_model(
    d_model: int = 768,
    n_layers: int = 32,
    n_heads: int = 12,
    n_kv_heads: int = 4,
    ffn_hidden: int = 2048,
    max_seq_len: int = 2048,
    use_fp8: bool = False,
    use_te_attention: bool = True,
):
    """
    Create a DenseTransformer model.
    
    Args:
        d_model: Model dimension (768 for GPT-2 small/medium)
        n_layers: Number of transformer layers (12 for ~125M, 32 for ~180M)
        n_heads: Number of attention heads
        n_kv_heads: Number of KV heads for GQA
        ffn_hidden: FFN hidden size (default 2048 = 2.67x for 768d)
        max_seq_len: Maximum sequence length
        use_fp8: Enable FP8 training (requires transformer_engine and H100+ GPU)
                 E4M3 format for forward pass, E5M2 for backward pass
        use_te_attention: Use TE FP8 attention (incompatible with torch.compile)
    
    Returns:
        DenseTransformer model
    """
    head_dim = d_model // n_heads
    
    model = DenseTransformer(
        vocab_size=32768,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        ffn_hidden=ffn_hidden,
        max_seq_len=max_seq_len,
        norm_eps=1e-6,
        rope_theta=10000,
        use_qk_norm=True,
        use_flash=True,
        tie_weights=True,
        use_fp8=use_fp8,
        use_te_attention=use_te_attention,
    )
    
    total_params = model.get_num_params(non_embedding=False) / 1e6
    non_emb_params = model.get_num_params(non_embedding=True) / 1e6
    
    print(f"\nModel Architecture:")
    print(f"  d_model: {d_model}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads} (KV heads: {n_kv_heads}, ratio: {n_heads//n_kv_heads}:1)")
    print(f"  head_dim: {head_dim}")
    print(f"  ffn_hidden: {ffn_hidden} ({ffn_hidden/d_model:.2f}x d_model)")
    print(f"  vocab_size: 32768")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"\nModel Size:")
    print(f"  Total parameters: {total_params:.2f}M")
    print(f"  Non-embedding parameters: {non_emb_params:.2f}M")
    
    if use_fp8:
        if HAS_TE:
            print("✓ FP8 training mode enabled")
            print("  - Format: E4M3 (forward), E5M2 (backward)")
            print("  - Fused QKV: Single kernel for Q/K/V")
            print("  - FP8 attention: QK^T + softmax in FP8")
            print("  - Requires: H100 or newer GPU")
            print("  - Expected speedup: 1.8-2.3x over BF16")
        else:
            print("⚠ FP8 requested but transformer_engine not installed")
            print("  Install with: pip install transformer-engine")
    else:
        print("✓ BF16/FP32 training mode (PyTorch Flash Attention)")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 training")
    args = parser.parse_args()
    
    print("="*60)
    print("Dense Transformer Model Test")
    print("="*60)
    
    # Create model
    model = create_model(use_fp8=args.fp8)
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 32768, (batch_size, seq_len))
    labels = torch.randint(0, 32768, (batch_size, seq_len))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        print(f"\n✓ Testing on GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print(f"\n✓ Testing on CPU")
    
    model.eval()
    with torch.no_grad():
        logits, loss = model(input_ids, labels)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Device: {logits.device}")
    print(f"  Dtype: {logits.dtype}")
    
    if args.fp8 and HAS_TE:
        print(f"\n✓ FP8 forward pass successful!")
        print(f"  Note: FP8 provides 1.5-2x speedup on H100+ GPUs")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
