import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
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
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

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

        if self.use_flash:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w2 = nn.Linear(ffn_hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ffn_hidden, bias=False)

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
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, norm_eps)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, use_qk_norm, use_flash)
        self.ffn_norm = RMSNorm(d_model, norm_eps)
        self.ffn = SwiGLU(d_model, ffn_hidden)

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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_theta)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, n_kv_heads, head_dim, ffn_hidden,
                norm_eps, use_qk_norm, use_flash
            )
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model, norm_eps)
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
        h = self.tok_embeddings(input_ids)

        rope_cos, rope_sin = self.rope(h, seq_len)

        for layer in self.layers:
            h = layer(h, rope_cos, rope_sin)

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

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
            if not self.tie_weights:
                n_params -= self.output.weight.numel()
        return n_params


def create_model():
    model = DenseTransformer(
        vocab_size=32768,
        d_model=768,
        n_layers=32,
        n_heads=12,
        n_kv_heads=4,
        head_dim=64,
        ffn_hidden=2048,
        max_seq_len=2048,
        norm_eps=1e-6,
        rope_theta=10000,
        use_qk_norm=True,
        use_flash=True,
        tie_weights=True,
    )
    
    total_params = model.get_num_params(non_embedding=False) / 1e6
    non_emb_params = model.get_num_params(non_embedding=True) / 1e6
    print(f"Total parameters: {total_params:.2f}M")
    print(f"Non-embedding parameters: {non_emb_params:.2f}M")
    
    return model


if __name__ == "__main__":
    model = create_model()
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 32768, (batch_size, seq_len))
    labels = torch.randint(0, 32768, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(input_ids, labels)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
