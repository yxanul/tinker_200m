import os
import math
import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be limited.")

from model import create_model
from data import create_dataloaders


# Enable TF32 and Flash Attention for faster training
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)


class CosineScheduleWithWarmup:
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_ratio

    def step(self, current_step: int):
        if current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


class Trainer:
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.setup_device()
        self.setup_model()
        self.setup_optimizer()
        self.setup_data()
        self.setup_logging()

        self.global_step = 0
        self.tokens_seen = 0
        self.best_eval_loss = float('inf')  # Track best eval loss

    def setup_distributed(self):
        self.is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
        self.is_main = int(os.environ.get("RANK", 0)) == 0

        if self.is_distributed:
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def setup_device(self):
        if torch.cuda.is_available():
            if self.is_distributed:
                self.device = torch.device(f"cuda:{self.rank}")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=self.dtype) if self.device.type == "cuda" else nullcontext()

        if self.is_main:
            print(f"Using device: {self.device}")
            print(f"Compute dtype: {self.dtype}")
            print(f"Distributed: {self.is_distributed} (world_size: {self.world_size})")

    def setup_model(self):
        if self.is_main:
            print("\nInitializing model...")
        
        # Note: TE FP8/NVFP4 attention doesn't work with torch.compile
        # If both are requested, we disable TE attention and use PyTorch Flash Attention
        use_fp8_compute = self.args.use_fp8
        use_nvfp4_compute = self.args.use_nvfp4
        use_te_attention = (self.args.use_fp8 or self.args.use_nvfp4) and not self.args.compile
        
        # Mutual exclusivity check
        if self.args.use_fp8 and self.args.use_nvfp4:
            if self.is_main:
                print("\n⚠️  Warning: Both --use_fp8 and --use_nvfp4 specified")
                print("  Using NVFP4 (E2M1 4-bit) instead of FP8 (E4M3/E5M2 8-bit)")
            use_fp8_compute = False  # NVFP4 takes precedence
        
        if (self.args.use_fp8 or self.args.use_nvfp4) and self.args.compile and self.is_main:
            format_name = "NVFP4 (E2M1)" if self.args.use_nvfp4 else "FP8 (E4M3/E5M2)"
            print(f"\n⚠️  Note: torch.compile + {format_name} mode")
            print(f"  - {format_name} compute: Enabled (Linear, RMSNorm, Fused QKV)")
            print("  - TE attention: Disabled (incompatible with compile)")
            print("  - Using PyTorch Flash Attention instead")
        
        # Auto-calculate FFN hidden size if not specified (2.67x for SwiGLU)
        ffn_hidden = self.args.ffn_hidden
        if ffn_hidden is None:
            ffn_hidden = int(self.args.d_model * 8 / 3)  # 2.67x rule
        
        self.model = create_model(
            d_model=self.args.d_model,
            n_layers=self.args.n_layers,
            n_heads=self.args.n_heads,
            n_kv_heads=self.args.n_kv_heads,
            ffn_hidden=ffn_hidden,
            max_seq_len=self.args.max_seq_len,
            use_fp8=use_fp8_compute,
            use_nvfp4=use_nvfp4_compute,
            use_te_attention=use_te_attention,
        )
        self.model = self.model.to(self.device)
        
        # Compile model for additional 10-20% speedup
        if self.args.compile:
            if self.is_main:
                print(f"\nCompiling model with torch.compile (mode: {self.args.compile_mode})...")
                print("  ⏳ First iteration will be slow (~30 sec - 10 min depending on mode)")
                print("  After compilation, expect 10-20% throughput increase")
            
            try:
                self.model = torch.compile(
                    self.model,
                    mode=self.args.compile_mode,
                    fullgraph=False,  # Allow graph breaks for TE compatibility
                )
                if self.is_main:
                    print("  ✓ Model compilation enabled")
            except Exception as e:
                if self.is_main:
                    print(f"  ⚠️ Compilation failed: {e}")
                    print("  Continuing without compilation...")

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.rank])
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model

    def setup_optimizer(self):
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.raw_model.named_parameters():
            if param.requires_grad:
                if param.ndim >= 2:  # Weights
                    decay_params.append(param)
                else:  # Biases and norms
                    no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW for ~20-30% faster optimizer step on CUDA
        use_fused = torch.cuda.is_available()
        
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            fused=use_fused,
        )

        self.scheduler = CosineScheduleWithWarmup(
            self.optimizer,
            warmup_steps=self.args.warmup_steps,
            total_steps=self.args.total_steps,
            max_lr=self.args.learning_rate,
            min_lr_ratio=0.1,
        )

        if self.is_main:
            print(f"\nOptimizer: AdamW")
            print(f"  Fused: {use_fused} (20-30% faster optimizer step)")
            print(f"  Decay params: {len(decay_params)}")
            print(f"  No decay params: {len(no_decay_params)}")
            print(f"  Learning rate: {self.args.learning_rate}")
            print(f"  Weight decay: {self.args.weight_decay}")

    def setup_data(self):
        if self.is_main:
            print("\nInitializing dataloaders...")
            print(f"  Train/eval offset strategy:")
            print(f"    Train: starts from document 0")
            print(f"    Eval: starts from document {self.args.eval_skip}")
            print(f"    Different seeds further reduce overlap")

        self.train_loader, self.eval_loader = create_dataloaders(
            batch_size=self.args.batch_size,
            max_length=self.args.max_seq_len,
            num_workers=self.args.num_workers,
            buffer_size=self.args.buffer_size,
            seed=self.args.seed,
            eval_skip=self.args.eval_skip,
            pin_memory=True,
        )

        self.tokens_per_batch = self.args.batch_size * self.args.max_seq_len * self.args.grad_accum_steps * self.world_size

        if self.is_main:
            print(f"  Batch size per GPU: {self.args.batch_size}")
            print(f"  Gradient accumulation steps: {self.args.grad_accum_steps}")
            print(f"  Effective batch size: {self.args.batch_size * self.args.grad_accum_steps * self.world_size}")
            print(f"  Tokens per batch: {self.tokens_per_batch:,}")

    def setup_logging(self):
        if self.is_main and WANDB_AVAILABLE and not self.args.no_wandb:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.run_name,
                config=vars(self.args),
            )
            if self.is_main:
                print(f"\nWandB logging enabled: {self.args.wandb_project}/{self.args.run_name}")
        else:
            if self.is_main:
                print("\nWandB logging disabled")

    @torch.no_grad()
    def evaluate(self, num_batches: int = 50):
        self.model.eval()
        total_loss = 0.0
        count = 0

        eval_iter = iter(self.eval_loader)
        for _ in range(num_batches):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with self.ctx:
                _, loss = self.model(input_ids, labels)

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

        self.model.train()
        return avg_loss, perplexity

    def train(self):
        if self.is_main:
            print(f"\n{'='*60}")
            print(f"Starting training for {self.args.total_steps:,} steps")
            print(f"Target: ~{(self.args.total_steps * self.tokens_per_batch) / 1e9:.2f}B tokens")
            print(f"{'='*60}\n")

        self.model.train()
        train_iter = iter(self.train_loader)
        
        losses = []
        start_time = time.time()
        tokens_per_sec_ema = 0

        while self.global_step < self.args.total_steps:
            # Update learning rate
            lr = self.scheduler.step(self.global_step)

            # Gradient accumulation loop
            self.optimizer.zero_grad()
            accum_loss = 0.0

            for micro_step in range(self.args.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                with self.ctx:
                    _, loss = self.model(input_ids, labels)
                    loss = loss / self.args.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()

            # Gradient clipping
            # Note: grad_norm is the total norm BEFORE clipping (PyTorch behavior)
            # Gradients are still clipped to args.grad_clip (default 1.0)
            if self.args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.grad_clip
                )
            else:
                grad_norm = 0.0

            self.optimizer.step()

            # Update metrics
            self.global_step += 1
            self.tokens_seen += self.tokens_per_batch
            losses.append(accum_loss)

            # Calculate tokens/sec
            elapsed = time.time() - start_time
            tokens_per_sec = self.tokens_per_batch / elapsed if elapsed > 0 else 0
            tokens_per_sec_ema = 0.9 * tokens_per_sec_ema + 0.1 * tokens_per_sec if tokens_per_sec_ema > 0 else tokens_per_sec
            start_time = time.time()

            # Logging
            if self.global_step % self.args.log_interval == 0 and self.is_main:
                avg_loss = sum(losses) / len(losses)
                ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
                
                print(f"Step {self.global_step:5d}/{self.args.total_steps} | "
                      f"Loss: {avg_loss:.4f} | PPL: {ppl:7.2f} | "
                      f"LR: {lr:.2e} | GradNorm: {grad_norm:.3f} | "
                      f"Tokens/sec: {tokens_per_sec_ema:.0f} | "
                      f"Tokens: {self.tokens_seen / 1e9:.3f}B")

                if WANDB_AVAILABLE and not self.args.no_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/perplexity": ppl,
                        "train/learning_rate": lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec_ema,
                        "tokens_seen": self.tokens_seen,
                        "step": self.global_step,
                    })

                losses = []

            # Evaluation
            if self.global_step % self.args.eval_interval == 0 and self.is_main:
                eval_loss, eval_ppl = self.evaluate(self.args.eval_batches)
                print(f"\n{'='*60}")
                print(f"Evaluation at step {self.global_step}")
                print(f"  Eval Loss: {eval_loss:.4f}")
                print(f"  Eval PPL: {eval_ppl:.2f}")
                
                # Check if this is the best model so far
                is_best = eval_loss < self.best_eval_loss
                if is_best:
                    prev_best = self.best_eval_loss
                    self.best_eval_loss = eval_loss
                    if prev_best == float('inf'):
                        print(f"  🎯 New best eval loss!")
                    else:
                        print(f"  🎯 New best eval loss! (previous: {prev_best:.4f})")
                else:
                    print(f"  Best eval loss: {self.best_eval_loss:.4f}")
                print(f"{'='*60}\n")

                if WANDB_AVAILABLE and not self.args.no_wandb:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/perplexity": eval_ppl,
                        "eval/best_loss": self.best_eval_loss,
                        "step": self.global_step,
                    })

                # Save best model checkpoint
                if is_best:
                    self.save_checkpoint(is_best=True)

            # Regular checkpointing
            if self.global_step % self.args.save_interval == 0 and self.is_main:
                self.save_checkpoint()

        if self.is_main:
            print("\nTraining completed!")
            self.save_checkpoint(final=True)

    def save_checkpoint(self, final: bool = False, is_best: bool = False):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        if final:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, "final_model.pt")
        elif is_best:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, "best_model.pt")
        else:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")

        checkpoint = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_eval_loss": self.best_eval_loss,
            "args": vars(self.args),
        }

        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            print(f"💾 Best model checkpoint saved: {checkpoint_path}")
        else:
            print(f"💾 Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    
    # Model args
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (768 for GPT-2 small/medium)")
    parser.add_argument("--n_layers", type=int, default=32, help="Number of transformer layers (12 for ~125M, 32 for ~180M)")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=4, help="Number of KV heads for GQA")
    parser.add_argument("--ffn_hidden", type=int, default=None, help="FFN hidden size (default: 2.67 × d_model for SwiGLU)")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--use_fp8", action="store_true", help="Enable FP8 training (E4M3/E5M2, requires H100+)")
    parser.add_argument("--use_nvfp4", action="store_true", help="Enable NVFP4 training (E2M1 4-bit, requires RTX 5090/B200+)")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile (10-20% faster)")
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Data args
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_skip", type=int, default=100000, help="Eval dataset starts from this document offset")
    
    # Logging args
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default="dense-llm-pretraining")
    parser.add_argument("--run_name", type=str, default="dense-180m")
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
