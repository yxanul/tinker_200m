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
        
        self.model = create_model()
        self.model = self.model.to(self.device)

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

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
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
            print(f"  Decay params: {len(decay_params)}")
            print(f"  No decay params: {len(no_decay_params)}")
            print(f"  Learning rate: {self.args.learning_rate}")
            print(f"  Weight decay: {self.args.weight_decay}")

    def setup_data(self):
        if self.is_main:
            print("\nInitializing dataloaders...")

        self.train_loader, self.eval_loader = create_dataloaders(
            batch_size=self.args.batch_size,
            max_length=self.args.max_seq_len,
            num_workers=self.args.num_workers,
            buffer_size=self.args.buffer_size,
            seed=self.args.seed,
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
                print(f"{'='*60}\n")

                if WANDB_AVAILABLE and not self.args.no_wandb:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/perplexity": eval_ppl,
                        "step": self.global_step,
                    })

            # Checkpointing
            if self.global_step % self.args.save_interval == 0 and self.is_main:
                self.save_checkpoint()

        if self.is_main:
            print("\nTraining completed!")
            self.save_checkpoint(final=True)

    def save_checkpoint(self, final: bool = False):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        if final:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, "final_model.pt")
        else:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")

        checkpoint = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "args": vars(self.args),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    
    # Model args
    parser.add_argument("--max_seq_len", type=int, default=2048)
    
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
    
    # Logging args
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default="dense-llm-pretraining")
    parser.add_argument("--run_name", type=str, default="dense-180m")
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
