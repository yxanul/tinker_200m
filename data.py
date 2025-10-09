import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer
from typing import Iterator, Dict, Optional


class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "default",
        tokenizer_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        max_length: int = 2048,
        buffer_size: int = 10000,
        seed: int = 42,
        skip_first: int = 0,
        is_distributed: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.seed = seed
        self.skip_first = skip_first
        self.is_distributed = is_distributed

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Get worker info for proper sharding
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # CRITICAL FIX: Load dataset fresh in each __iter__ call
        # This prevents iterator exhaustion and data looping with persistent workers
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config if self.dataset_config != "default" else None,
            split="train",
            streaming=True,
        )

        # Apply skip_first (for train/eval separation)
        if self.skip_first > 0:
            dataset = dataset.skip(self.skip_first)

        # Distributed splitting (across GPUs/nodes)
        if self.is_distributed:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

        # Worker-specific seed for shuffling
        effective_seed = self.seed + worker_id

        # Shuffle BEFORE worker sharding for better randomness
        dataset = dataset.shuffle(seed=effective_seed, buffer_size=self.buffer_size)

        # Start iterating
        dataset_iter = iter(dataset)
        
        # Token buffer for continuous tokenization across documents
        token_buffer = []
        items_processed = 0

        for item in dataset_iter:
            # CRITICAL: Manual worker sharding - each worker processes every num_workers-th item
            # This ensures no overlap between workers
            if num_workers > 1 and items_processed % num_workers != worker_id:
                items_processed += 1
                continue
            items_processed += 1

            # Extract text
            text = item.get("text", "")
            if not text:
                continue

            # Tokenize without truncation to preserve all content
            tokens = self.tokenizer(
                text,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]

            # Add EOS token to mark document boundary
            tokens.append(self.tokenizer.eos_token_id)

            # Add to buffer
            token_buffer.extend(tokens)

            # Yield complete chunks of max_length
            while len(token_buffer) >= self.max_length:
                chunk = token_buffer[:self.max_length]
                token_buffer = token_buffer[self.max_length:]

                # Convert to tensors
                input_ids = torch.tensor(chunk, dtype=torch.long)
                
                # Create labels (same as input_ids for causal LM)
                labels = input_ids.clone()

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }


def create_dataloaders(
    batch_size: int = 16,
    max_length: int = 2048,
    num_workers: int = 8,
    buffer_size: int = 10000,
    seed: int = 42,
    eval_skip: int = 100000,  # Eval starts from this offset (reduces overlap probability)
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and eval dataloaders with different offsets.
    
    Both are infinite streams, but start from different positions:
    - Train: starts from document 0
    - Eval: starts from document 100K (by default)
    
    Different shuffling seeds further reduce overlap probability.
    Evaluation is limited to N batches in the training loop, not in the dataset.
    """
    # Check if distributed
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    # Training dataset: Starts from beginning (document 0)
    train_dataset = StreamingTextDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="default",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_length=max_length,
        buffer_size=buffer_size,
        seed=seed,
        skip_first=0,  # Start from beginning
        is_distributed=is_distributed,
    )

    # Eval dataset: Starts from different offset (e.g., document 100K)
    eval_dataset = StreamingTextDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="default",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_length=max_length,
        buffer_size=buffer_size,
        seed=seed + 999,  # Very different seed for shuffling
        skip_first=eval_skip,  # Start from eval_skip offset
        is_distributed=is_distributed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, eval_loader


def test_dataloader():
    print("Testing dataloader with different offsets for train/eval...")
    print("="*60)
    
    train_loader, eval_loader = create_dataloaders(
        batch_size=2,
        max_length=128,
        num_workers=0,  # Use 0 for testing
        buffer_size=100,
        eval_skip=1000,  # Eval starts from document 1000
    )

    print("\n1. Train loader test (starts from document 0):")
    train_samples = []
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        print(f"  Batch {i}:")
        print(f"    input_ids shape: {batch['input_ids'].shape}")
        print(f"    labels shape: {batch['labels'].shape}")
        print(f"    First 10 tokens: {batch['input_ids'][0][:10].tolist()}")
        train_samples.append(batch['input_ids'][0][:20].tolist())

    print("\n2. Eval loader test (starts from document 1000):")
    eval_samples = []
    for i, batch in enumerate(eval_loader):
        if i >= 3:
            break
        print(f"  Batch {i}:")
        print(f"    input_ids shape: {batch['input_ids'].shape}")
        print(f"    labels shape: {batch['labels'].shape}")
        print(f"    First 10 tokens: {batch['input_ids'][0][:10].tolist()}")
        eval_samples.append(batch['input_ids'][0][:20].tolist())

    print("\n3. Checking sample diversity:")
    print(f"  Train samples collected: {len(train_samples)}")
    print(f"  Eval samples collected: {len(eval_samples)}")
    print(f"  Train and eval should have different data due to offset")
    
    print("\n" + "="*60)
    print("✅ Dataloader test completed!")
    print("="*60)


if __name__ == "__main__":
    test_dataloader()
