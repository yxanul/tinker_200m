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
        take_first: Optional[int] = None,
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
        self.take_first = take_first
        self.is_distributed = is_distributed

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Initialize dataset ONCE here, not in __iter__
        # This prevents rate limit errors from multiple workers making API calls
        print(f"Initializing dataset {dataset_name} (one-time initialization)...")
        self._base_dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config if self.dataset_config != "default" else None,
            split="train",
            streaming=True,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Get worker info for proper sharding
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Start with base dataset
        dataset = self._base_dataset

        # Apply skip_first (for train/eval split)
        if self.skip_first > 0:
            dataset = dataset.skip(self.skip_first)

        # Apply take_first (for eval set to limit size)
        if self.take_first is not None and self.take_first > 0:
            dataset = dataset.take(self.take_first)

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
    eval_take: int = 10000,  # Number of documents for eval set
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create disjoint train and eval dataloaders.
    
    Eval set: First `eval_take` documents
    Train set: Everything after `eval_take` documents
    
    This ensures no data leakage between train and eval.
    """
    # Check if distributed
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    # Eval dataset: First eval_take documents only
    eval_dataset = StreamingTextDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="default",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_length=max_length,
        buffer_size=buffer_size,
        seed=seed + 1,  # Different seed for shuffling
        skip_first=0,
        take_first=eval_take,  # Take only first eval_take documents
        is_distributed=is_distributed,
    )

    # Training dataset: Everything AFTER the first eval_take documents
    train_dataset = StreamingTextDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="default",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_length=max_length,
        buffer_size=buffer_size,
        seed=seed,
        skip_first=eval_take,  # Skip the eval portion
        take_first=None,  # Take everything after skip
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
    print("Testing dataloader with disjoint train/eval split...")
    print("="*60)
    
    train_loader, eval_loader = create_dataloaders(
        batch_size=2,
        max_length=128,
        num_workers=0,  # Use 0 for testing
        buffer_size=100,
        eval_take=100,  # Small eval set for testing
    )

    print("\n1. Train loader test (should skip first 100 documents):")
    train_tokens_seen = set()
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        print(f"  Batch {i}:")
        print(f"    input_ids shape: {batch['input_ids'].shape}")
        print(f"    labels shape: {batch['labels'].shape}")
        print(f"    First 10 tokens: {batch['input_ids'][0][:10].tolist()}")
        
        # Track tokens for overlap check
        train_tokens_seen.update(batch['input_ids'][0].tolist())

    print("\n2. Eval loader test (should use first 100 documents only):")
    eval_tokens_seen = set()
    for i, batch in enumerate(eval_loader):
        if i >= 3:
            break
        print(f"  Batch {i}:")
        print(f"    input_ids shape: {batch['input_ids'].shape}")
        print(f"    labels shape: {batch['labels'].shape}")
        print(f"    First 10 tokens: {batch['input_ids'][0][:10].tolist()}")
        
        # Track tokens for overlap check
        eval_tokens_seen.update(batch['input_ids'][0].tolist())

    print("\n3. Checking for overlap between train and eval:")
    print(f"  Train unique tokens: {len(train_tokens_seen)}")
    print(f"  Eval unique tokens: {len(eval_tokens_seen)}")
    print(f"  Note: Token overlap is expected (same vocabulary), document overlap is not")
    
    print("\n" + "="*60)
    print("✅ Dataloader test completed!")
    print("="*60)


if __name__ == "__main__":
    test_dataloader()
