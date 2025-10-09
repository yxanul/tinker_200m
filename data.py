import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer
from typing import Iterator, Dict


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

        self.dataset = None

    def _initialize_dataset(self):
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config if self.dataset_config != "default" else None,
            split="train",
            streaming=True,
        )

        # Shuffle to avoid stale data from 2013
        dataset = dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        # Skip first N samples for validation split
        if self.skip_first > 0:
            dataset = dataset.skip(self.skip_first)

        # Distributed splitting
        if self.is_distributed:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

        self.dataset = dataset

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.dataset is None:
            self._initialize_dataset()

        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue

            # Tokenize on the fly
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            # Create labels (shift input_ids by 1 for causal LM)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore padding in loss

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
    eval_skip: int = 1000000,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    # Check if distributed
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    # Training dataset (uses all data or skips eval portion)
    train_dataset = StreamingTextDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="default",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_length=max_length,
        buffer_size=buffer_size,
        seed=seed,
        skip_first=0,
        is_distributed=is_distributed,
    )

    # Eval dataset (uses first eval_skip samples with different seed)
    eval_dataset = StreamingTextDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="default",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_length=max_length,
        buffer_size=buffer_size,
        seed=seed + 1,  # Different seed for eval
        skip_first=0,
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
    print("Testing dataloader...")
    train_loader, eval_loader = create_dataloaders(
        batch_size=2,
        max_length=128,
        num_workers=0,  # Use 0 for testing
        buffer_size=100,
    )

    print("\nTrain loader test:")
    for i, batch in enumerate(train_loader):
        if i >= 2:
            break
        print(f"Batch {i}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  Sample tokens: {batch['input_ids'][0][:10]}")

    print("\nEval loader test:")
    for i, batch in enumerate(eval_loader):
        if i >= 1:
            break
        print(f"Batch {i}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")

    print("\nDataloader test completed!")


if __name__ == "__main__":
    test_dataloader()
