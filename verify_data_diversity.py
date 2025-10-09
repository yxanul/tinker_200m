"""
Script to verify that the dataloader is producing diverse data and not repeating batches.
This helps debug potential data pipeline issues.
"""

import torch
from data import create_dataloaders
from collections import defaultdict


def hash_tensor(tensor):
    """Create a simple hash of a tensor for comparison."""
    return hash(tensor.cpu().numpy().tobytes())


def verify_diversity(num_batches=100, batch_size=4, num_workers=4):
    print(f"Testing data diversity with {num_workers} workers...")
    print(f"Collecting {num_batches} batches of size {batch_size}\n")
    
    train_loader, _ = create_dataloaders(
        batch_size=batch_size,
        max_length=512,  # Shorter for faster testing
        num_workers=num_workers,
        buffer_size=1000,
        seed=42,
    )
    
    seen_hashes = set()
    duplicate_count = 0
    unique_tokens = set()
    
    print("Sampling batches...")
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        
        input_ids = batch["input_ids"]
        
        # Check for duplicate batches
        for j in range(input_ids.shape[0]):
            sample_hash = hash_tensor(input_ids[j])
            if sample_hash in seen_hashes:
                duplicate_count += 1
                print(f"⚠️  Found duplicate at batch {i}, sample {j}")
            seen_hashes.add(sample_hash)
            
            # Track unique tokens
            unique_tokens.update(input_ids[j].unique().tolist())
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_batches} batches...")
    
    total_samples = len(seen_hashes) + duplicate_count
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Total samples seen: {total_samples}")
    print(f"  Unique samples: {len(seen_hashes)}")
    print(f"  Duplicates: {duplicate_count}")
    print(f"  Duplicate rate: {duplicate_count / total_samples * 100:.2f}%")
    print(f"  Unique tokens: {len(unique_tokens)}")
    print(f"{'='*60}\n")
    
    if duplicate_count == 0:
        print("✅ No duplicates found - data pipeline is working correctly!")
    elif duplicate_count < total_samples * 0.01:  # Less than 1%
        print("⚠️  Very few duplicates found - might be acceptable due to shuffling")
    else:
        print("❌ Too many duplicates - data pipeline has issues!")
    
    return duplicate_count, total_samples


def test_first_tokens(num_batches=5, batch_size=2, num_workers=4):
    """Print first few tokens from each batch to visually inspect diversity."""
    print(f"\nShowing first tokens from {num_batches} batches:\n")
    
    train_loader, _ = create_dataloaders(
        batch_size=batch_size,
        max_length=128,
        num_workers=num_workers,
        buffer_size=1000,
        seed=42,
    )
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        
        input_ids = batch["input_ids"]
        print(f"Batch {i}:")
        for j in range(min(2, input_ids.shape[0])):
            tokens = input_ids[j][:20].tolist()
            print(f"  Sample {j}: {tokens}")
        print()


if __name__ == "__main__":
    print("Data Diversity Verification\n")
    print("="*60)
    
    # Test 1: Check for duplicates
    verify_diversity(num_batches=100, batch_size=4, num_workers=4)
    
    # Test 2: Visual inspection
    test_first_tokens(num_batches=5, batch_size=2, num_workers=4)
    
    print("\nDone! If you see duplicates, the data pipeline needs fixing.")
    print("If no duplicates, the fast loss drop is likely due to easy patterns in the data.")
