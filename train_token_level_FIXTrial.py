#!/usr/bin/env python3
"""
Standalone training script for token-level code completion with gradient accumulation support.
This can be used to reduce GPU memory usage by accumulating gradients over multiple batches.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# Import from existing files
from model import CodeCompletionTransformer, ModelConfig
from train import Vocabulary, TokenLevelDataset, collate_token_level

def train_token_level(model, train_loader, val_loader, config, num_epochs=10, device='cuda', output_dir='.', accumulation_steps=1):
    """Train model for token-level completion with gradient accumulation support."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    best_val_loss = float('inf')
    model_path = os.path.join(output_dir, 'best_model_token_level.pt')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            logits = model(input_ids)  # [B, T, vocab_size]
            
            # Get logits for last position (next token prediction)
            last_logits = logits[:, -1, :]  # [B, vocab_size]
            
            # Compute loss
            loss = criterion(last_logits, targets)
            
            # Scale loss by accumulation steps (important for gradient accumulation!)
            loss = loss / accumulation_steps
            
            # Backward pass (accumulates gradients)
            loss.backward()
            
            train_loss += loss.item() * accumulation_steps  # Scale back for logging
            
            # Update weights every N batches
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()  # Zero gradients after update
            
            train_pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        # Handle remaining gradients if batch count isn't divisible by accumulation_steps
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                
                logits = model(input_ids)
                last_logits = logits[:, -1, :]
                loss = criterion(last_logits, targets)
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path} (val_loss: {avg_val_loss:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Train token-level code completion model with gradient accumulation")
    parser.add_argument("--task", type=str, default="token", choices=['token'],
                        help="Task type (always token for this script)")
    parser.add_argument("--dataset_dir", type=str, default="completion_datasets",
                        help="Directory containing completion datasets")
    parser.add_argument("--tokenized_dir", type=str, default="token_completion",
                        help="Directory with tokenized files for vocabulary building")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--vocab_min_freq", type=int, default=10,
                        help="Minimum frequency for vocabulary tokens (higher = smaller vocab)")
    parser.add_argument("--vocab_sample_lines", type=int, default=50000,
                        help="Sample N lines for vocabulary building (None = all)")
    parser.add_argument("--max_train_examples", type=int, default=None,
                        help="Limit number of training examples to load")
    parser.add_argument("--max_val_examples", type=int, default=10000,
                        help="Limit number of validation examples to load")
    parser.add_argument("--lazy_load", action="store_true", default=False,
                        help="Use lazy loading for datasets")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (1 = normal training, >1 = accumulate gradients)")
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    if args.accumulation_steps > 1:
        print(f"Gradient accumulation: updating every {args.accumulation_steps} batches")
        print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    
    # Create run directory based on parameters
    run_name = f"run_token_bs{args.batch_size}_ep{args.num_epochs}_len{args.max_length}_vocab{args.vocab_min_freq}"
    if args.max_train_examples:
        run_name += f"_train{args.max_train_examples}"
    if args.max_val_examples and args.max_val_examples != 10000:
        run_name += f"_val{args.max_val_examples}"
    if args.accumulation_steps > 1:
        run_name += f"_acc{args.accumulation_steps}"
    run_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = os.path.join('runs', run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRun directory: {output_dir}")
    
    # Save training parameters to run directory
    params_file = os.path.join(output_dir, 'training_params.json')
    with open(params_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training parameters to {params_file}")
    
    # Build vocabulary from tokenized files
    vocab = Vocabulary()
    vocab_files = [
        os.path.join(args.tokenized_dir, "train.txt"),
        os.path.join(args.tokenized_dir, "dev.txt"),
        os.path.join(args.tokenized_dir, "test.txt")
    ]
    vocab_size = vocab.build_from_files(vocab_files, min_freq=args.vocab_min_freq, max_lines=args.vocab_sample_lines)
    
    # Update config with actual vocab size
    config = ModelConfig()
    config.vocab_size = vocab_size
    config.max_len = args.max_length
    
    # Warn if vocab is very large
    if vocab_size > 50000:
        print(f"\nWARNING: Vocabulary size ({vocab_size:,}) is very large!")
        print("Consider using --vocab_min_freq 10 or higher to reduce vocabulary size.")
        print("Large vocabularies lead to very large models and slow training.\n")
    
    # Save vocabulary to run directory
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump({
            'token_to_idx': vocab.token_to_idx,
            'idx_to_token': {str(k): v for k, v in vocab.idx_to_token.items()}
        }, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Create model
    model = CodeCompletionTransformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets and dataloaders
    train_dataset = TokenLevelDataset(
        os.path.join(args.dataset_dir, "token_level", "train.jsonl"),
        vocab, args.max_length, lazy_load=args.lazy_load, max_examples=args.max_train_examples
    )
    val_dataset = TokenLevelDataset(
        os.path.join(args.dataset_dir, "token_level", "dev.jsonl"),
        vocab, args.max_length, lazy_load=args.lazy_load, max_examples=args.max_val_examples
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_token_level,
                             num_workers=args.num_workers, pin_memory=True if args.device == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, collate_fn=collate_token_level,
                           num_workers=args.num_workers, pin_memory=True if args.device == 'cuda' else False)
    
    # Train model
    train_token_level(model, train_loader, val_loader, config, 
                     args.num_epochs, args.device, output_dir, args.accumulation_steps)
    
    print(f"\nTraining complete! All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
