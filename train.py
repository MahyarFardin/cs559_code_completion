#!/usr/bin/env python3
"""
Training script for code completion model.
Supports both token-level and line-level completion tasks.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from model import CodeCompletionTransformer, ModelConfig

class Vocabulary:
    """Build vocabulary from tokenized data."""
    
    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<s>', '</s>', '<EOL>']
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.token_counts = Counter()
        
        # Add special tokens first
        for token in self.special_tokens:
            self.add_token(token)
    
    def add_token(self, token):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
    
    def build_from_files(self, file_paths, min_freq=1, max_lines=None):
        """Build vocabulary from tokenized text files."""
        print("Building vocabulary...")
        line_count = 0
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            print(f"  Processing {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_lines and line_count >= max_lines:
                        break
                    tokens = line.strip().split()
                    self.token_counts.update(tokens)
                    line_count += 1
                    if line_count % 10000 == 0:
                        print(f"    Processed {line_count} lines...")
                    if max_lines and line_count >= max_lines:
                        break
        
        print(f"  Found {len(self.token_counts)} unique tokens")
        print("  Building vocabulary...")
        # Add tokens that meet minimum frequency
        for token, count in self.token_counts.items():
            if count >= min_freq:
                self.add_token(token)
        
        print(f"Vocabulary size: {len(self.token_to_idx)}")
        return len(self.token_to_idx)
    
    def encode(self, tokens, max_length=None, pad=True):
        """Convert tokens to indices."""
        indices = [self.token_to_idx.get(token, self.token_to_idx['<UNK>']) for token in tokens]
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            elif pad and len(indices) < max_length:
                pad_idx = self.token_to_idx['<PAD>']
                indices = indices + [pad_idx] * (max_length - len(indices))
        
        return indices
    
    def decode(self, indices):
        """Convert indices to tokens."""
        return [self.idx_to_token.get(idx, '<UNK>') for idx in indices]

class TokenLevelDataset(Dataset):
    """Dataset for token-level code completion."""
    
    def __init__(self, jsonl_file, vocab, max_length=256, lazy_load=True, max_examples=None):
        self.vocab = vocab
        self.max_length = max_length
        self.jsonl_file = jsonl_file
        self.lazy_load = lazy_load
        self.max_examples = max_examples
        
        if lazy_load:
            # Count lines without loading all data
            print(f"Counting examples in {jsonl_file}...")
            self.num_examples = 0
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    self.num_examples += 1
                    if max_examples and self.num_examples >= max_examples:
                        break
                    if self.num_examples % 10000 == 0:
                        print(f"  Counted {self.num_examples} examples...")
            if max_examples:
                self.num_examples = min(self.num_examples, max_examples)
            print(f"Found {self.num_examples} examples (lazy loading enabled)")
            self.examples = None
        else:
            # Load examples into memory (with optional limit)
            self.examples = []
            print(f"Loading {jsonl_file}...")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_examples and len(self.examples) >= max_examples:
                        break
                    example = json.loads(line)
                    self.examples.append(example)
                    if len(self.examples) % 10000 == 0:
                        print(f"  Loaded {len(self.examples)} examples...")
            print(f"Loaded {len(self.examples)} examples")
            self.num_examples = len(self.examples)
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if self.lazy_load:
            # Use seek-based access for faster random access
            # Cache file positions for efficiency
            if not hasattr(self, '_file_positions'):
                # Build position cache on first access
                self._file_positions = []
                with open(self.jsonl_file, 'rb') as f:
                    pos = 0
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        self._file_positions.append(pos)
                        pos = f.tell()
            
            # Seek to position and read line
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                f.seek(self._file_positions[idx])
                line = f.readline()
                example = json.loads(line)
        else:
            example = self.examples[idx]
        
        context = example['context'].split()
        target = example['target']
        
        # Create input sequence: context + target
        #
        # IMPORTANT:
        # We will train/evaluate using the logits at the *last context token* position
        # to predict `target`. Do NOT use logits at the last position.
        input_tokens = context + [target]
        
        # If sequence is too long, keep most recent tokens so `target` stays last.
        if len(input_tokens) > self.max_length:
            input_tokens = input_tokens[-self.max_length:]
        
        # Encode to indices
        input_ids = self.vocab.encode(input_tokens, max_length=self.max_length, pad=True)
        target_idx = self.vocab.token_to_idx.get(target, self.vocab.token_to_idx['<UNK>'])
        
        context_length = max(0, len(input_tokens) - 1)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target': torch.tensor(target_idx, dtype=torch.long),
            'context_length': context_length
        }

class LineLevelDataset(Dataset):
    """Dataset for line-level code completion."""
    
    def __init__(self, jsonl_file, vocab, max_context_length=256, max_suffix_length=64, lazy_load=True, max_examples=None):
        self.vocab = vocab
        self.max_context_length = max_context_length
        self.max_suffix_length = max_suffix_length
        self.jsonl_file = jsonl_file
        self.lazy_load = lazy_load
        self.max_examples = max_examples
        
        if lazy_load:
            # Count lines without loading all data
            print(f"Counting examples in {jsonl_file}...")
            self.num_examples = 0
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    self.num_examples += 1
                    if max_examples and self.num_examples >= max_examples:
                        break
                    if self.num_examples % 10000 == 0:
                        print(f"  Counted {self.num_examples} examples...")
            if max_examples:
                self.num_examples = min(self.num_examples, max_examples)
            print(f"Found {self.num_examples} examples (lazy loading enabled)")
            self.examples = None
        else:
            # Load examples into memory (with optional limit)
            self.examples = []
            print(f"Loading {jsonl_file}...")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_examples and len(self.examples) >= max_examples:
                        break
                    example = json.loads(line)
                    self.examples.append(example)
                    if len(self.examples) % 10000 == 0:
                        print(f"  Loaded {len(self.examples)} examples...")
            print(f"Loaded {len(self.examples)} examples")
            self.num_examples = len(self.examples)
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if self.lazy_load:
            # Use seek-based access for faster random access
            # Cache file positions for efficiency
            if not hasattr(self, '_file_positions'):
                # Build position cache on first access
                self._file_positions = []
                with open(self.jsonl_file, 'rb') as f:
                    pos = 0
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        self._file_positions.append(pos)
                        pos = f.tell()
            
            # Seek to position and read line
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                f.seek(self._file_positions[idx])
                line = f.readline()
                example = json.loads(line)
        else:
            example = self.examples[idx]
        
        previous_lines = example['previous_lines'].split() if example['previous_lines'] else []
        prefix = example['prefix'].split()
        suffix = example['suffix'].split()
        
        # Combine context: previous_lines + prefix
        context = previous_lines + ['<EOL>'] + prefix if previous_lines else prefix
        
        # Truncate if needed
        if len(context) > self.max_context_length:
            context = context[-self.max_context_length:]
        
        if len(suffix) > self.max_suffix_length:
            suffix = suffix[:self.max_suffix_length]
        
        # Encode
        context_ids = self.vocab.encode(context, max_length=self.max_context_length, pad=True)
        suffix_ids = self.vocab.encode(suffix, max_length=self.max_suffix_length, pad=True)
        
        return {
            'context_ids': torch.tensor(context_ids, dtype=torch.long),
            'suffix_ids': torch.tensor(suffix_ids, dtype=torch.long),
            'context_length': len(context),
            'suffix_length': len(suffix)
        }

def collate_token_level(batch):
    """Collate function for token-level dataset."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    context_lengths = torch.tensor([item['context_length'] for item in batch], dtype=torch.long)
    return {
        'input_ids': input_ids,
        'targets': targets,
        'context_lengths': context_lengths
    }

def collate_line_level(batch):
    """Collate function for line-level dataset."""
    context_ids = torch.stack([item['context_ids'] for item in batch])
    suffix_ids = torch.stack([item['suffix_ids'] for item in batch])
    return {
        'context_ids': context_ids,
        'suffix_ids': suffix_ids
    }

def train_token_level(model, train_loader, val_loader, config, num_epochs=10, device='cuda', output_dir='.'):
    """Train model for token-level completion."""
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
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            context_lengths = batch.get('context_lengths', None)
            
            # Forward pass
            logits = model(input_ids)  # [B, T, vocab_size]
            
            # Predict next token using logits at last *context token* position
            if context_lengths is None:
                pred_logits = logits[:, -1, :]
            else:
                pos = (context_lengths.to(device) - 1).clamp(min=0)
                pred_logits = logits[torch.arange(logits.size(0), device=device), pos, :]
            
            # Compute loss
            loss = criterion(pred_logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                context_lengths = batch.get('context_lengths', None)
                
                logits = model(input_ids)
                if context_lengths is None:
                    pred_logits = logits[:, -1, :]
                else:
                    pos = (context_lengths.to(device) - 1).clamp(min=0)
                    pred_logits = logits[torch.arange(logits.size(0), device=device), pos, :]
                loss = criterion(pred_logits, targets)
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path} (val_loss: {avg_val_loss:.4f})")

def train_line_level(model, train_loader, val_loader, config,
                     num_epochs=10, device='cuda', output_dir='.'):
    """
    Train a model for line-level code completion using teacher forcing.

    At each step, the model predicts the next suffix token given:
      (context tokens) + (previous ground-truth suffix tokens)

    The context grows token-by-token (ground truth) and is truncated to config.max_len.
    Loss is computed only on non-padding targets.
    """

    # Move model to device (GPU/CPU)
    model = model.to(device)

    # Optimizer and loss (PAD tokens are ignored)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    pad_idx = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Enable mixed precision on CUDA if available
    use_amp = (str(device).startswith("cuda") and torch.cuda.is_available())
    try:
        from torch import amp
        scaler = amp.GradScaler('cuda', enabled=use_amp)
        autocast = lambda: amp.autocast(device_type='cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    best_val_loss = float('inf')
    model_path = os.path.join(output_dir, 'best_model_line_level.pt')

    for epoch in range(num_epochs):
        #################
        # Training loop
        #################
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in train_pbar:
            # Batch contains padded context and padded suffix
            context_ids = batch['context_ids'].to(device)  # [B, max_context_len]
            suffix_ids  = batch['suffix_ids'].to(device)   # [B, max_suffix_len]

            # We predict suffix tokens (excluding the last position, which has no "next" token)
            steps = suffix_ids.size(1) - 1
            if steps <= 0:
                continue

            # Count how many non-pad target tokens exist in this batch
            # (used to compute a mean loss over real tokens)
            target_block = suffix_ids[:, :steps]
            total_valid_tokens = (target_block != pad_idx).sum().item()
            if total_valid_tokens == 0:
                continue

            # Reset gradients
            optimizer.zero_grad(set_to_none=True)

            # Start input as the (padded) context; we will append ground-truth suffix tokens step-by-step
            current_input = context_ids
            batch_loss_value = 0.0

            for i in range(steps):
                # Target token at this step
                next_target = suffix_ids[:, i]          # [B]
                mask = (next_target != pad_idx)         # only compute loss where target is not PAD

                # If every target is PAD at this step, there is nothing to predict beyond this point
                if not mask.any():
                    break

                # Forward pass (optionally in mixed precision)
                with autocast():
                    logits = model(current_input)       # [B, T, vocab]
                    next_logits = logits[:, -1, :]      # [B, vocab] -> prediction for next token

                    # Compute loss only on valid (non-pad) targets
                    loss_step = criterion(next_logits[mask], next_target[mask])

                    # Weight this step's contribution proportional to how many valid targets it has,
                    # so the total loss is effectively an average per valid target token.
                    valid_here = mask.sum().item()
                    loss_scaled = loss_step * (valid_here / total_valid_tokens)

                # Backpropagate this step's contribution
                scaler.scale(loss_scaled).backward()
                batch_loss_value += float(loss_scaled.item())

                # Teacher forcing: append the ground-truth token for the next prediction step
                next_token = suffix_ids[:, i:i+1]       # [B, 1]
                current_input = torch.cat([current_input, next_token], dim=1)

                # Keep only the most recent config.max_len tokens
                if current_input.size(1) > config.max_len:
                    current_input = current_input[:, -config.max_len:]

            # Apply gradient clipping and optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += batch_loss_value
            train_pbar.set_postfix({'loss': batch_loss_value})

        avg_train_loss = train_loss / max(1, len(train_loader))

        #################
        # Validation loop
        #################
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                context_ids = batch['context_ids'].to(device)
                suffix_ids  = batch['suffix_ids'].to(device)

                steps = suffix_ids.size(1) - 1
                if steps <= 0:
                    continue

                target_block = suffix_ids[:, :steps]
                total_valid_tokens = (target_block != pad_idx).sum().item()
                if total_valid_tokens == 0:
                    continue

                current_input = context_ids
                batch_loss_value = 0.0

                for i in range(steps):
                    next_target = suffix_ids[:, i]
                    mask = (next_target != pad_idx)
                    if not mask.any():
                        break

                    with autocast():
                        logits = model(current_input)
                        next_logits = logits[:, -1, :]
                        loss_step = criterion(next_logits[mask], next_target[mask])

                        valid_here = mask.sum().item()
                        loss_scaled = loss_step * (valid_here / total_valid_tokens)

                    batch_loss_value += float(loss_scaled.item())

                    next_token = suffix_ids[:, i:i+1]
                    current_input = torch.cat([current_input, next_token], dim=1)
                    if current_input.size(1) > config.max_len:
                        current_input = current_input[:, -config.max_len:]

                val_loss += batch_loss_value
                val_pbar.set_postfix({'loss': batch_loss_value})

        avg_val_loss = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save best checkpoint by validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path} (val_loss: {avg_val_loss:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Train code completion model")
    parser.add_argument("--dataset_dir", type=str, default="completion_datasets",
                        help="Directory containing completion datasets")
    parser.add_argument("--task", type=str, choices=['token', 'line'], default='token',
                        help="Task type: token-level or line-level")
    parser.add_argument("--tokenized_dir", type=str, default="token_completion",
                        help="Directory with tokenized files for vocabulary building")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model architecture arguments (override ModelConfig defaults)
    parser.add_argument("--d_model", type=int, default=ModelConfig.d_model,
                        help="Transformer width / embedding dimension")
    parser.add_argument("--n_layer", type=int, default=ModelConfig.n_layer,
                        help="Number of transformer blocks (depth)")
    parser.add_argument("--n_head", type=int, default=ModelConfig.n_head,
                        help="Number of attention heads (must divide d_model)")
    parser.add_argument("--d_ff", type=int, default=ModelConfig.d_ff,
                        help="Feed-forward (MLP) hidden dimension")
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout,
                        help="Dropout probability")
    parser.add_argument("--vocab_min_freq", type=int, default=10,
                        help="Minimum frequency for vocabulary tokens (higher = smaller vocab)")
    parser.add_argument("--vocab_sample_lines", type=int, default=50000,
                        help="Sample N lines for vocabulary building (None = all)")
    parser.add_argument("--max_train_examples", type=int, default=None,
                        help="Limit number of training examples to load (for testing/smaller models)")
    parser.add_argument("--max_val_examples", type=int, default=10000,
                        help="Limit number of validation examples to load")
    parser.add_argument("--lazy_load", action="store_true", default=False,
                        help="Use lazy loading for datasets (saves memory but slower). Omit this flag to disable lazy loading.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers (0 = single process)")
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Create run directory based on parameters
    do_tag = int(round(args.dropout * 100))
    run_name = (
        f"run_{args.task}"
        f"_dm{args.d_model}_ly{args.n_layer}_hd{args.n_head}_ff{args.d_ff}_do{do_tag}"
        f"_bs{args.batch_size}_ep{args.num_epochs}_len{args.max_length}_vocab{args.vocab_min_freq}"
    )
    if args.max_train_examples:
        run_name += f"_train{args.max_train_examples}"
    if args.max_val_examples and args.max_val_examples != 10000:
        run_name += f"_val{args.max_val_examples}"
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
    config.d_model = args.d_model
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.d_ff = args.d_ff
    config.dropout = args.dropout

    if config.d_model % config.n_head != 0:
        raise ValueError(f"d_model ({config.d_model}) must be divisible by n_head ({config.n_head})")
    
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
    if args.task == 'token':
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
        
        train_token_level(model, train_loader, val_loader, config, 
                         args.num_epochs, args.device, output_dir)
    
    else:  # line-level
        train_dataset = LineLevelDataset(
            os.path.join(args.dataset_dir, "line_level", "train.jsonl"),
            vocab, args.max_length, max_suffix_length=64, lazy_load=args.lazy_load, max_examples=args.max_train_examples
        )
        val_dataset = LineLevelDataset(
            os.path.join(args.dataset_dir, "line_level", "dev.jsonl"),
            vocab, args.max_length, max_suffix_length=64, lazy_load=args.lazy_load, max_examples=args.max_val_examples
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, collate_fn=collate_line_level,
                                 num_workers=args.num_workers, pin_memory=True if args.device == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, collate_fn=collate_line_level,
                               num_workers=args.num_workers, pin_memory=True if args.device == 'cuda' else False)
        
        train_line_level(model, train_loader, val_loader, config, 
                        args.num_epochs, args.device, output_dir)
    
    print(f"\nTraining complete! All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()

