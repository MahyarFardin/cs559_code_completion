#!/usr/bin/env python3
"""
Training script v2 for code completion transformer model.
Clean, well-structured implementation with gradient accumulation support.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
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
            # Load examples into memory
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
        input_tokens = context + [target]
        
        # Encode to indices
        input_ids = self.vocab.encode(input_tokens, max_length=self.max_length, pad=True)
        target_idx = self.vocab.token_to_idx.get(target, self.vocab.token_to_idx['<UNK>'])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target': torch.tensor(target_idx, dtype=torch.long),
            'context_length': len(context)
        }


def collate_token_level(batch):
    """Collate function for token-level dataset."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    return {
        'input_ids': input_ids,
        'targets': targets
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
            # Load examples into memory
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


def collate_line_level(batch):
    """Collate function for line-level dataset."""
    context_ids = torch.stack([item['context_ids'] for item in batch])
    suffix_ids = torch.stack([item['suffix_ids'] for item in batch])
    return {
        'context_ids': context_ids,
        'suffix_ids': suffix_ids
    }


def train_epoch_token(model, train_loader, optimizer, criterion, device, accumulation_steps=1):
    """Train for one epoch (token-level)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward pass
        logits = model(input_ids)  # [B, T, vocab_size]
        last_logits = logits[:, -1, :]  # [B, vocab_size] - next token prediction
        
        # Compute loss
        loss = criterion(last_logits, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Update weights every N batches
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    # Handle remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_epoch_line(model, train_loader, optimizer, criterion, config, device, accumulation_steps=1):
    """Train for one epoch (line-level)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    pad_idx = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        context_ids = batch['context_ids'].to(device)  # [B, max_context_len]
        suffix_ids = batch['suffix_ids'].to(device)   # [B, max_suffix_len]
        
        # Predict suffix tokens sequentially
        steps = suffix_ids.size(1) - 1
        if steps <= 0:
            continue
        
        # Count valid (non-pad) tokens
        target_block = suffix_ids[:, :steps]
        total_valid_tokens = (target_block != pad_idx).sum().item()
        if total_valid_tokens == 0:
            continue
        
        # Start with context, append suffix tokens step-by-step
        current_input = context_ids
        batch_loss_value = 0.0
        
        for i in range(steps):
            next_target = suffix_ids[:, i]  # [B]
            mask = (next_target != pad_idx)
            
            if not mask.any():
                break
            
            # Forward pass
            logits = model(current_input)  # [B, T, vocab]
            next_logits = logits[:, -1, :]  # [B, vocab]
            
            # Compute loss only on valid targets
            loss_step = criterion(next_logits[mask], next_target[mask])
            
            # Weight by number of valid tokens
            valid_here = mask.sum().item()
            loss_scaled = loss_step * (valid_here / total_valid_tokens) / accumulation_steps
            
            # Backward pass
            loss_scaled.backward()
            batch_loss_value += float(loss_scaled.item() * accumulation_steps)
            
            # Teacher forcing: append ground-truth token
            next_token = suffix_ids[:, i:i+1]  # [B, 1]
            current_input = torch.cat([current_input, next_token], dim=1)
            
            # Truncate if too long
            if current_input.size(1) > config.max_len:
                current_input = current_input[:, -config.max_len:]
        
        total_loss += batch_loss_value
        num_batches += 1
        
        # Update weights every N batches
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        pbar.set_postfix({'loss': batch_loss_value})
    
    # Handle remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_token(model, val_loader, criterion, device):
    """Validate the model (token-level)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            last_logits = logits[:, -1, :]
            
            # Compute loss
            loss = criterion(last_logits, targets)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(last_logits, dim=-1)
            correct = (predictions == targets).sum().item()
            total_correct += correct
            total_tokens += targets.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return avg_loss, accuracy


def validate_line(model, val_loader, criterion, config, device):
    """Validate the model (line-level)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    pad_idx = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            context_ids = batch['context_ids'].to(device)
            suffix_ids = batch['suffix_ids'].to(device)
            
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
            
            total_loss += batch_loss_value
            num_batches += 1
            pbar.set_postfix({'loss': batch_loss_value})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # Line-level doesn't have a simple accuracy metric
    return avg_loss, None


def main():
    parser = argparse.ArgumentParser(description="Train code completion transformer model")
    
    # Data arguments
    parser.add_argument("--dataset_dir", type=str, default="completion_datasets",
                        help="Directory containing completion datasets")
    parser.add_argument("--tokenized_dir", type=str, default="token_completion",
                        help="Directory with tokenized files for vocabulary building")
    parser.add_argument("--task", type=str, choices=['token', 'line'], default='token',
                        help="Task type: token-level or line-level")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length (context length for line-level)")
    parser.add_argument("--max_suffix_length", type=int, default=64,
                        help="Maximum suffix length for line-level task")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (1 = normal, >1 = accumulate)")
    
    # Vocabulary arguments
    parser.add_argument("--vocab_min_freq", type=int, default=10,
                        help="Minimum frequency for vocabulary tokens")
    parser.add_argument("--vocab_sample_lines", type=int, default=50000,
                        help="Sample N lines for vocabulary building")
    
    # Data loading arguments
    parser.add_argument("--max_train_examples", type=int, default=None,
                        help="Limit number of training examples")
    parser.add_argument("--max_val_examples", type=int, default=10000,
                        help="Limit number of validation examples")
    parser.add_argument("--lazy_load", action="store_true", default=False,
                        help="Use lazy loading for datasets")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # System arguments
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("="*60)
    print("CODE COMPLETION TRANSFORMER TRAINING")
    print("="*60)
    print(f"Task: {args.task}-level")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.accumulation_steps}")
    if args.accumulation_steps > 1:
        print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print()
    
    # Create run directory
    run_name = f"run_{args.task}_v2_bs{args.batch_size}_ep{args.num_epochs}_len{args.max_length}_vocab{args.vocab_min_freq}"
    if args.max_train_examples:
        run_name += f"_train{args.max_train_examples}"
    if args.accumulation_steps > 1:
        run_name += f"_acc{args.accumulation_steps}"
    run_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = os.path.join('runs', run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Run directory: {output_dir}\n")
    
    # Save training parameters
    params_file = os.path.join(output_dir, 'training_params.json')
    with open(params_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training parameters to {params_file}\n")
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab_files = [
        os.path.join(args.tokenized_dir, "train.txt"),
        os.path.join(args.tokenized_dir, "dev.txt"),
        os.path.join(args.tokenized_dir, "test.txt")
    ]
    vocab_size = vocab.build_from_files(vocab_files, min_freq=args.vocab_min_freq, max_lines=args.vocab_sample_lines)
    print()
    
    # Create model config
    config = ModelConfig()
    config.vocab_size = vocab_size
    config.max_len = args.max_length
    
    # Warn if vocab is very large
    if vocab_size > 50000:
        print(f"WARNING: Vocabulary size ({vocab_size:,}) is very large!")
        print("Consider using --vocab_min_freq 10 or higher to reduce vocabulary size.\n")
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump({
            'token_to_idx': vocab.token_to_idx,
            'idx_to_token': {str(k): v for k, v in vocab.idx_to_token.items()}
        }, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}\n")
    
    # Create model
    model = CodeCompletionTransformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Model size: {param_count * 4 / 1024 / 1024:.2f} MB (float32)\n")
    
    # Create datasets
    if args.task == 'token':
        train_dataset = TokenLevelDataset(
            os.path.join(args.dataset_dir, "token_level", "train.jsonl"),
            vocab, args.max_length, lazy_load=args.lazy_load, max_examples=args.max_train_examples
        )
        val_dataset = TokenLevelDataset(
            os.path.join(args.dataset_dir, "token_level", "dev.jsonl"),
            vocab, args.max_length, lazy_load=args.lazy_load, max_examples=args.max_val_examples
        )
        collate_fn = collate_token_level
    else:  # line-level
        train_dataset = LineLevelDataset(
            os.path.join(args.dataset_dir, "line_level", "train.jsonl"),
            vocab, args.max_length, max_suffix_length=args.max_suffix_length,
            lazy_load=args.lazy_load, max_examples=args.max_train_examples
        )
        val_dataset = LineLevelDataset(
            os.path.join(args.dataset_dir, "line_level", "dev.jsonl"),
            vocab, args.max_length, max_suffix_length=args.max_suffix_length,
            lazy_load=args.lazy_load, max_examples=args.max_val_examples
        )
        collate_fn = collate_line_level
    
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Setup training
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Training loop
    best_val_loss = float('inf')
    model_filename = f'best_model_{args.task}_level.pt'
    model_path = os.path.join(output_dir, model_filename)
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 60)
        
        # Train
        if args.task == 'token':
            train_loss = train_epoch_token(model, train_loader, optimizer, criterion, args.device, args.accumulation_steps)
            val_loss, val_accuracy = validate_token(model, val_loader, criterion, args.device)
        else:  # line-level
            train_loss = train_epoch_line(model, train_loader, optimizer, criterion, config, args.device, args.accumulation_steps)
            val_loss, val_accuracy = validate_line(model, val_loader, criterion, config, args.device)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        if val_accuracy is not None:
            print(f"Val Accuracy: {val_accuracy*100:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
