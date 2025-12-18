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
from torch.optim.lr_scheduler import OneCycleLR

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
        #
        # IMPORTANT:
        # We will train/evaluate using the logits at the *last context token* position
        # to predict `target`. We still append `target` here so the sequence length is
        # context_length+1, but we must NOT take logits from the final position (which
        # would correspond to the target token itself or padding).
        # input is ONLY the context
        input_tokens = context

        # If sequence is too long, keep the most recent tokens
        if len(input_tokens) > self.max_length:
            input_tokens = input_tokens[-self.max_length:]

        # Encode context
        input_ids = self.vocab.encode(input_tokens, max_length=self.max_length, pad=True)

        # Encode target as the next-token label
        target_idx = self.vocab.token_to_idx.get(target, self.vocab.token_to_idx['<UNK>'])

        # Context length after truncation (no target inside input)
        context_length = len(input_tokens)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'targets': torch.tensor(target_idx, dtype=torch.long),   # <- match train_v2.py expects 'targets'
            'context_lengths': torch.tensor(context_length, dtype=torch.long)  # <- optional, but match naming
        }


def collate_token_level(batch):
    """Collate function for token-level dataset."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    # context_lengths is already a tensor from dataset, just stack them
    context_lengths = torch.stack([item['context_lengths'] for item in batch])
    assert input_ids.ndim == 2 and targets.ndim == 1 and context_lengths.ndim == 1
    return {
        'input_ids': input_ids,
        'targets': targets,
        'context_lengths': context_lengths
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


def train_epoch_token(model, train_loader, optimizer, criterion, device, accumulation_steps=1, scheduler=None):
    """Train for one epoch (token-level)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        context_lengths = batch.get('context_lengths', None)
        
        # Forward pass
        logits = model(input_ids)  # [B, T, vocab_size]
        # Use logits at the last context token position to predict the target token.
        # Input is only context (no target), so we use context_lengths - 1 to get the last context token.
        if context_lengths is None:
            pred_logits = logits[:, -1, :]  # Back-compat fallback (uses last position, might be padding)
        else:
            # context_lengths gives actual context length, so last context token is at index (context_lengths - 1)
            pos = (context_lengths.to(device) - 1).clamp(min=0)  # [B]
            pred_logits = logits[torch.arange(logits.size(0), device=device), pos, :]  # [B, vocab]
        
        # Compute loss
        loss = criterion(pred_logits, targets)
        
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
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    # Handle remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_epoch_line(model, train_loader, optimizer, criterion, config, device, accumulation_steps=1, scheduler=None):
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
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        pbar.set_postfix({'loss': batch_loss_value})
    
    # Handle remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
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
            context_lengths = batch.get('context_lengths', None)
            
            # Forward pass
            logits = model(input_ids)
            # Input is only context, use context_lengths - 1 to get last context token position
            if context_lengths is None:
                pred_logits = logits[:, -1, :]  # Back-compat fallback
            else:
                pos = (context_lengths.to(device) - 1).clamp(min=0)  # [B]
                pred_logits = logits[torch.arange(logits.size(0), device=device), pos, :]  # [B, vocab]
            
            # Compute loss
            loss = criterion(pred_logits, targets)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(pred_logits, dim=-1)
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
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (1 = normal, >1 = accumulate)")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Early stopping patience (None = disabled, N = stop after N epochs without improvement)")

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
    
    # Vocabulary arguments
    parser.add_argument("--vocab_min_freq", type=int, default=10,
                        help="Minimum frequency for vocabulary tokens")
    parser.add_argument("--vocab_sample_lines", type=int, default=100000,
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
    print(f"Model: d_model={args.d_model}, n_layer={args.n_layer}, n_head={args.n_head}, d_ff={args.d_ff}, dropout={args.dropout}")
    print()
    
    # Create run directory
    # Note: Using a run_name without the timestamp at this stage allows resuming from a previous run
    # that used the same training parameters, provided the directory already exists.
    # Keep run names deterministic so identical configs can resume safely.
    do_tag = int(round(args.dropout * 100))
    run_name = (
        f"run_{args.task}_v2"
        f"_dm{args.d_model}_ly{args.n_layer}_hd{args.n_head}_ff{args.d_ff}_do{do_tag}"
        f"_lr{args.learning_rate:g}_wd{args.weight_decay:g}"
        f"_bs{args.batch_size}_ep{args.num_epochs}_len{args.max_length}_vocab{args.vocab_min_freq}"
    )
    if args.max_train_examples:
        run_name += f"_train{args.max_train_examples}"
    if args.accumulation_steps > 1:
        run_name += f"_acc{args.accumulation_steps}"
    
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
    config.d_model = args.d_model
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.d_ff = args.d_ff
    config.dropout = args.dropout

    if config.d_model % config.n_head != 0:
        raise ValueError(f"d_model ({config.d_model}) must be divisible by n_head ({config.n_head})")
    
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
    
    # Setup training components
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # --- Checkpoint Loading Mechanism ---
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Find latest checkpoint
    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    
    if checkpoint_files:
        try:
            # Extract epoch number from filenames (e.g., 'checkpoint_epoch_5.pt' -> 5)
            latest_epoch = max([int(f.split('_')[-1].replace('.pt', '')) for f in checkpoint_files])
            latest_checkpoint_file = os.path.join(output_dir, f'checkpoint_epoch_{latest_epoch}.pt')
            
            print(f"Found checkpoint: {latest_checkpoint_file}. Attempting to resume training...")
            checkpoint = torch.load(latest_checkpoint_file, map_location=args.device)

            # Refuse to resume if key hyperparameters changed (prevents subtle mismatches)
            ck_args = checkpoint.get('args', {})
            keys_to_match = [
                'task', 'max_length', 'vocab_min_freq',
                'd_model', 'n_layer', 'n_head', 'd_ff', 'dropout',
                'learning_rate', 'weight_decay'
            ]
            mismatches = []
            for k in keys_to_match:
                if k in ck_args and ck_args.get(k) != getattr(args, k, None):
                    mismatches.append((k, ck_args.get(k), getattr(args, k, None)))
            if mismatches:
                print("Found checkpoint, but config differs from current args; NOT resuming.")
                for k, old_v, new_v in mismatches[:10]:
                    print(f"  - {k}: checkpoint={old_v} vs current={new_v}")
                start_epoch = 0
                best_val_loss = float('inf')
                checkpoint = None
                raise RuntimeError("Checkpoint args mismatch")
            
            # Load model state (weights)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Move optimizer state tensors to the current device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(args.device)
            
            # Update training metadata
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            # Reset early stopping counter when resuming (allows patience more epochs from this point)
            epochs_without_improvement = 0
            
            print(f"Successfully loaded model and optimizer states. Resuming from Epoch {start_epoch + 1}/{args.num_epochs}.")
            print(f"Current best validation loss: {best_val_loss:.4f}")
            
        except Exception as e:
            print(f"Error loading checkpoint {latest_checkpoint_file}: {e}. Starting training from epoch 1.")
            start_epoch = 0
            best_val_loss = float('inf')
            epochs_without_improvement = 0

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
    
    # Create learning rate scheduler (OneCycleLR)
    total_steps = (args.num_epochs - start_epoch) * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1  # 10% warmup
    )
    
    # Training loop setup (model and optimizer are already setup/loaded)
    model_filename = f'best_model_{args.task}_level.pt'
    model_path = os.path.join(output_dir, model_filename)
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Loop starts from the epoch after the last saved checkpoint (start_epoch)
    for epoch in range(start_epoch, args.num_epochs): 
        current_epoch = epoch + 1
        print(f"\nEpoch {current_epoch}/{args.num_epochs}")
        print("-" * 60)
        
        # Train
        if args.task == 'token':
            train_loss = train_epoch_token(model, train_loader, optimizer, criterion, args.device, args.accumulation_steps, scheduler)
            val_loss, val_accuracy = validate_token(model, val_loader, criterion, args.device)
        else:  # line-level
            train_loss = train_epoch_line(model, train_loader, optimizer, criterion, config, args.device, args.accumulation_steps, scheduler)
            val_loss, val_accuracy = validate_line(model, val_loader, criterion, config, args.device)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        if val_accuracy is not None:
            print(f"Val Accuracy: {val_accuracy*100:.2f}%")
        
        # Save best model (model weights only) and track early stopping
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            improved = True
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved best model state (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience is not None:
                print(f"  No improvement for {epochs_without_improvement}/{args.early_stopping_patience} epochs")
            
        # --- Save Full Checkpoint after every epoch ---
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{current_epoch}.pt')
        checkpoint_to_save = {
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'args': vars(args), # Save arguments for context
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        torch.save(checkpoint_to_save, checkpoint_path)
        print(f"✓ Saved epoch checkpoint to {checkpoint_path}")
        
        # Early stopping check
        if args.early_stopping_patience is not None and epochs_without_improvement >= args.early_stopping_patience:
            print("\n" + "="*60)
            print(f"EARLY STOPPING: No improvement for {epochs_without_improvement} epochs (patience={args.early_stopping_patience})")
            print(f"Best validation loss was: {best_val_loss:.4f}")
            print("="*60)
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
