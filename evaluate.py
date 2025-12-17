#!/usr/bin/env python3
"""
Evaluate trained model on test set.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import CodeCompletionTransformer, ModelConfig
from train import Vocabulary, TokenLevelDataset, LineLevelDataset, collate_token_level, collate_line_level

def evaluate_token_level(model, test_loader, device='cuda'):
    """Evaluate token-level model on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            context_lengths = batch.get('context_lengths', None)
            
            # Forward pass
            logits = model(input_ids)
            if context_lengths is None:
                pred_logits = logits[:, -1, :]
            else:
                pos = (context_lengths.to(device) - 1).clamp(min=0)
                pred_logits = logits[torch.arange(logits.size(0), device=device), pos, :]
            
            # Compute loss
            loss = criterion(pred_logits, targets)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(pred_logits, dim=-1)
            correct = (predictions == targets).sum().item()
            total_correct += correct
            total_tokens += targets.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_examples': total_tokens
    }

def evaluate_line_level(model, test_loader, config, device='cuda'):
    """Evaluate line-level model on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            context_ids = batch['context_ids'].to(device)
            suffix_ids = batch['suffix_ids'].to(device)
            
            # Predict suffix tokens sequentially
            current_input = context_ids
            batch_loss = 0
            batch_tokens = 0
            
            for i in range(suffix_ids.size(1) - 1):
                logits = model(current_input)
                next_logits = logits[:, -1, :]
                next_target = suffix_ids[:, i]
                
                loss = criterion(next_logits, next_target)
                batch_loss += loss.item()
                batch_tokens += 1
                
                # Append predicted token for next step
                next_token = suffix_ids[:, i:i+1]
                current_input = torch.cat([current_input, next_token], dim=1)
                if current_input.size(1) > config.max_len:
                    current_input = current_input[:, -config.max_len:]
            
            total_loss += batch_loss / batch_tokens if batch_tokens > 0 else 0
            total_tokens += 1
    
    avg_loss = total_loss / len(test_loader)
    
    return {
        'loss': avg_loss,
        'total_examples': total_tokens
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--vocab_path", type=str, default="vocab.json",
                        help="Path to vocabulary file")
    parser.add_argument("--dataset_dir", type=str, default="completion_datasets",
                        help="Directory containing test datasets")
    parser.add_argument("--task", type=str, choices=['token', 'line'], default='token',
                        help="Task type")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--max_test_examples", type=int, default=None,
                        help="Limit number of test examples (None = all)")
    parser.add_argument("--lazy_load", action="store_true", default=True,
                        help="Use lazy loading for test dataset (saves memory, recommended)")
    parser.add_argument("--no_lazy_load", dest="lazy_load", action="store_false",
                        help="Disable lazy loading (loads all examples into memory)")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Model architecture overrides (if training_params.json is missing these or you want to override)
    parser.add_argument("--d_model", type=int, default=None,
                        help="Override d_model (auto-detected from training_params.json if available)")
    parser.add_argument("--n_layer", type=int, default=None,
                        help="Override n_layer (auto-detected from training_params.json if available)")
    parser.add_argument("--n_head", type=int, default=None,
                        help="Override n_head (auto-detected from training_params.json if available)")
    parser.add_argument("--d_ff", type=int, default=None,
                        help="Override d_ff (auto-detected from training_params.json if available)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout (auto-detected from training_params.json if available)")
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # If user passed a directory, auto-detect the model file
    if os.path.isdir(args.model_path):
        model_dir = os.path.abspath(args.model_path)
        # Look for best_model_{task}_level.pt or best_model.pt
        task_suffix = f"{args.task}_level"
        candidate_paths = [
            os.path.join(model_dir, f"best_model_{task_suffix}.pt"),
            os.path.join(model_dir, "best_model.pt"),
            os.path.join(model_dir, "best_model_token_level.pt"),
            os.path.join(model_dir, "best_model_line_level.pt"),
        ]
        found = None
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                found = candidate
                break
        if found:
            print(f"Auto-detected model file: {found}")
            args.model_path = found
        else:
            raise FileNotFoundError(
                f"Model directory provided but no model file found. Tried: {candidate_paths}"
            )
    
    # Auto-detect vocab and training params from model directory
    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    
    # Always prefer vocab.json in the same directory as the model (if it exists)
    run_vocab_path = os.path.join(model_dir, 'vocab.json')
    if os.path.exists(run_vocab_path):
        args.vocab_path = run_vocab_path
        print(f"Using vocab from model directory: {args.vocab_path}")
    elif args.vocab_path == "vocab.json":
        print(f"Using default vocab.json (not found in {model_dir})")
    
    training_params = None
    # Try to load training parameters for max_length / architecture if in run directory
    training_params_path = os.path.join(model_dir, 'training_params.json')
    if os.path.exists(training_params_path):
        print(f"Loading training parameters from {training_params_path}...")
        with open(training_params_path, 'r') as f:
            training_params = json.load(f)
        if 'max_length' in training_params and args.max_length == 256:  # Only override if default
            args.max_length = training_params['max_length']
            print(f"Using max_length={args.max_length} from training parameters")
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab_path}...")
    vocab = Vocabulary()
    with open(args.vocab_path, 'r') as f:
        vocab_data = json.load(f)
        vocab.token_to_idx = vocab_data['token_to_idx']
        vocab.idx_to_token = {int(k): v for k, v in vocab_data['idx_to_token'].items()}
    vocab_size = len(vocab.token_to_idx)
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Create config
    config = ModelConfig()
    config.vocab_size = vocab_size
    config.max_len = args.max_length
    
    # Load architecture params (priority: CLI args > training_params.json > classic defaults)
    arch_params = {}
    if isinstance(training_params, dict):
        for k in ['d_model', 'n_layer', 'n_head', 'd_ff', 'dropout']:
            if k in training_params:
                arch_params[k] = training_params[k]
    
    # Apply architecture params (CLI overrides take precedence)
    if args.d_model is not None:
        config.d_model = args.d_model
    elif 'd_model' in arch_params:
        config.d_model = arch_params['d_model']
    else:
        config.d_model = 512  # Classic default for old runs
    
    if args.n_layer is not None:
        config.n_layer = args.n_layer
    elif 'n_layer' in arch_params:
        config.n_layer = arch_params['n_layer']
    else:
        config.n_layer = 6  # Classic default
    
    if args.n_head is not None:
        config.n_head = args.n_head
    elif 'n_head' in arch_params:
        config.n_head = arch_params['n_head']
    else:
        config.n_head = 8  # Classic default
    
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    elif 'd_ff' in arch_params:
        config.d_ff = arch_params['d_ff']
    else:
        config.d_ff = 2048  # Classic default
    
    if args.dropout is not None:
        config.dropout = args.dropout
    elif 'dropout' in arch_params:
        config.dropout = arch_params['dropout']
    else:
        config.dropout = 0.1  # Classic default
    
    if not isinstance(training_params, dict) or not any(k in training_params for k in ['d_model', 'n_layer']):
        print("Note: Using classic architecture defaults (d_model=512, n_layer=6, n_head=8, d_ff=2048, dropout=0.1)")
        print("      If this causes shape mismatches, use --d_model --n_layer --n_head --d_ff --dropout to override")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    print(f"Model config: d_model={config.d_model}, n_layer={config.n_layer}, n_head={config.n_head}, "
          f"d_ff={config.d_ff}, dropout={config.dropout}, vocab_size={config.vocab_size}, max_len={config.max_len}")
    
    # Handle device mapping: always load to CPU first, then move to target device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        args.device = 'cpu'
    
    # Load model (map to CPU first to avoid device mismatch errors)
    map_location = 'cpu' if args.device == 'cpu' else args.device
    model = CodeCompletionTransformer(config)
    loaded_obj = torch.load(args.model_path, map_location=map_location)
    # Support both:
    # - raw state_dict files: torch.save(model.state_dict(), path)
    # - full checkpoints: torch.save({'model_state_dict': ..., ...}, path)
    if isinstance(loaded_obj, dict) and 'model_state_dict' in loaded_obj:
        state_dict = loaded_obj['model_state_dict']
    else:
        state_dict = loaded_obj
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        if "size mismatch" in str(e) or "shape" in str(e).lower():
            print("\n" + "="*60)
            print("ERROR: Model architecture mismatch!")
            print("="*60)
            print("The saved model has a different architecture than the config being used.")
            print(f"\nAttempted config: d_model={config.d_model}, n_layer={config.n_layer}, "
                  f"n_head={config.n_head}, d_ff={config.d_ff}")
            print("\nTo fix this, you can:")
            print("1. Manually inspect the model checkpoint to determine its architecture")
            print("2. Or re-train with the architecture flags so training_params.json includes them")
            print("="*60)
        raise
    model = model.to(args.device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Create test dataset
    print("Loading test dataset...")
    if args.lazy_load:
        print("Using lazy loading (memory efficient)")
    else:
        print("Loading all examples into memory (may be slow for large datasets)")
    
    if args.task == 'token':
        test_dataset = TokenLevelDataset(
            os.path.join(args.dataset_dir, "token_level", "test.jsonl"),
            vocab, args.max_length, lazy_load=args.lazy_load, max_examples=args.max_test_examples
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                               shuffle=False, collate_fn=collate_token_level,
                               num_workers=args.num_workers)
        
        print(f"Test examples: {len(test_dataset):,}")
        print("\nEvaluating on test set...")
        results = evaluate_token_level(model, test_loader, args.device)
        
        print("\n" + "="*50)
        print("TEST SET RESULTS (Token-Level)")
        print("="*50)
        print(f"Test Loss: {results['loss']:.4f}")
        print(f"Test Accuracy: {results['accuracy']*100:.2f}%")
        print(f"Total Examples: {results['total_examples']:,}")
        print("="*50)
    
    else:  # line-level
        test_dataset = LineLevelDataset(
            os.path.join(args.dataset_dir, "line_level", "test.jsonl"),
            vocab, args.max_length, max_suffix_length=64, lazy_load=args.lazy_load, 
            max_examples=args.max_test_examples
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                               shuffle=False, collate_fn=collate_line_level,
                               num_workers=args.num_workers)
        
        print(f"Test examples: {len(test_dataset):,}")
        print("\nEvaluating on test set...")
        results = evaluate_line_level(model, test_loader, config, args.device)
        
        print("\n" + "="*50)
        print("TEST SET RESULTS (Line-Level)")
        print("="*50)
        print(f"Test Loss: {results['loss']:.4f}")
        print(f"Total Examples: {results['total_examples']:,}")
        print("="*50)
    
    # Determine output directory based on model path
    # If model is in a runs directory, save results there
    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    if 'runs' in model_dir:
        # Model is in a run directory, save results there
        output_dir = model_dir
    else:
        # Model is not in a run directory, create a results directory based on model name
        model_name = os.path.basename(args.model_path).replace('.pt', '')
        output_dir = os.path.join('runs', f'eval_{model_name}')
        os.makedirs(output_dir, exist_ok=True)
    
    # Add evaluation parameters to results
    results['evaluation_params'] = {
        'model_path': args.model_path,
        'task': args.task,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'max_test_examples': args.max_test_examples,
        'device': args.device
    }
    
    # Save results
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()

