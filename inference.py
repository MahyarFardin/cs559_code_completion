#!/usr/bin/env python3
"""
Inference script for code completion model.
"""

import torch
import argparse
import json
import os
from train import Vocabulary
from model import CodeCompletionTransformer, ModelConfig

def load_model(model_path, vocab_path, config=None, device='cuda'):
    """Load trained model and vocabulary."""
    # Load vocabulary
    vocab = Vocabulary()
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        vocab.token_to_idx = vocab_data['token_to_idx']
        vocab.idx_to_token = {int(k): v for k, v in vocab_data['idx_to_token'].items()}
    
    # If config not provided, create one matching the vocabulary
    if config is None:
        config = ModelConfig()
        config.vocab_size = len(vocab.token_to_idx)
    
    # Create model with correct configuration
    model = CodeCompletionTransformer(config)
    loaded_obj = torch.load(model_path, map_location=device)
    # Support both raw state_dict and full checkpoints
    if isinstance(loaded_obj, dict) and 'model_state_dict' in loaded_obj:
        state_dict = loaded_obj['model_state_dict']
    else:
        state_dict = loaded_obj
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, vocab, config

def predict_next_token(model, vocab, context_tokens, device='cuda', top_k=5):
    """Predict next token given context."""
    # Encode context
    context_ids = vocab.encode(context_tokens, max_length=model.config.max_len, pad=True)
    input_ids = torch.tensor([context_ids], dtype=torch.long).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        last_logits = logits[0, -1, :]  # [vocab_size]
        
        # Get top-k predictions
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            token = vocab.idx_to_token[idx.item()]
            predictions.append((token, prob.item()))
    
    return predictions

def complete_line(model, vocab, context_tokens, max_tokens=20, device='cuda'):
    """Complete a line given context."""
    current_tokens = context_tokens.copy()
    completed = []
    
    for _ in range(max_tokens):
        predictions = predict_next_token(model, vocab, current_tokens, device, top_k=1)
        next_token = predictions[0][0]
        
        # Stop at EOL or end token
        if next_token in ['<EOL>', '</s>', '<PAD>']:
            break
        
        completed.append(next_token)
        current_tokens.append(next_token)
        
        # Truncate if too long
        if len(current_tokens) > model.config.max_len - 1:
            current_tokens = current_tokens[-(model.config.max_len - 1):]
    
    return completed

def main():
    parser = argparse.ArgumentParser(description="Code completion inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--vocab_path", type=str, default="vocab.json",
                        help="Path to vocabulary file")
    parser.add_argument("--task", type=str, choices=['token', 'line'], default='token',
                        help="Task type")
    parser.add_argument("--context", type=str, required=True,
                        help="Input context (space-separated tokens)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top predictions to show")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Auto-detect vocab and training params from model directory
    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    training_params_path = os.path.join(model_dir, 'training_params.json')

    run_vocab_path = os.path.join(model_dir, 'vocab.json')
    if os.path.exists(run_vocab_path):
        args.vocab_path = run_vocab_path
        print(f"Using vocab from model directory: {args.vocab_path}")
    
    config = ModelConfig()
    
    # If training parameters exist, use them to set max_length / architecture
    if os.path.exists(training_params_path):
        print(f"Loading training parameters from {training_params_path}...")
        with open(training_params_path, 'r') as f:
            training_params = json.load(f)
        if 'max_length' in training_params:
            config.max_len = training_params['max_length']
            print(f"Using max_length={config.max_len} from training parameters")
        for k in ['d_model', 'n_layer', 'n_head', 'd_ff', 'dropout']:
            if k in training_params:
                setattr(config, k, training_params[k])
    
    # Load vocabulary to get actual vocab size
    with open(args.vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    # Set vocab_size from actual vocabulary
    config.vocab_size = len(vocab_data['token_to_idx'])
    print(f"Using vocab_size={config.vocab_size} from vocabulary file")
    
    # Load model and vocabulary
    print("Loading model...")
    model, vocab, config = load_model(args.model_path, args.vocab_path, config, args.device)
    print("Model loaded!")
    
    # Parse context
    context_tokens = args.context.split()
    
    if args.task == 'token':
        # Token-level prediction
        print(f"\nContext: {' '.join(context_tokens)}")
        print("\nTop predictions for next token:")
        predictions = predict_next_token(model, vocab, context_tokens, args.device, args.top_k)
        for i, (token, prob) in enumerate(predictions, 1):
            print(f"  {i}. {token} (prob: {prob:.4f})")
    
    else:
        # Line-level completion
        print(f"\nContext: {' '.join(context_tokens)}")
        print("\nCompleting line...")
        completed = complete_line(model, vocab, context_tokens, device=args.device)
        print(f"Completion: {' '.join(completed)}")
        print(f"\nFull line: {' '.join(context_tokens)} {' '.join(completed)}")

if __name__ == "__main__":
    main()

