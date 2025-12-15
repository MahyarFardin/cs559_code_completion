#!/usr/bin/env python3
"""
Diagnostic script to identify why accuracy dropped from 93% to 6%.
Checks for common issues like vocabulary mismatch, model configuration, etc.
"""

import os
import json
import torch
import argparse
from train import Vocabulary
from model import CodeCompletionTransformer, ModelConfig

def check_vocabulary_consistency(vocab_path, model_path):
    """Check if vocabulary matches model expectations."""
    print("="*60)
    print("VOCABULARY CHECK")
    print("="*60)
    
    # Load vocabulary
    vocab = Vocabulary()
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        vocab.token_to_idx = vocab_data['token_to_idx']
        vocab.idx_to_token = {int(k): v for k, v in vocab_data['idx_to_token'].items()}
    
    vocab_size = len(vocab.token_to_idx)
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Check special tokens
    special_tokens = ['<PAD>', '<UNK>', '<s>', '</s>', '<EOL>']
    print("\nSpecial token indices:")
    for token in special_tokens:
        if token in vocab.token_to_idx:
            idx = vocab.token_to_idx[token]
            print(f"  {token}: {idx}")
        else:
            print(f"  {token}: NOT FOUND (WARNING!)")
    
    # Check PAD token is 0
    if vocab.token_to_idx.get('<PAD>', -1) != 0:
        print("\nWARNING: <PAD> token is not index 0!")
        print(f"   Current <PAD> index: {vocab.token_to_idx.get('<PAD>', 'NOT FOUND')}")
    
    # Try to load model and check vocab size
    print("\n" + "="*60)
    print("MODEL CHECK")
    print("="*60)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check token_embedding weight shape
        if 'token_embedding.weight' in checkpoint:
            model_vocab_size = checkpoint['token_embedding.weight'].shape[0]
            print(f"Model vocab_size (from checkpoint): {model_vocab_size:,}")
            print(f"Vocabulary file vocab_size: {vocab_size:,}")
            
            if model_vocab_size != vocab_size:
                print("\nMISMATCH DETECTED!")
                print(f"   Model was trained with vocab_size={model_vocab_size}")
                print(f"   But vocabulary file has vocab_size={vocab_size}")
                print("   This will cause accuracy to be very low!")
                return False
            else:
                print("Vocabulary size matches!")
        else:
            print("Could not find token_embedding.weight in checkpoint")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    return True

def check_training_params(model_dir):
    """Check training parameters from run directory."""
    print("\n" + "="*60)
    print("TRAINING PARAMETERS CHECK")
    print("="*60)
    
    params_path = os.path.join(model_dir, 'training_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        print("Training parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        return params
    else:
        print(f"No training_params.json found in {model_dir}")
        return None

def check_sample_predictions(model_path, vocab_path, device='cpu'):
    """Check if model makes reasonable predictions on sample data."""
    print("\n" + "="*60)
    print("SAMPLE PREDICTION CHECK")
    print("="*60)
    
    # Load vocabulary
    vocab = Vocabulary()
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        vocab.token_to_idx = vocab_data['token_to_idx']
        vocab.idx_to_token = {int(k): v for k, v in vocab_data['idx_to_token'].items()}
    
    # Create config
    config = ModelConfig()
    config.vocab_size = len(vocab.token_to_idx)
    
    # Try to load training params for max_len
    model_dir = os.path.dirname(os.path.abspath(model_path))
    params_path = os.path.join(model_dir, 'training_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        if 'max_length' in params:
            config.max_len = params['max_length']
    
    # Load model
    model = CodeCompletionTransformer(config)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Test prediction
    test_contexts = [
        "def hello",
        "import torch",
        "for i in",
        "if x >",
    ]
    
    print("\nSample predictions:")
    with torch.no_grad():
        for context_str in test_contexts:
            context_tokens = context_str.split()
            context_ids = vocab.encode(context_tokens, max_length=config.max_len, pad=True)
            input_ids = torch.tensor([context_ids], dtype=torch.long).to(device)
            
            logits = model(input_ids)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
            
            print(f"\n  Context: '{context_str}'")
            print("  Top 5 predictions:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
                token = vocab.idx_to_token.get(idx.item(), f"<UNK:{idx.item()}>")
                print(f"    {i}. {token} ({prob.item():.4f})")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Diagnose accuracy issues")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to vocabulary file")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Device to use")
    
    args = parser.parse_args()
    
    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    
    print("\n" + "="*60)
    print("ACCURACY DIAGNOSTIC TOOL")
    print("="*60)
    print(f"\nModel: {args.model_path}")
    print(f"Vocabulary: {args.vocab_path}")
    print(f"Run directory: {model_dir}")
    
    # Check 1: Vocabulary consistency
    vocab_ok = check_vocabulary_consistency(args.vocab_path, args.model_path)
    
    # Check 2: Training parameters
    params = check_training_params(model_dir)
    
    # Check 3: Sample predictions
    if vocab_ok:
        pred_ok = check_sample_predictions(args.model_path, args.vocab_path, args.device)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not vocab_ok:
        print("\nMAIN ISSUE: Vocabulary mismatch!")
        print("   SOLUTION: Make sure you're using the vocabulary file")
        print("   from the same run directory as the model checkpoint.")
        print("   Check the run directory structure:")
        print("     runs/run_*/")
        print("       ├── best_model_*.pt")
        print("       ├── vocab.json  <-- Use this one!")
        print("       └── training_params.json")
    else:
        print("\nVocabulary size matches model")
        print("   If accuracy is still low, check:")
        print("   1. Are you using the correct test dataset?")
        print("   2. Did the data preprocessing change?")
        print("   3. Is the model actually trained? (check loss values)")
        print("   4. Are you evaluating on the same task (token vs line)?")
    
    print()

if __name__ == "__main__":
    main()
