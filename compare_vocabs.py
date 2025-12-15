#!/usr/bin/env python3
"""
Compare vocabulary files to understand the loss difference.
"""

import json
import os
import sys

def analyze_vocab(vocab_path):
    """Analyze a vocabulary file."""
    if not os.path.exists(vocab_path):
        return None
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    vocab_size = len(vocab_data['token_to_idx'])
    
    # Check special tokens
    special_tokens = ['<PAD>', '<UNK>', '<s>', '</s>', '<EOL>']
    special_info = {}
    for token in special_tokens:
        if token in vocab_data['token_to_idx']:
            special_info[token] = vocab_data['token_to_idx'][token]
    
    return {
        'vocab_size': vocab_size,
        'special_tokens': special_info,
        'path': vocab_path
    }

def main():
    print("="*60)
    print("VOCABULARY COMPARISON")
    print("="*60)
    
    # Check root vocab.json
    root_vocab = analyze_vocab('vocab.json')
    if root_vocab:
        print(f"\nRoot vocab.json:")
        print(f"  Path: {root_vocab['path']}")
        print(f"  Vocabulary size: {root_vocab['vocab_size']:,}")
        print(f"  Special tokens: {root_vocab['special_tokens']}")
    
    # Check run directories
    if os.path.exists('runs'):
        print(f"\nRun directories:")
        for run_dir in sorted(os.listdir('runs')):
            run_path = os.path.join('runs', run_dir)
            if os.path.isdir(run_path):
                vocab_path = os.path.join(run_path, 'vocab.json')
                vocab_info = analyze_vocab(vocab_path)
                if vocab_info:
                    print(f"\n  {run_dir}:")
                    print(f"    Vocabulary size: {vocab_info['vocab_size']:,}")
                    
                    # Check training params
                    params_path = os.path.join(run_path, 'training_params.json')
                    if os.path.exists(params_path):
                        with open(params_path, 'r') as f:
                            params = json.load(f)
                        print(f"    vocab_min_freq: {params.get('vocab_min_freq', 'N/A')}")
                        print(f"    max_train_examples: {params.get('max_train_examples', 'N/A')}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if root_vocab:
        print(f"\nRoot vocab.json has {root_vocab['vocab_size']:,} tokens")
        print("This was likely from your first run (before run directories).")
        print("\nIf your current run uses vocab_min_freq=50, it will have a")
        print("SMALLER vocabulary, which can cause:")
        print("  1. More tokens mapped to <UNK>")
        print("  2. Higher loss (harder to predict)")
        print("  3. Lower accuracy")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("\nTo match your first run's performance:")
    print("  1. Use vocab_min_freq=10 (or lower) instead of 50")
    print("  2. Or use the root vocab.json from your first run")
    print("\nTo find your first model:")
    print("  - Check if best_model_token_level.pt exists in root directory")
    print("  - Or check your training logs for where it was saved")

if __name__ == "__main__":
    main()
