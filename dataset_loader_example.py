#!/usr/bin/env python3
"""
Example script showing how to load and use the code completion datasets.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader

class TokenLevelDataset(Dataset):
    """Dataset for token-level code completion."""
    
    def __init__(self, jsonl_file, tokenizer=None, max_length=256):
        """
        Args:
            jsonl_file: Path to JSONL file with token-level examples
            tokenizer: Tokenizer function (if None, uses simple split)
            max_length: Maximum sequence length
        """
        self.examples = []
        self.tokenizer = tokenizer or (lambda x: x.split())
        self.max_length = max_length
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        context = example['context']
        target = example['target']
        
        # Tokenize context and target
        context_tokens = self.tokenizer(context)
        target_token = self.tokenizer(target)[0] if isinstance(self.tokenizer(target), list) else target
        
        # Truncate context if needed
        if len(context_tokens) > self.max_length - 1:
            context_tokens = context_tokens[-(self.max_length - 1):]
        
        # Create input sequence: context + target
        input_sequence = context_tokens + [target_token]
        
        return {
            'input': input_sequence,
            'target': target_token,
            'context': context
        }

class LineLevelDataset(Dataset):
    """Dataset for line-level code completion."""
    
    def __init__(self, jsonl_file, tokenizer=None, max_context_length=512, max_line_length=128):
        """
        Args:
            jsonl_file: Path to JSONL file with line-level examples
            tokenizer: Tokenizer function (if None, uses simple split)
            max_context_length: Maximum length for previous_lines + prefix
            max_line_length: Maximum length for suffix
        """
        self.examples = []
        self.tokenizer = tokenizer or (lambda x: x.split())
        self.max_context_length = max_context_length
        self.max_line_length = max_line_length
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        previous_lines = example['previous_lines']
        prefix = example['prefix']
        suffix = example['suffix']
        
        # Tokenize
        prev_tokens = self.tokenizer(previous_lines) if previous_lines else []
        prefix_tokens = self.tokenizer(prefix)
        suffix_tokens = self.tokenizer(suffix)
        
        # Combine previous lines and prefix as context
        context = prev_tokens + ['<EOL>'] + prefix_tokens if prev_tokens else prefix_tokens
        
        # Truncate if needed
        if len(context) > self.max_context_length:
            context = context[-self.max_context_length:]
        
        if len(suffix_tokens) > self.max_line_length:
            suffix_tokens = suffix_tokens[:self.max_line_length]
        
        return {
            'context': context,
            'prefix': prefix_tokens,
            'suffix': suffix_tokens,
            'previous_lines': previous_lines,
            'prefix_str': prefix,
            'suffix_str': suffix
        }

def collate_token_level(batch, pad_token='<PAD>'):
    """Collate function for token-level dataset."""
    # This is a simple example - you'd need a proper tokenizer/vocab in practice
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    
    return {
        'inputs': inputs,
        'targets': targets
    }

def collate_line_level(batch, pad_token='<PAD>'):
    """Collate function for line-level dataset."""
    contexts = [item['context'] for item in batch]
    prefixes = [item['prefix'] for item in batch]
    suffixes = [item['suffix'] for item in batch]
    
    return {
        'contexts': contexts,
        'prefixes': prefixes,
        'suffixes': suffixes
    }

# Example usage
if __name__ == "__main__":
    # Example: Load token-level dataset
    print("Loading token-level dataset...")
    token_dataset = TokenLevelDataset("completion_datasets/token_level/train.jsonl")
    print(f"Loaded {len(token_dataset)} examples")
    
    # Show first example
    example = token_dataset[0]
    print(f"\nToken-level example:")
    print(f"  Context: {example['context'][:100]}...")
    print(f"  Target: {example['target']}")
    
    # Example: Load line-level dataset
    print("\nLoading line-level dataset...")
    line_dataset = LineLevelDataset("completion_datasets/line_level/train.jsonl")
    print(f"Loaded {len(line_dataset)} examples")
    
    # Show first example
    example = line_dataset[0]
    print(f"\nLine-level example:")
    print(f"  Previous lines: {example['previous_lines'][:100]}...")
    print(f"  Prefix: {example['prefix_str']}")
    print(f"  Suffix: {example['suffix_str']}")
    
    # Create DataLoaders (you'd need proper tokenization/vocab for real training)
    # token_loader = DataLoader(token_dataset, batch_size=32, shuffle=True, collate_fn=collate_token_level)
    # line_loader = DataLoader(line_dataset, batch_size=32, shuffle=True, collate_fn=collate_line_level)

