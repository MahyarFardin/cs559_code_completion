#!/usr/bin/env python3
"""
Create code completion datasets from tokenized files.
Supports token-level and line-level completion tasks.
"""

import os
import argparse
import json
from typing import List, Tuple

def load_tokenized_file(file_path: str, limit: int = None) -> List[List[str]]:
    """Load tokenized file and return list of token sequences."""
    sequences = []
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
            line = line.strip()
            if line:
                tokens = line.split()
                # Remove <s> and </s> markers for processing
                if tokens and tokens[0] == "<s>":
                    tokens = tokens[1:]
                if tokens and tokens[-1] == "</s>":
                    tokens = tokens[:-1]
                if tokens:  # Only add non-empty sequences
                    sequences.append(tokens)
                    count += 1
    return sequences

def create_token_level_dataset(sequences: List[List[str]], max_length: int = 256) -> List[Tuple[str, str]]:
    """
    Create token-level dataset for next-token prediction.
    Each example: (context, target_token)
    """
    examples = []
    for tokens in sequences:
        # Create sliding window examples
        for i in range(1, len(tokens)):
            # Context: tokens up to (but not including) target
            context = tokens[:i]
            target = tokens[i]
            
            # Truncate context if too long
            if len(context) > max_length - 1:
                context = context[-(max_length - 1):]
            
            context_str = " ".join(context)
            examples.append((context_str, target))
    
    return examples

def split_into_lines(tokens: List[str]) -> List[List[str]]:
    """Split token sequence into lines based on <EOL> markers."""
    lines = []
    current_line = []
    
    for token in tokens:
        if token == "<EOL>":
            if current_line:  # Only add non-empty lines
                lines.append(current_line)
                current_line = []
        else:
            current_line.append(token)
    
    # Add last line if exists
    if current_line:
        lines.append(current_line)
    
    return lines

def create_line_level_dataset(sequences: List[List[str]], 
                              min_prefix_length: int = 3,
                              max_prefix_ratio: float = 0.8,
                              examples_per_line: int = 1) -> List[Tuple[str, str, str]]:
    """
    Create line-level dataset for line completion.
    Each example: (previous_lines, prefix, suffix)
    - previous_lines: all lines before the current line
    - prefix: beginning of current line (to predict from)
    - suffix: rest of current line (to predict)
    """
    import random
    examples = []
    
    for tokens in sequences:
        lines = split_into_lines(tokens)
        
        for line_idx, line in enumerate(lines):
            if len(line) < min_prefix_length + 1:
                continue  # Skip lines that are too short
            
            # Previous lines context
            previous_lines = lines[:line_idx]
            previous_context = " <EOL> ".join([" ".join(l) for l in previous_lines])
            
            # Split current line into prefix and suffix
            # Prefix length: between min_prefix_length and max_prefix_ratio of line length
            max_prefix_len = max(min_prefix_length, int(len(line) * max_prefix_ratio))
            min_prefix_len = min(min_prefix_length, len(line) - 1)
            
            # Create 1 example per line (or more if specified)
            # Use a random split point for diversity
            valid_split_points = list(range(min_prefix_len, min(max_prefix_len + 1, len(line))))
            if not valid_split_points:
                continue
            
            # Sample split points
            if examples_per_line == 1:
                split_points = [random.choice(valid_split_points)]
            else:
                split_points = random.sample(valid_split_points, min(examples_per_line, len(valid_split_points)))
            
            for split_point in split_points:
                prefix = line[:split_point]
                suffix = line[split_point:]
                
                if len(suffix) > 0:  # Only add if there's something to predict
                    prefix_str = " ".join(prefix)
                    suffix_str = " ".join(suffix)
                    examples.append((previous_context, prefix_str, suffix_str))
    
    return examples

def save_token_level_dataset(examples: List[Tuple[str, str]], output_file: str):
    """Save token-level dataset in JSONL format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for context, target in examples:
            example = {
                "context": context,
                "target": target
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

def save_line_level_dataset(examples: List[Tuple[str, str, str]], output_file: str):
    """Save line-level dataset in JSONL format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for previous_lines, prefix, suffix in examples:
            example = {
                "previous_lines": previous_lines,
                "prefix": prefix,
                "suffix": suffix
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Create code completion datasets from tokenized files")
    parser.add_argument("--input_dir", type=str, default="token_completion",
                        help="Directory containing tokenized files (train.txt, dev.txt, test.txt)")
    parser.add_argument("--output_dir", type=str, default="completion_datasets",
                        help="Output directory for completion datasets")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length for token-level dataset")
    parser.add_argument("--min_prefix_length", type=int, default=3,
                        help="Minimum prefix length for line-level dataset")
    parser.add_argument("--max_prefix_ratio", type=float, default=0.8,
                        help="Maximum prefix ratio (0-1) for line-level dataset")
    parser.add_argument("--examples_per_line", type=int, default=1,
                        help="Number of examples to create per line (default: 1)")
    parser.add_argument("--token_level", action="store_true", default=True,
                        help="Create token-level dataset")
    parser.add_argument("--line_level", action="store_true", default=True,
                        help="Create line-level dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of sequences to process (for testing)")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Process sequences in chunks to save memory")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.token_level:
        os.makedirs(os.path.join(args.output_dir, "token_level"), exist_ok=True)
    if args.line_level:
        os.makedirs(os.path.join(args.output_dir, "line_level"), exist_ok=True)
    
    # Process each split
    for split in ["train", "dev", "test"]:
        input_file = os.path.join(args.input_dir, f"{split}.txt")
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        print(f"\nProcessing {split}...")
        sequences = load_tokenized_file(input_file, limit=args.limit)
        
        if args.limit:
            print(f"  Limited to {len(sequences)} sequences")
        
        print(f"  Loaded {len(sequences)} sequences")
        
        # Token-level dataset
        if args.token_level:
            print(f"  Creating token-level dataset...")
            token_examples = []
            # Process in chunks to save memory
            chunk_size = args.chunk_size
            for i in range(0, len(sequences), chunk_size):
                chunk = sequences[i:i+chunk_size]
                chunk_examples = create_token_level_dataset(chunk, args.max_length)
                token_examples.extend(chunk_examples)
                if (i // chunk_size + 1) % 10 == 0:
                    print(f"    Processed {min(i+chunk_size, len(sequences))}/{len(sequences)} sequences...")
            
            output_file = os.path.join(args.output_dir, "token_level", f"{split}.jsonl")
            save_token_level_dataset(token_examples, output_file)
            print(f"  Created {len(token_examples)} token-level examples -> {output_file}")
        
        # Line-level dataset
        if args.line_level:
            print(f"  Creating line-level dataset...")
            line_examples = []
            # Process in chunks to save memory
            chunk_size = args.chunk_size
            for i in range(0, len(sequences), chunk_size):
                chunk = sequences[i:i+chunk_size]
                chunk_examples = create_line_level_dataset(
                    chunk, 
                    args.min_prefix_length, 
                    args.max_prefix_ratio,
                    args.examples_per_line
                )
                line_examples.extend(chunk_examples)
                if (i // chunk_size + 1) % 10 == 0:
                    print(f"    Processed {min(i+chunk_size, len(sequences))}/{len(sequences)} sequences...")
            
            output_file = os.path.join(args.output_dir, "line_level", f"{split}.jsonl")
            save_line_level_dataset(line_examples, output_file)
            print(f"  Created {len(line_examples)} line-level examples -> {output_file}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

