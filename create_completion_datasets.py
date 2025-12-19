#!/usr/bin/env python3
"""
Create code completion datasets from tokenized files.
Supports token-level and line-level completion tasks.
"""

import os
import argparse
import json
from typing import List, Tuple
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

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
    # If neither flag is provided, default to creating BOTH datasets (backwards compatible).
    parser.add_argument("--token_level", action="store_true", default=False,
                        help="Create token-level dataset")
    parser.add_argument("--line_level", action="store_true", default=False,
                        help="Create line-level dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of sequences to process (for testing)")
    parser.add_argument("--chunk_size", type=int, default=5000,
                        help="Process sequences in chunks to save memory (default: 5000, increase for faster processing on Colab)")
    parser.add_argument("--write_batch_size", type=int, default=10000,
                        help="Batch size for writing JSONL lines (default: 10000, larger = faster but more memory)")
    parser.add_argument("--disable_progress", action="store_true",
                        help="Disable progress bars (useful if redirecting output)")
    
    args = parser.parse_args()

    # Backwards-compatible default: if user didn't choose, create both.
    if not args.token_level and not args.line_level:
        args.token_level = True
        args.line_level = True
    
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
        
        # Process file line-by-line to avoid loading everything into memory
        chunk_size = args.chunk_size
        sequence_count = 0
        chunk = []
        
        # Token-level dataset
        if args.token_level:
            print(f"  Creating token-level dataset...")
            output_file = os.path.join(args.output_dir, "token_level", f"{split}.jsonl")
            total_examples = 0
            write_batch = []
            write_batch_size = args.write_batch_size
            
            # Count total lines for progress bar (approximate)
            if not args.disable_progress and HAS_TQDM:
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        total_lines = sum(1 for _ in f)
                except:
                    total_lines = None
            else:
                total_lines = None
            
            with open(output_file, 'w', encoding='utf-8') as f, \
                 open(input_file, 'r', encoding='utf-8') as infile:
                iterator = tqdm(infile, total=total_lines, desc=f"    Token-level {split}", 
                               disable=args.disable_progress or not HAS_TQDM, unit="lines")
                for line in iterator:
                    if args.limit and sequence_count >= args.limit:
                        break
                    line = line.strip()
                    if line:
                        tokens = line.split()
                        # Remove <s> and </s> markers
                        if tokens and tokens[0] == "<s>":
                            tokens = tokens[1:]
                        if tokens and tokens[-1] == "</s>":
                            tokens = tokens[:-1]
                        if tokens:
                            chunk.append(tokens)
                            sequence_count += 1
                            
                            # Process chunk when it reaches chunk_size
                            if len(chunk) >= chunk_size:
                                chunk_examples = create_token_level_dataset(chunk, args.max_length)
                                for context, target in chunk_examples:
                                    example = {"context": context, "target": target}
                                    json_line = json.dumps(example, ensure_ascii=False) + "\n"
                                    write_batch.append(json_line)
                                    
                                    # Write batch when it reaches write_batch_size
                                    if len(write_batch) >= write_batch_size:
                                        f.writelines(write_batch)
                                        write_batch = []
                                
                                total_examples += len(chunk_examples)
                                chunk = []  # Clear chunk
                                
                                if not args.disable_progress and HAS_TQDM:
                                    iterator.set_postfix({"examples": f"{total_examples:,}"})
                
                # Process remaining chunk
                if chunk:
                    chunk_examples = create_token_level_dataset(chunk, args.max_length)
                    for context, target in chunk_examples:
                        example = {"context": context, "target": target}
                        json_line = json.dumps(example, ensure_ascii=False) + "\n"
                        write_batch.append(json_line)
                    total_examples += len(chunk_examples)
                
                # Write remaining batch
                if write_batch:
                    f.writelines(write_batch)
            print(f"  Created {total_examples:,} token-level examples from {sequence_count} sequences -> {output_file}")
        
        # Line-level dataset - process file again (or could combine, but simpler to separate)
        if args.line_level:
            print(f"  Creating line-level dataset...")
            output_file = os.path.join(args.output_dir, "line_level", f"{split}.jsonl")
            total_examples = 0
            sequence_count = 0
            chunk = []
            write_batch = []
            write_batch_size = args.write_batch_size
            
            # Count total lines for progress bar (approximate)
            if not args.disable_progress and HAS_TQDM:
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        total_lines = sum(1 for _ in f)
                except:
                    total_lines = None
            else:
                total_lines = None
            
            with open(output_file, 'w', encoding='utf-8') as f, \
                 open(input_file, 'r', encoding='utf-8') as infile:
                iterator = tqdm(infile, total=total_lines, desc=f"    Line-level {split}",
                               disable=args.disable_progress or not HAS_TQDM, unit="lines")
                for line in iterator:
                    if args.limit and sequence_count >= args.limit:
                        break
                    line = line.strip()
                    if line:
                        tokens = line.split()
                        # Remove <s> and </s> markers
                        if tokens and tokens[0] == "<s>":
                            tokens = tokens[1:]
                        if tokens and tokens[-1] == "</s>":
                            tokens = tokens[:-1]
                        if tokens:
                            chunk.append(tokens)
                            sequence_count += 1
                            
                            # Process chunk when it reaches chunk_size
                            if len(chunk) >= chunk_size:
                                chunk_examples = create_line_level_dataset(
                                    chunk, 
                                    args.min_prefix_length, 
                                    args.max_prefix_ratio,
                                    args.examples_per_line
                                )
                                for previous_lines, prefix, suffix in chunk_examples:
                                    example = {
                                        "previous_lines": previous_lines,
                                        "prefix": prefix,
                                        "suffix": suffix
                                    }
                                    json_line = json.dumps(example, ensure_ascii=False) + "\n"
                                    write_batch.append(json_line)
                                    
                                    # Write batch when it reaches write_batch_size
                                    if len(write_batch) >= write_batch_size:
                                        f.writelines(write_batch)
                                        write_batch = []
                                
                                total_examples += len(chunk_examples)
                                chunk = []  # Clear chunk
                                
                                if not args.disable_progress and HAS_TQDM:
                                    iterator.set_postfix({"examples": f"{total_examples:,}"})
                
                # Process remaining chunk
                if chunk:
                    chunk_examples = create_line_level_dataset(
                        chunk, 
                        args.min_prefix_length, 
                        args.max_prefix_ratio,
                        args.examples_per_line
                    )
                    for previous_lines, prefix, suffix in chunk_examples:
                        example = {
                            "previous_lines": previous_lines,
                            "prefix": prefix,
                            "suffix": suffix
                        }
                        json_line = json.dumps(example, ensure_ascii=False) + "\n"
                        write_batch.append(json_line)
                    total_examples += len(chunk_examples)
                
                # Write remaining batch
                if write_batch:
                    f.writelines(write_batch)
            print(f"  Created {total_examples:,} line-level examples from {sequence_count} sequences -> {output_file}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

