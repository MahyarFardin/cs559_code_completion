import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train import (
    Vocabulary,
    TokenLevelDataset,
    LineLevelDataset,
    collate_token_level,
    collate_line_level,
)
from rnn_model import RNNLanguageModel, RNNConfig


def load_vocab(vocab_path: str) -> Vocabulary:
    """Load vocabulary from a saved vocab.json file."""
    vocab = Vocabulary()
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    vocab.token_to_idx = {tok: int(idx) for tok, idx in vocab_data["token_to_idx"].items()}
    vocab.idx_to_token = {int(k): v for k, v in vocab_data["idx_to_token"].items()}

    print(f"Vocabulary size: {len(vocab.token_to_idx):,}")
    return vocab


def evaluate_token_level_rnn(model: nn.Module,
                             test_loader: DataLoader,
                             pad_idx: int,
                             device: str = "cuda"):
    """Evaluate token-level next-token prediction accuracy and loss."""
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating RNN (token-level)"):
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            logits = model(input_ids)            # [B, T, V]
            last_logits = logits[:, -1, :]       # [B, V]

            loss = criterion(last_logits, targets)
            total_loss += loss.item()

            preds = last_logits.argmax(dim=-1)   # [B]
            total_correct += (preds == targets).sum().item()
            total_examples += targets.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_examples if total_examples > 0 else 0.0

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "total_examples": int(total_examples),
    }


def evaluate_line_level_rnn(model: nn.Module,
                            test_loader: DataLoader,
                            config: RNNConfig,
                            pad_idx: int,
                            device: str = "cuda"):
    """
    Evaluate line-level completion by computing average per-token loss
    over the predicted suffix tokens.
    """
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating RNN (line-level)"):
            context_ids = batch["context_ids"].to(device)  # [B, Lc]
            suffix_ids = batch["suffix_ids"].to(device)    # [B, Ls]

            current_input = context_ids
            batch_loss = 0.0
            batch_steps = 0

            for i in range(suffix_ids.size(1) - 1):
                logits = model(current_input)              # [B, cur_len, V]
                next_logits = logits[:, -1, :]             # [B, V]
                next_target = suffix_ids[:, i]             # [B]

                loss = criterion(next_logits, next_target)
                batch_loss += loss.item()
                batch_steps += 1

                next_token = suffix_ids[:, i:i+1]          # [B, 1]
                current_input = torch.cat([current_input, next_token], dim=1)
                if current_input.size(1) > config.max_len:
                    current_input = current_input[:, -config.max_len:]

            if batch_steps > 0:
                total_loss += batch_loss / batch_steps
                total_examples += 1

    avg_loss = total_loss / len(test_loader)

    return {
        "loss": float(avg_loss),
        "total_examples": int(total_examples),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RNN baseline on test set")

    parser.add_argument("--dataset_dir", type=str, default="completion_datasets",
                        help="Directory containing completion datasets")
    parser.add_argument("--task", type=str, choices=["token", "line"], default="token",
                        help="Task type: token-level or line-level")
    parser.add_argument("--vocab_path", type=str, default="vocab.json",
                        help="Path to vocabulary JSON file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained RNN model weights (.pt)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum context length (must match training)")
    parser.add_argument("--max_suffix_length", type=int, default=64,
                        help="Maximum suffix length for line-level dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lazy_load", action="store_true", default=False,
                        help="Use lazy loading for datasets")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader worker processes")

    # RNN architecture parameters must match training
    parser.add_argument("--rnn_type", type=str, choices=["lstm", "gru"], default="lstm",
                        help="RNN type used during training")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Hidden size / embedding size (must match training)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of RNN layers (must match training)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate used during training")

    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load vocabulary
    vocab = load_vocab(args.vocab_path)
    vocab_size = len(vocab.token_to_idx)
    pad_idx = vocab.token_to_idx["<PAD>"]

    # Recreate RNN config and model architecture
    config = RNNConfig(
        vocab_size=vocab_size,
        d_model=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        max_len=args.max_length,
    )
    model = RNNLanguageModel(config)
    print(f"RNN model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load trained weights
    print(f"Loading model weights from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)

    # Prepare dataset and dataloader
    if args.task == "token":
        test_dataset = TokenLevelDataset(
            os.path.join(args.dataset_dir, "token_level", "test.jsonl"),
            vocab,
            args.max_length,
            lazy_load=args.lazy_load,
            max_examples=None,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_token_level,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False,
        )

        metrics = evaluate_token_level_rnn(
            model,
            test_loader,
            pad_idx=pad_idx,
            device=args.device,
        )
        print("\nToken-level RNN evaluation:")
        print(json.dumps(metrics, indent=2))

    else:  # line-level
        test_dataset = LineLevelDataset(
            os.path.join(args.dataset_dir, "line_level", "test.jsonl"),
            vocab,
            args.max_length,
            max_suffix_length=args.max_suffix_length,
            lazy_load=args.lazy_load,
            max_examples=None,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_line_level,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False,
        )

        metrics = evaluate_line_level_rnn(
            model,
            test_loader,
            config=config,
            pad_idx=pad_idx,
            device=args.device,
        )
        print("\nLine-level RNN evaluation:")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
