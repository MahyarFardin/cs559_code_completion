import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def build_or_load_vocab(tokenized_dir: str,
                        vocab_min_freq: int,
                        vocab_sample_lines: int,
                        vocab_path: str = "vocab.json") -> Vocabulary:
    """
    Load an existing vocabulary from vocab.json if present, otherwise
    build it from the tokenized files and save it.
    """
    vocab = Vocabulary()

    if os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        # Overwrite the default mappings created in __init__
        vocab.token_to_idx = {tok: int(idx) for tok, idx in vocab_data["token_to_idx"].items()}
        vocab.idx_to_token = {int(k): v for k, v in vocab_data["idx_to_token"].items()}
        print(f"Vocabulary size: {len(vocab.token_to_idx):,}")
        return vocab

    print("Building vocabulary from tokenized data...")
    vocab_files = [
        os.path.join(tokenized_dir, "train.txt"),
        os.path.join(tokenized_dir, "dev.txt"),
        os.path.join(tokenized_dir, "test.txt"),
    ]
    vocab_size = vocab.build_from_files(
        vocab_files,
        min_freq=vocab_min_freq,
        max_lines=vocab_sample_lines,
    )

    # Save to disk so it can be reused by other scripts (Transformer / RNN)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "token_to_idx": vocab.token_to_idx,
                "idx_to_token": {str(k): v for k, v in vocab.idx_to_token.items()},
            },
            f,
            indent=2,
        )
    print(f"Saved vocabulary ({vocab_size:,} tokens) to {vocab_path}")

    return vocab


def train_token_level_rnn(model: nn.Module,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          pad_idx: int,
                          num_epochs: int = 10,
                          device: str = "cuda",
                          lr: float = 1e-3,
                          output_path: str = "best_rnn_token_level.pt"):
    """Train RNN model for token-level completion."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train-token]")

        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)   # [B, T]
            targets = batch["targets"].to(device)       # [B]

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids)                   # [B, T, V]
            last_logits = logits[:, -1, :]              # [B, V]

            loss = criterion(last_logits, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val-token]")

        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                targets = batch["targets"].to(device)

                logits = model(input_ids)
                last_logits = logits[:, -1, :]
                loss = criterion(last_logits, targets)

                val_loss += loss.item()
                val_pbar.set_postfix({"loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_path)
            print(f"Saved best token-level RNN model to {output_path} (val_loss={avg_val_loss:.4f})")

    print(f"Token-level RNN training complete. Best val loss: {best_val_loss:.4f}")


def train_line_level_rnn(model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         pad_idx: int,
                         max_len: int,
                         num_epochs: int = 10,
                         device: str = "cuda",
                         lr: float = 1e-3,
                         output_path: str = "best_rnn_line_level.pt"):
    """Train RNN model for line-level completion (prefix → suffix)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train-line]")

        for batch in train_pbar:
            context_ids = batch["context_ids"].to(device)  # [B, Lc]
            suffix_ids = batch["suffix_ids"].to(device)    # [B, Ls]

            # Start with the encoded previous lines + prefix
            current_input = context_ids
            total_loss = 0.0
            steps = 0

            # Predict each suffix token in sequence (teacher forcing)
            for i in range(suffix_ids.size(1) - 1):
                logits = model(current_input)              # [B, cur_len, V]
                next_logits = logits[:, -1, :]             # [B, V]
                next_target = suffix_ids[:, i]             # [B]

                loss = criterion(next_logits, next_target)
                total_loss += loss
                steps += 1

                # Append the gold next token for teacher forcing
                next_token = suffix_ids[:, i:i+1]          # [B, 1]
                current_input = torch.cat([current_input, next_token], dim=1)

                # Truncate to keep length under control
                if current_input.size(1) > max_len:
                    current_input = current_input[:, -max_len:]

            if steps == 0:
                avg_loss = torch.tensor(0.0, device=device)
            else:
                avg_loss = total_loss / steps

            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += avg_loss.item()
            train_pbar.set_postfix({"loss": avg_loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val-line]")

        with torch.no_grad():
            for batch in val_pbar:
                context_ids = batch["context_ids"].to(device)
                suffix_ids = batch["suffix_ids"].to(device)

                current_input = context_ids
                batch_loss = 0.0
                batch_steps = 0

                for i in range(suffix_ids.size(1) - 1):
                    logits = model(current_input)
                    next_logits = logits[:, -1, :]
                    next_target = suffix_ids[:, i]

                    loss = criterion(next_logits, next_target)
                    batch_loss += loss.item()
                    batch_steps += 1

                    next_token = suffix_ids[:, i:i+1]
                    current_input = torch.cat([current_input, next_token], dim=1)
                    if current_input.size(1) > max_len:
                        current_input = current_input[:, -max_len:]

                if batch_steps > 0:
                    val_loss += batch_loss / batch_steps

                val_pbar.set_postfix({"loss": (batch_loss / batch_steps) if batch_steps > 0 else 0.0})

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_path)
            print(f"Saved best line-level RNN model to {output_path} (val_loss={avg_val_loss:.4f})")

    print(f"Line-level RNN training complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train RNN baseline for code completion")

    parser.add_argument("--dataset_dir", type=str, default="completion_datasets",
                        help="Directory containing completion datasets")
    parser.add_argument("--tokenized_dir", type=str, default="token_completion",
                        help="Directory with tokenized files for vocabulary building")
    parser.add_argument("--task", type=str, choices=["token", "line"], default="token",
                        help="Task type: token-level or line-level")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length (context + suffix while training)")
    parser.add_argument("--max_suffix_length", type=int, default=64,
                        help="Maximum suffix length for line-level dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--vocab_min_freq", type=int, default=10,
                        help="Minimum frequency for vocabulary tokens")
    parser.add_argument("--vocab_sample_lines", type=int, default=50000,
                        help="Number of lines to sample for vocab building (None = all)")
    parser.add_argument("--max_train_examples", type=int, default=None,
                        help="Limit number of training examples (for debugging)")
    parser.add_argument("--max_val_examples", type=int, default=10000,
                        help="Limit number of validation examples")
    parser.add_argument("--lazy_load", action="store_true", default=False,
                        help="Use lazy loading for datasets")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader worker processes")

    # RNN-specific hyperparameters
    parser.add_argument("--rnn_type", type=str, choices=["lstm", "gru"], default="lstm",
                        help="Type of recurrent layer to use")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Hidden size / embedding size (d_model)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of RNN layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate inside RNN and on outputs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for AdamW optimizer")

    parser.add_argument("--output_model_path", type=str, default=None,
                        help="Where to save the best model. "
                             "If not set, a default is chosen based on the task.")

    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Build or load vocabulary
    vocab = build_or_load_vocab(
        tokenized_dir=args.tokenized_dir,
        vocab_min_freq=args.vocab_min_freq,
        vocab_sample_lines=args.vocab_sample_lines,
        vocab_path="vocab.json",
    )
    vocab_size = len(vocab.token_to_idx)
    pad_idx = vocab.token_to_idx["<PAD>"]

    print(f"Vocabulary size: {vocab_size:,}")

    # Create RNN config and model
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

    # Prepare datasets and loaders
    if args.task == "token":
        train_dataset = TokenLevelDataset(
            os.path.join(args.dataset_dir, "token_level", "train.jsonl"),
            vocab,
            args.max_length,
            lazy_load=args.lazy_load,
            max_examples=args.max_train_examples,
        )
        val_dataset = TokenLevelDataset(
            os.path.join(args.dataset_dir, "token_level", "dev.jsonl"),
            vocab,
            args.max_length,
            lazy_load=args.lazy_load,
            max_examples=args.max_val_examples,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_token_level,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_token_level,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False,
        )

        output_path = args.output_model_path or "best_rnn_token_level.pt"
        train_token_level_rnn(
            model,
            train_loader,
            val_loader,
            pad_idx=pad_idx,
            num_epochs=args.num_epochs,
            device=args.device,
            lr=args.lr,
            output_path=output_path,
        )

    else:  # line-level
        train_dataset = LineLevelDataset(
            os.path.join(args.dataset_dir, "line_level", "train.jsonl"),
            vocab,
            args.max_length,
            max_suffix_length=args.max_suffix_length,
            lazy_load=args.lazy_load,
            max_examples=args.max_train_examples,
        )
        val_dataset = LineLevelDataset(
            os.path.join(args.dataset_dir, "line_level", "dev.jsonl"),
            vocab,
            args.max_length,
            max_suffix_length=args.max_suffix_length,
            lazy_load=args.lazy_load,
            max_examples=args.max_val_examples,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_line_level,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_line_level,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False,
        )

        output_path = args.output_model_path or "best_rnn_line_level.pt"
        train_line_level_rnn(
            model,
            train_loader,
            val_loader,
            pad_idx=pad_idx,
            max_len=args.max_length,
            num_epochs=args.num_epochs,
            device=args.device,
            lr=args.lr,
            output_path=output_path,
        )

    print("RNN baseline training complete!")


if __name__ == "__main__":
    main()
