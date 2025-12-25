#!/usr/bin/env python3
"""
Train an RNN baseline (RNN/LSTM/GRU) for code completion.

Supports:
- token-level: predict next token (target) given context
- line-level: predict suffix tokens step-by-step using teacher forcing

This script reports BOTH loss and accuracy for BOTH tasks.
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

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

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASELINE_DIR, "logs")
FIG_DIR = os.path.join(BASELINE_DIR, "figures")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def plot_curves(history, title, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def build_or_load_vocab(tokenized_dir: str,
                        vocab_min_freq: int,
                        vocab_sample_lines: int,
                        vocab_path: str = "vocab.json",
                        rebuild: bool = False) -> Vocabulary:
    """
    Load vocab from vocab_path if it exists (unless rebuild=True),
    otherwise build from tokenized files and save it.
    """
    vocab = Vocabulary()

    if os.path.exists(vocab_path) and not rebuild:
        print(f"Loading vocabulary from {vocab_path}...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
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


def _gather_logits_at_positions(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, T, V]
    positions: [B] integer indices in [0, T-1]
    returns: [B, V] logits for each batch element at its specified position
    """
    bsz = logits.size(0)
    batch_idx = torch.arange(bsz, device=logits.device)
    return logits[batch_idx, positions, :]


def train_token_level_rnn(model: nn.Module,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          pad_idx: int,
                          num_epochs: int = 10,
                          device: str = "cuda",
                          lr: float = 1e-3,
                          output_path: str = "best_rnn_token_level.pt",
                          early_stop_patience: int = 10,
                          early_stop_delta: float = 1e-4,
                          ):
    """
    Token-level training:
    - Dataset provides input_ids = (context + target) padded on the right.
    - We must predict 'target' from the last *context* position, not from the last padded position.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        # -----------------
        # Train
        # -----------------
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train-token]")
        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)   # [B, T] right-padded
            targets = batch["targets"].to(device)       # [B]

            # Determine the position of the last context token:
            # input_ids = context + target + PAD...
            # nonpad_count = len(context) + 1
            # predict target using logits at position (len(context)-1) = nonpad_count - 2
            nonpad_count = (input_ids != pad_idx).sum(dim=1)                 # [B]
            pos = (nonpad_count - 2).clamp(min=0)                            # [B]

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids)                                        # [B, T, V]
            pred_logits = _gather_logits_at_positions(logits, pos)           # [B, V]

            loss = criterion(pred_logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += float(loss.item()) * targets.size(0)
            preds = pred_logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            train_pbar.set_postfix({"loss": float(loss.item()), "acc": (correct / max(1, total))})

        train_loss = loss_sum / max(1, total)
        train_acc = correct / max(1, total)

        # -----------------
        # Val
        # -----------------
        model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val-token]")
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                targets = batch["targets"].to(device)

                nonpad_count = (input_ids != pad_idx).sum(dim=1)
                pos = (nonpad_count - 2).clamp(min=0)

                logits = model(input_ids)
                pred_logits = _gather_logits_at_positions(logits, pos)

                loss = criterion(pred_logits, targets)

                loss_sum += float(loss.item()) * targets.size(0)
                preds = pred_logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                val_pbar.set_postfix({"loss": float(loss.item()), "acc": (correct / max(1, total))})

        val_loss = loss_sum / max(1, total)
        val_acc = correct / max(1, total)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)


        if val_loss < best_val_loss - early_stop_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_path)
            print(f"Saved best token-level RNN to {output_path} (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best val loss: {best_val_loss:.4f}"
            )
            break

    print(f"Token-level training complete. Best val loss: {best_val_loss:.4f}")

    # ---- Save logs ----
    run_name = f"{model.config.rnn_type}_token_h{model.config.d_model}_l{model.config.num_layers}"

    log_path = os.path.join(LOG_DIR, f"{run_name}_metrics.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training log to {log_path}")

    # ---- Plot curves ----
    fig_path = os.path.join(FIG_DIR, f"{run_name}_curves.png")
    plot_curves(
        history,
        title=f"{model.config.rnn_type.upper()} Token-Level Training",
        save_path=fig_path
    )

    print(f"Saved learning curves to {fig_path}")


def train_line_level_rnn(model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         pad_idx: int,
                         max_len: int,
                         num_epochs: int = 10,
                         device: str = "cuda",
                         lr: float = 1e-3,
                         output_path: str = "best_rnn_line_level.pt",
                         early_stop_patience: int = 10,
                         early_stop_delta: float = 1e-4,
                         ):
    """
    Line-level training (teacher forcing):
    - Start from context_ids (right-padded).
    - Predict suffix tokens one-by-one, appending ground-truth suffix tokens into the input.
    - We compute loss/accuracy only on non-PAD suffix targets.
    - We avoid the "PAD-at-the-end" bug by always predicting from the last *real* token position.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        # -----------------
        # Train
        # -----------------
        model.train()
        loss_token_sum = 0.0
        correct_tokens = 0
        total_tokens = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train-line]")
        for batch in train_pbar:
            context_ids = batch["context_ids"].to(device)     # [B, max_len] right-padded
            suffix_ids = batch["suffix_ids"].to(device)       # [B, max_suffix] right-padded

            steps = suffix_ids.size(1) - 1
            if steps <= 0:
                continue

            # Total valid tokens across all steps (for stable normalization)
            target_block = suffix_ids[:, :steps]
            total_valid = (target_block != pad_idx).sum().item()
            if total_valid == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            # We will keep an editable input buffer of size [B, max_len]
            current_input = context_ids.clone()
            # Current (non-pad) lengths in the buffer
            cur_len = (current_input != pad_idx).sum(dim=1)  # [B], counts non-pad in initial context

            for i in range(steps):
                next_target = suffix_ids[:, i]               # [B]
                mask = (next_target != pad_idx)
                if not mask.any():
                    break

                logits = model(current_input)                # [B, max_len, V]
                # Predict from the last real token position (cur_len-1)
                pos = (cur_len - 1).clamp(min=0)
                next_logits = _gather_logits_at_positions(logits, pos)  # [B, V]

                # Compute CE only on valid targets
                loss_step = criterion(next_logits[mask], next_target[mask])  # mean over valid
                valid_here = mask.sum().item()

                # Scale so the batch gradient corresponds to mean CE over all valid tokens
                loss_scaled = loss_step * (valid_here / total_valid)
                loss_scaled.backward()

                # Metrics (token accuracy)
                preds = next_logits.argmax(dim=-1)
                correct_tokens += (preds[mask] == next_target[mask]).sum().item()
                total_tokens += valid_here
                loss_token_sum += float(loss_step.item()) * valid_here

                # Teacher forcing append: insert next_target into current_input at position cur_len
                # If cur_len < max_len, write into that slot; otherwise shift left and write at end.
                idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

                # those that still have space
                has_space = cur_len[idx] < max_len
                idx_space = idx[has_space]
                if idx_space.numel() > 0:
                    pos_write = cur_len[idx_space]
                    current_input[idx_space, pos_write] = next_target[idx_space]
                    cur_len[idx_space] = cur_len[idx_space] + 1

                # those that are full -> shift left by 1 and write at last
                idx_full = idx[~has_space]
                if idx_full.numel() > 0:
                    current_input[idx_full, :-1] = current_input[idx_full, 1:]
                    current_input[idx_full, -1] = next_target[idx_full]
                    cur_len[idx_full] = max_len

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            avg_loss_so_far = loss_token_sum / max(1, total_tokens)
            acc_so_far = correct_tokens / max(1, total_tokens)
            train_pbar.set_postfix({"loss": avg_loss_so_far, "acc": acc_so_far})

        train_loss = loss_token_sum / max(1, total_tokens)
        train_acc = correct_tokens / max(1, total_tokens)

        # -----------------
        # Val
        # -----------------
        model.eval()
        loss_token_sum = 0.0
        correct_tokens = 0
        total_tokens = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val-line]")
        with torch.no_grad():
            for batch in val_pbar:
                context_ids = batch["context_ids"].to(device)
                suffix_ids = batch["suffix_ids"].to(device)

                steps = suffix_ids.size(1) - 1
                if steps <= 0:
                    continue

                target_block = suffix_ids[:, :steps]
                total_valid = (target_block != pad_idx).sum().item()
                if total_valid == 0:
                    continue

                current_input = context_ids.clone()
                cur_len = (current_input != pad_idx).sum(dim=1)

                for i in range(steps):
                    next_target = suffix_ids[:, i]
                    mask = (next_target != pad_idx)
                    if not mask.any():
                        break

                    logits = model(current_input)
                    pos = (cur_len - 1).clamp(min=0)
                    next_logits = _gather_logits_at_positions(logits, pos)

                    loss_step = criterion(next_logits[mask], next_target[mask])
                    valid_here = mask.sum().item()

                    preds = next_logits.argmax(dim=-1)
                    correct_tokens += (preds[mask] == next_target[mask]).sum().item()
                    total_tokens += valid_here
                    loss_token_sum += float(loss_step.item()) * valid_here

                    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    has_space = cur_len[idx] < max_len
                    idx_space = idx[has_space]
                    if idx_space.numel() > 0:
                        pos_write = cur_len[idx_space]
                        current_input[idx_space, pos_write] = next_target[idx_space]
                        cur_len[idx_space] = cur_len[idx_space] + 1

                    idx_full = idx[~has_space]
                    if idx_full.numel() > 0:
                        current_input[idx_full, :-1] = current_input[idx_full, 1:]
                        current_input[idx_full, -1] = next_target[idx_full]
                        cur_len[idx_full] = max_len

                val_pbar.set_postfix({
                    "loss": (loss_token_sum / max(1, total_tokens)),
                    "acc": (correct_tokens / max(1, total_tokens))
                })

        val_loss = loss_token_sum / max(1, total_tokens)
        val_acc = correct_tokens / max(1, total_tokens)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss - early_stop_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_path)
            print(f"Saved best line-level RNN to {output_path} (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best val loss: {best_val_loss:.4f}"
            )
            break

    print(f"Line-level training complete. Best val loss: {best_val_loss:.4f}")


    run_name = f"{model.config.rnn_type}_line_h{model.config.d_model}_l{model.config.num_layers}"

    log_path = os.path.join(LOG_DIR, f"{run_name}_metrics.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training log to {log_path}")

    fig_path = os.path.join(FIG_DIR, f"{run_name}_curves.png")
    plot_curves(
        history,
        title=f"{model.config.rnn_type.upper()} Line-Level Training",
        save_path=fig_path
    )

    print(f"Saved learning curves to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RNN baseline for code completion")

    parser.add_argument("--dataset_dir", type=str, default="completion_datasets")
    parser.add_argument("--tokenized_dir", type=str, default="token_completion")
    parser.add_argument("--task", type=str, choices=["token", "line"], default="token")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256,
                        help="Context buffer length for RNN forward (must match dataset context length).")
    parser.add_argument("--max_suffix_length", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--vocab_path", type=str, default="vocab.json",
                        help="Path to vocab.json (shared across runs unless changed).")
    parser.add_argument("--rebuild_vocab", action="store_true", default=False,
                        help="Force rebuild of vocabulary even if vocab_path exists.")
    parser.add_argument("--vocab_min_freq", type=int, default=10)
    parser.add_argument("--vocab_sample_lines", type=int, default=50000)

    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=10000)
    parser.add_argument("--lazy_load", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4)


    # RNN-specific
    parser.add_argument("--rnn_type", type=str, choices=["rnn", "lstm", "gru"], default="lstm")
    parser.add_argument("--nonlinearity", type=str, choices=["tanh", "relu"], default="tanh",
                        help="Only used when --rnn_type rnn.")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--output_model_path", type=str, default=None)

    args = parser.parse_args()
    print(f"Using device: {args.device}")

    vocab = build_or_load_vocab(
        tokenized_dir=args.tokenized_dir,
        vocab_min_freq=args.vocab_min_freq,
        vocab_sample_lines=args.vocab_sample_lines,
        vocab_path=args.vocab_path,
        rebuild=args.rebuild_vocab,
    )
    pad_idx = vocab.token_to_idx["<PAD>"]
    vocab_size = len(vocab.token_to_idx)
    print(f"Vocabulary size: {vocab_size:,} (pad_idx={pad_idx})")

    config = RNNConfig(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        d_model=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        nonlinearity=args.nonlinearity,
        dropout=args.dropout,
        max_len=args.max_length,
    )
    model = RNNLanguageModel(config)
    print(f"RNN model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.task == "token":
        train_dataset = TokenLevelDataset(
            os.path.join(args.dataset_dir, "token_level", "train.jsonl"),
            vocab, args.max_length, lazy_load=args.lazy_load, max_examples=args.max_train_examples
        )
        val_dataset = TokenLevelDataset(
            os.path.join(args.dataset_dir, "token_level", "dev.jsonl"),
            vocab, args.max_length, lazy_load=args.lazy_load, max_examples=args.max_val_examples
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_token_level, num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_token_level, num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )

        out_path = args.output_model_path or "best_rnn_token_level.pt"
        train_token_level_rnn(
            model, train_loader, val_loader,
            pad_idx=pad_idx,
            num_epochs=args.num_epochs,
            device=args.device,
            lr=args.lr,
            output_path=out_path,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
        )


    else:
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

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_line_level, num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_line_level, num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )

        out_path = args.output_model_path or "best_rnn_line_level.pt"
        train_line_level_rnn(
            model, train_loader, val_loader,
            pad_idx=pad_idx,
            max_len=args.max_length,
            num_epochs=args.num_epochs,
            device=args.device,
            lr=args.lr,
            output_path=out_path,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
        )

    print("RNN baseline training complete!")


if __name__ == "__main__":
    main()
