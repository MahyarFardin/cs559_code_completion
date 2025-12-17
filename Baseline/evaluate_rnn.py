"""
Evaluate an RNN baseline (RNN/LSTM/GRU) on the test set.

Reports loss + accuracy for BOTH tasks:
- token-level: next-token accuracy
- line-level: suffix token accuracy (and also exact-match accuracy for the suffix)
"""

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


def _gather_logits_at_positions(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Pick logits[:, pos[b], :] for each batch element b."""
    bsz = logits.size(0)
    batch_idx = torch.arange(bsz, device=logits.device)
    return logits[batch_idx, positions, :]


def evaluate_token_level_rnn(model: nn.Module,
                             test_loader: DataLoader,
                             pad_idx: int,
                             device: str = "cuda"):
    """
    Token-level evaluation:
    - input_ids = context + target + PAD...
    - predict target from the last context position (nonpad_count - 2)
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating RNN (token-level)"):
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

    avg_loss = loss_sum / max(1, total)
    accuracy = correct / max(1, total)

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "total_examples": int(total),
    }


def evaluate_line_level_rnn(model: nn.Module,
                            test_loader: DataLoader,
                            config: RNNConfig,
                            pad_idx: int,
                            device: str = "cuda"):
    """
    Line-level evaluation (teacher forcing):
    Reports:
    - average CE loss per valid suffix token
    - suffix token accuracy
    - exact-match accuracy over the suffix (teacher-forced; ignores PAD positions)
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    loss_token_sum = 0.0
    correct_tokens = 0
    total_tokens = 0

    exact_correct = 0
    exact_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating RNN (line-level)"):
            context_ids = batch["context_ids"].to(device)  # [B, max_len]
            suffix_ids = batch["suffix_ids"].to(device)    # [B, max_suffix]

            steps = suffix_ids.size(1) - 1
            if steps <= 0:
                continue

            # If a batch has no valid suffix tokens, skip it
            target_block = suffix_ids[:, :steps]
            if (target_block != pad_idx).sum().item() == 0:
                continue

            current_input = context_ids.clone()
            cur_len = (current_input != pad_idx).sum(dim=1)  # [B]

            # Track per-example exact match for the suffix (ignore PAD positions)
            exact_ok = torch.ones(context_ids.size(0), dtype=torch.bool, device=device)
            any_valid = torch.zeros(context_ids.size(0), dtype=torch.bool, device=device)

            for i in range(steps):
                next_target = suffix_ids[:, i]
                mask = (next_target != pad_idx)
                if not mask.any():
                    break

                logits = model(current_input)
                pos = (cur_len - 1).clamp(min=0)
                next_logits = _gather_logits_at_positions(logits, pos)

                preds = next_logits.argmax(dim=-1)

                # loss + token accuracy only over valid tokens
                loss_step = criterion(next_logits[mask], next_target[mask])
                valid_here = mask.sum().item()

                loss_token_sum += float(loss_step.item()) * valid_here
                correct_tokens += (preds[mask] == next_target[mask]).sum().item()
                total_tokens += valid_here

                # exact match bookkeeping
                any_valid |= mask
                exact_ok[mask] &= (preds[mask] == next_target[mask])

                # teacher forcing append (same logic as training)
                idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

                has_space = cur_len[idx] < config.max_len
                idx_space = idx[has_space]
                if idx_space.numel() > 0:
                    pos_write = cur_len[idx_space]
                    current_input[idx_space, pos_write] = next_target[idx_space]
                    cur_len[idx_space] = cur_len[idx_space] + 1

                idx_full = idx[~has_space]
                if idx_full.numel() > 0:
                    current_input[idx_full, :-1] = current_input[idx_full, 1:]
                    current_input[idx_full, -1] = next_target[idx_full]
                    cur_len[idx_full] = config.max_len

            # Only count examples that actually had at least one valid suffix token
            valid_examples = any_valid.sum().item()
            if valid_examples > 0:
                exact_correct += (exact_ok & any_valid).sum().item()
                exact_total += valid_examples

    avg_loss = loss_token_sum / max(1, total_tokens)
    token_acc = correct_tokens / max(1, total_tokens)
    exact_acc = exact_correct / max(1, exact_total)

    return {
        "loss": float(avg_loss),
        "accuracy": float(token_acc),            # suffix token accuracy
        "exact_match_accuracy": float(exact_acc),
        "total_tokens": int(total_tokens),
        "total_examples": int(exact_total),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RNN baseline on test set")

    parser.add_argument("--dataset_dir", type=str, default="completion_datasets")
    parser.add_argument("--task", type=str, choices=["token", "line"], default="token")
    parser.add_argument("--vocab_path", type=str, default="vocab.json")
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=256,
                        help="Must match the context buffer length used in training.")
    parser.add_argument("--max_suffix_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lazy_load", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)

    # architecture must match training
    parser.add_argument("--rnn_type", type=str, choices=["rnn", "lstm", "gru"], default="lstm")
    parser.add_argument("--nonlinearity", type=str, choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max_test_examples", type=int, default=None,
                        help="Limit test examples for faster eval (optional).")

    args = parser.parse_args()
    print(f"Using device: {args.device}")

    vocab = load_vocab(args.vocab_path)
    pad_idx = vocab.token_to_idx["<PAD>"]
    vocab_size = len(vocab.token_to_idx)

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

    print(f"Loading model weights from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)

    if args.task == "token":
        test_dataset = TokenLevelDataset(
            os.path.join(args.dataset_dir, "token_level", "test.jsonl"),
            vocab, args.max_length, lazy_load=args.lazy_load, max_examples=args.max_test_examples
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_token_level, num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )

        metrics = evaluate_token_level_rnn(model, test_loader, pad_idx=pad_idx, device=args.device)
        print("\nToken-level RNN evaluation:")
        print(json.dumps(metrics, indent=2))

    else:
        test_dataset = LineLevelDataset(
            os.path.join(args.dataset_dir, "line_level", "test.jsonl"),
            vocab, args.max_length, max_suffix_length=args.max_suffix_length,
            lazy_load=args.lazy_load, max_examples=args.max_test_examples
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_line_level, num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )

        metrics = evaluate_line_level_rnn(
            model, test_loader, config=config, pad_idx=pad_idx, device=args.device
        )
        print("\nLine-level RNN evaluation:")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
