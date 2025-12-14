from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class RNNConfig:
    vocab_size: int
    d_model: int = 512
    num_layers: int = 2
    rnn_type: str = "lstm"   # "lstm" or "gru"
    dropout: float = 0.1
    max_len: int = 256       # used for line-level truncation in training/eval


class RNNLanguageModel(nn.Module):
    """
    Simple RNN language model with embedding + LSTM/GRU + linear head.
    The forward API matches the Transformer model: forward(input_ids)
    returns logits of shape [batch_size, seq_len, vocab_size].
    """

    def __init__(self, config: RNNConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        rnn_type = config.rnn_type.lower()
        if rnn_type == "lstm":
            rnn_class = nn.LSTM
        elif rnn_type == "gru":
            rnn_class = nn.GRU
        else:
            raise ValueError(f"Unknown rnn_type: {config.rnn_type}. Use 'lstm' or 'gru'.")

        self.rnn = rnn_class(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len] of token indices
        returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed input tokens
        x = self.embedding(input_ids)     # [B, T, d_model]

        # RNN over the sequence
        outputs, _ = self.rnn(x)         # [B, T, d_model]

        # Dropout + projection to vocab
        outputs = self.dropout(outputs)  # [B, T, d_model]
        logits = self.fc_out(outputs)    # [B, T, vocab_size]

        return logits
