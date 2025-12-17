from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class RNNConfig:
    """
    Configuration for an RNN language model baseline.

    Notes:
    - pad_idx is used to set embedding.padding_idx (prevents updating PAD embeddings).
    - d_model is used for both embedding size and hidden size to keep the model simple.
    """
    vocab_size: int
    pad_idx: int = 0
    d_model: int = 512
    num_layers: int = 2
    rnn_type: str = "lstm"          # "rnn", "lstm", or "gru"
    nonlinearity: str = "tanh"      # only used if rnn_type == "rnn" ("tanh" or "relu")
    dropout: float = 0.1
    max_len: int = 256             # used for line-level truncation in training/eval


class RNNLanguageModel(nn.Module):
    """
    Simple language model:
      Embedding -> (RNN/LSTM/GRU) -> Dropout -> Linear vocab projection

    forward(input_ids) returns logits with shape [B, T, V].
    """

    def __init__(self, config: RNNConfig):
        super().__init__()
        self.config = config

        # padding_idx prevents gradients from updating the PAD embedding row
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)

        rnn_type = config.rnn_type.lower()
        if rnn_type == "lstm":
            rnn_class = nn.LSTM
            rnn_kwargs = {}
        elif rnn_type == "gru":
            rnn_class = nn.GRU
            rnn_kwargs = {}
        elif rnn_type == "rnn":
            rnn_class = nn.RNN
            rnn_kwargs = {"nonlinearity": config.nonlinearity}
        else:
            raise ValueError(f"Unknown rnn_type: {config.rnn_type}. Use 'rnn', 'lstm', or 'gru'.")

        self.rnn = rnn_class(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            **rnn_kwargs,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T] token indices
        returns:   [B, T, V] logits
        """
        x = self.embedding(input_ids)     # [B, T, d_model]
        out, _ = self.rnn(x)              # [B, T, d_model]
        out = self.dropout(out)           # [B, T, d_model]
        logits = self.fc_out(out)         # [B, T, V]
        return logits
