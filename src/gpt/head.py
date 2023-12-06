import torch
from torch import nn


class Head(nn.Module):
    """One head of the Self-Attention."""

    def __init__(
        self, head_size: int, n_embd: int, block_size: int, dropout: int
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )
        self.dropout = nn.Dropout(dropout)
