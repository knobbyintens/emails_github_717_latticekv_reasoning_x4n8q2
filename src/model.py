"""Transformer wrapper that swaps the KV cache for a LatticeKV memory."""

import torch
import torch.nn as nn
from .memory import MemoryLattice


class LatticeKVAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_slots: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.memory = MemoryLattice(num_slots, self.head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        mem = self.memory(hidden_states.reshape(bsz, seq_len, self.head_dim))
        attn = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
        out = attn @ v + mem.unsqueeze(1)
        return out.reshape(bsz, seq_len, self.hidden_size)
