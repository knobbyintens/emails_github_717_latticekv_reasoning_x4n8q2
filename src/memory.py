"""Sparse memory lattice for key-value cache compression."""

import torch
import torch.nn as nn


class MemoryLattice(nn.Module):
    """Hierarchical sparse store with reconstruction-aware eviction."""

    def __init__(self, num_slots: int, slot_dim: int, branch_factor: int = 4):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.branch_factor = branch_factor
        self.keys = nn.Parameter(torch.zeros(num_slots, slot_dim))
        self.values = nn.Parameter(torch.zeros(num_slots, slot_dim))
        self.router = nn.Linear(slot_dim, branch_factor, bias=False)

    @torch.no_grad()
    def evict(self, usage: torch.Tensor, keep_ratio: float = 0.75) -> torch.Tensor:
        """Return a mask of slots that survive the eviction step."""
        flat = usage.reshape(-1)
        keep = max(1, int(flat.numel() * keep_ratio))
        threshold = torch.topk(flat, keep, largest=True).values.min()
        return (usage >= threshold).to(usage.dtype)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bsd,nd->bsn", queries, self.keys)
        gate = torch.softmax(self.router(queries), dim=-1).mean(dim=-1)
        weights = torch.softmax(logits, dim=-1) * gate.unsqueeze(-1)
        return torch.einsum("bsn,nd->bsd", weights, self.values)
