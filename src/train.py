"""Training entry point for LatticeKV."""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from .model import LatticeKVAttention


def build_model(cfg):
    return LatticeKVAttention(
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        num_slots=cfg["num_slots"],
    )


def train(cfg):
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loader = DataLoader(range(cfg["steps"]), batch_size=cfg["batch_size"])
    for step, _ in enumerate(loader):
        hidden = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["hidden_size"])
        loss = model(hidden).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        train(yaml.safe_load(handle))
