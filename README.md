# LatticeKV

Official implementation of **LatticeKV: Sparse Memory Compression for Long-Context Reasoning**.

> Paper status: Under review at COLM 2024.

## Repository Status

Released. This repository contains the complete implementation used for the
paper: data preparation, the sparse memory model, the training loop, the
evaluation code, and the released configuration and result files.

## What is included

```
src/
  memory.py      sparse memory lattice and eviction policy
  model.py       transformer wrapper with LatticeKV memory
  train.py       training entry point
eval/
  evaluate.py    long-context evaluation harness
experiments/
  colm_main.yaml main experimental configuration
results/
  main_table.json released numbers for Tables 2-4
scripts/
  preprocess.sh  data preparation
  train.sh       training launcher
  evaluate.sh    evaluation launcher
```

## Quick start

```bash
pip install -r requirements.txt
bash scripts/preprocess.sh
bash scripts/train.sh experiments/colm_main.yaml
bash scripts/evaluate.sh results/main_table.json
```

The released configuration reproduces the 128k-context results reported in
the paper on a single 8x A100 node.

## Citation

```bibtex
@inproceedings{scott2024latticekv,
  title={LatticeKV: Sparse Memory Compression for Long-Context Reasoning},
  author={Scott, Jeffrey and Alvarez, Maria and Bhatt, Rohan},
  booktitle={COLM},
  year={2024}
}
```
