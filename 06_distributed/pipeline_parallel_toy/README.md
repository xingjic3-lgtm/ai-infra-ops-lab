# Pipeline Parallel Toy

This lab splits a tiny model into two stages:

```text
rank 0: hidden = x * 2 + 1
rank 1: out = hidden - 3
```

Rank 0 sends each microbatch's hidden tensor to rank 1.

## Run

```powershell
python 06_distributed\pipeline_parallel_toy\pipeline_parallel_toy.py
```

