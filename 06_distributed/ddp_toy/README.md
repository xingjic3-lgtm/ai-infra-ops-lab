# DDP Toy

This lab starts two local processes and wraps one `Linear` layer with PyTorch
DistributedDataParallel.

Each rank owns different data, computes a local gradient, and DDP averages the
gradients before the optimizer step.

## Run

```powershell
python 06_distributed\ddp_toy\ddp_toy.py
```

Expected output includes matching gradients and matching updated weights on both
ranks.

