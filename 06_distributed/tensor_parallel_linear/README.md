# Tensor Parallel Linear

This lab splits the output dimension of a linear layer across two local ranks.

```text
full weight: [out_features, in_features]
rank 0 owns output rows 0..2
rank 1 owns output rows 3..5
```

Each rank computes its partial output, then `all_gather` rebuilds the full output.

## Run

```powershell
python 06_distributed\tensor_parallel_linear\tensor_parallel_linear.py
```

