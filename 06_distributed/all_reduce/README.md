# All Reduce

This lab starts two local processes and runs one `all_reduce`.

Each rank begins with a different tensor:

```text
rank 0 -> [1]
rank 1 -> [2]
```

After SUM all-reduce, every rank has:

```text
[3]
```

## Run

```powershell
python 06_distributed\all_reduce\all_reduce.py
```

