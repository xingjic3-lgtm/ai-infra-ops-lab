# Block Allocation Log

This lab logs the lifetime of KV cache blocks.

The allocator owns a small pool:

```text
free blocks = [0, 1, 2, 3, 4, 5]
```

Requests allocate a new block when their current block is full. When a request
finishes, all its blocks return to the free pool.

## Run

```powershell
python 05_vllm_hacking\block_allocation_log\block_allocation_log.py
```

