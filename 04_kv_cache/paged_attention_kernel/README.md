# Paged Attention Kernel

This lab is a small Python reference for what a paged attention kernel must do:

```text
query -> block table -> gather paged K/V tokens -> attention output
```

It is not an optimized CUDA kernel. It is intentionally written as a readable
reference before writing a real low-level kernel.

## Run

```powershell
python 04_kv_cache\paged_attention_kernel\paged_attention_kernel.py
```

Expected output includes:

```text
fill token 0: block 2, offset 0
...
allclose: True
```

The key address calculation is:

```python
block_slot = token_pos // block_size
offset_in_block = token_pos % block_size
physical_block = block_table[block_slot]
```

