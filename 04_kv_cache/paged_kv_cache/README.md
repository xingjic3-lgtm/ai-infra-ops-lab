# Paged KV Cache

This lab shows the smallest useful idea behind a paged KV cache.

Instead of reserving one long contiguous buffer per request, the cache owns a
pool of fixed-size blocks:

```text
k_blocks.shape = [num_blocks, block_size, num_heads, head_dim]
v_blocks.shape = [num_blocks, block_size, num_heads, head_dim]
```

Each request has a block table:

```text
logical block 0 -> physical block id
logical block 1 -> physical block id
```

## Run

```powershell
python 04_kv_cache\paged_kv_cache\paged_kv_cache.py
```

Expected output includes:

```text
request_a: token 0 -> block 0, offset 0
request_a: block table = [0]
...
request_a: token 4 -> block 2, offset 0
request_a: block table = [0, 2]
```

The important thing to read is the mapping:

```text
token_pos -> logical block slot -> physical block id -> offset inside block
```

