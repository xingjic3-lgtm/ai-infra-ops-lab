# Contiguous KV Cache

This lab builds a minimal contiguous KV cache with PyTorch tensors.

The core experiment is:

```text
q.shape = [B, H, T, D]
k.shape = [B, H, T, D]
v.shape = [B, H, T, D]

B = 2
H = 4
T = 8
D = 16
MAX_SEQ_LEN = 16
```

The cache pre-allocates two contiguous buffers:

```text
k_cache.shape = [B, H, MAX_SEQ_LEN, D]
v_cache.shape = [B, H, MAX_SEQ_LEN, D]
```

During decode, token `t` writes:

```text
k_cache[:, :, t:t+1, :] = k_new
v_cache[:, :, t:t+1, :] = v_new
```

Then the current query reads the prefix:

```text
k_cache[:, :, :t+1, :]
v_cache[:, :, :t+1, :]
```

## Files

```text
04_kv_cache/contiguous_kv_cache/
  contiguous_kv_cache.py
  README.md
```

## Run

From the repository root:

```powershell
python 04_kv_cache\contiguous_kv_cache\contiguous_kv_cache.py
```

Expected output includes:

```text
step 0: write cache[:, :, 0:1, :], read cache[:, :, :1, :]
...
step 7: write cache[:, :, 7:8, :], read cache[:, :, :8, :]
q shape: torch.Size([2, 4, 8, 16])
k_cache shape: torch.Size([2, 4, 16, 16])
v_cache shape: torch.Size([2, 4, 16, 16])
cached_out shape: torch.Size([2, 4, 8, 16])
expected shape: torch.Size([2, 4, 8, 16])
allclose: True
```

## Object Flow

At the beginning, the cache owns empty storage:

```python
k_cache = torch.empty((B, H, MAX_SEQ_LEN, D), device="cuda")
v_cache = torch.empty((B, H, MAX_SEQ_LEN, D), device="cuda")
```

On each decode step:

```text
new token K/V -> fixed cache slot -> prefix view -> attention output
```

For `token_idx = 3`, the write target is:

```text
k_cache[:, :, 3:4, :]
v_cache[:, :, 3:4, :]
```

The attention read range is:

```text
k_cache[:, :, :4, :]
v_cache[:, :, :4, :]
```

So the fourth output token can attend to cached tokens `0, 1, 2, 3`, but not to
future tokens.

## Validation

The script computes decode one token at a time, then compares the concatenated
result against PyTorch causal attention:

```python
causal_mask = torch.ones((T, T), device="cuda", dtype=torch.bool).tril()
expected = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
```

This verifies that repeatedly appending to the cache and reading the current
prefix produces the same result as a full causal attention call over the whole
sequence.

## Takeaway

This contiguous cache is simple because every sequence reserves a full
`MAX_SEQ_LEN` block up front:

```text
fast address calculation, simple slicing, possible unused memory
```

The next KV cache lab can compare this with paged/block-based allocation, where
tokens no longer have to live in one large contiguous sequence buffer.
