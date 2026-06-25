# Triton Attention

This lab implements a minimal scaled dot-product attention kernel with Triton
and compares it with PyTorch's `torch.nn.functional.scaled_dot_product_attention`.

The core experiment is:

```text
q.shape = [B, H, N, D]
k.shape = [B, H, N, D]
v.shape = [B, H, N, D]

B = 2
H = 4
N = 128
D = 64
```

The computation is:

```text
scores = q @ k.T / sqrt(D)
probs = softmax(scores)
out = probs @ v
```

## Files

```text
02_triton_kernels/attention/
  attention.py
  README.md
```

## Run

From the repository root:

```powershell
python 02_triton_kernels\attention\attention.py
```

Expected output includes:

```text
q shape: torch.Size([2, 4, 128, 64])
k shape: torch.Size([2, 4, 128, 64])
v shape: torch.Size([2, 4, 128, 64])
o shape: torch.Size([2, 4, 128, 64])
expected shape: torch.Size([2, 4, 128, 64])
allclose: True
triton attention ms: ...
torch attention ms: ...
```

## Kernel Shape

This teaching kernel uses:

```text
one Triton program -> one output row
```

The launch grid is:

```python
grid = (B * H, N)
```

For the current test:

```text
B * H = 8
N = 128
grid = (8, 128)
```

That means Triton starts:

```text
8 * 128 = 1024 programs
```

Each program owns one row:

```text
out[b, h, query_index, :]
```

## Program Id Mapping

Inside the kernel:

```python
pid_bh = tl.program_id(axis=0)
pid_q = tl.program_id(axis=1)
```

These two ids choose the output row:

```text
pid_bh -> one combined [batch, head] position
pid_q  -> one query row inside sequence length N
```

For example:

```text
pid_bh = 3
pid_q = 10
```

This program computes:

```text
out for one batch/head slot, query token 10, all D hidden values
```

## Address Mapping

The Python tensor is 4D:

```python
q[b, h, n, d]
k[b, h, n, d]
v[b, h, n, d]
```

The Triton kernel treats it as flat contiguous GPU memory. Because `B` and `H`
are combined into `pid_bh`, the base offset is:

```python
bh_base = pid_bh * N * D
```

Then the query row is:

```python
q_offsets = bh_base + pid_q * D + offs_d
```

The whole key/value sequence for this batch/head slot is:

```python
kv_offsets = bh_base + offs_n[:, None] * D + offs_d[None, :]
```

This creates a block shaped like:

```text
[N, D]
```

## Attention Computation

Each program loads:

```text
q row: [D]
k block: [N, D]
v block: [N, D]
```

Then it computes one score per key row:

```python
scores = tl.sum(k * q[None, :], axis=1) * scale
```

Then it applies a numerically stable softmax:

```python
scores = scores - tl.max(scores, axis=0)
probs = tl.exp(scores)
probs = probs / tl.sum(probs, axis=0)
```

Finally it mixes all value rows:

```python
out = tl.sum(probs[:, None] * v, axis=0)
```

The output has shape:

```text
[D]
```

That single vector is written to:

```text
out[b, h, query_index, :]
```

## Validation

The Triton result is compared against PyTorch:

```python
expected = F.scaled_dot_product_attention(q, k, v)
print("max diff:", torch.max(torch.abs(o - expected)).item())
print("allclose:", torch.allclose(o, expected, atol=1e-4, rtol=1e-4))
```

The tolerance is slightly loose because attention uses dot products, exponentials,
division, and reductions. Small floating-point differences are normal.

## Takeaway

This first attention kernel is intentionally simple:

```text
program -> one query row -> compare with all key rows -> softmax -> weighted sum of V
```

It is not yet FlashAttention. The goal is to make the attention data movement
visible before optimizing it into tiled streaming blocks.
