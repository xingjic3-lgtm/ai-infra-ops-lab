# Triton Matmul

This lab implements a minimal matrix multiplication kernel with Triton and
compares it with PyTorch's `a @ b`.

The core experiment is:

```text
A.shape = [M, K]
B.shape = [K, N]
C.shape = [M, N]

M = 1024
N = 1024
K = 1024
```

The computation is:

```text
C[row, col] = sum(A[row, k] * B[k, col]) for k in 0..K-1
```

## Files

```text
02_triton_kernels/matmul/
  matmul.py
  README.md
```

## Run

From the repository root:

```powershell
python 02_triton_kernels\matmul\matmul.py
```

Expected output includes:

```text
a shape: torch.Size([1024, 1024])
b shape: torch.Size([1024, 1024])
c shape: torch.Size([1024, 1024])
expected shape: torch.Size([1024, 1024])
allclose: True
triton matmul ms: ...
torch matmul ms: ...
```

## Kernel Shape

The CUDA extension baseline used this idea:

```text
one CUDA thread -> one C[row, col]
```

This Triton version uses:

```text
one Triton program -> one tile of C
```

The current tile size is:

```python
block_m = 16
block_n = 16
block_k = 32
```

So one Triton program computes a `16 x 16` output tile:

```text
256 output values from C
```

The launch grid is:

```python
grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
```

For `M = 1024` and `N = 1024`:

```text
grid = (64, 64)
```

That means Triton starts:

```text
64 * 64 = 4096 programs
```

Each program owns one `16 x 16` tile of output `C`.

## Program Id Mapping

Inside the kernel:

```python
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)
```

These two program ids choose which output tile the current program owns:

```text
pid_m -> tile position along rows of C
pid_n -> tile position along columns of C
```

For example:

```text
pid_m = 2
pid_n = 3
BLOCK_M = 16
BLOCK_N = 16
```

Then this program owns:

```text
rows 32..47 of C
cols 48..63 of C
```

The code builds those row and column vectors:

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

## K Dimension Loop

One output value needs the whole `K` dimension:

```text
C[row, col] = A[row, 0] * B[0, col]
            + A[row, 1] * B[1, col]
            + ...
            + A[row, K-1] * B[K-1, col]
```

The Triton kernel does not load the whole `K = 1024` dimension at once. It walks
through `K` in chunks:

```python
for k_start in range(0, K, BLOCK_K):
```

With `BLOCK_K = 32`, each loop loads:

```text
A tile: [16, 32]
B tile: [32, 16]
```

Then Triton computes:

```python
acc += tl.dot(a, b)
```

The accumulator shape is:

```text
acc: [16, 16]
```

After all `K` chunks are processed, `acc` contains the final `16 x 16` tile of
`C`.

## Address Mapping

The input tensors look 2D in Python:

```python
a[row, k]
b[k, col]
c[row, col]
```

Inside the Triton kernel, the tensors are pointer-based flat GPU memory.

For contiguous row-major tensors:

```text
A[row, k] -> a_ptr + row * K + k
B[k, col] -> b_ptr + k * N + col
C[row, col] -> c_ptr + row * N + col
```

That is why the kernel builds offsets like this:

```python
a_offsets = offs_m[:, None] * K + k_idxs[None, :]
b_offsets = k_idxs[:, None] * N + offs_n[None, :]
c_offsets = offs_m[:, None] * N + offs_n[None, :]
```

This is the key Triton move:

```text
row vector + column vector -> matrix of memory offsets
```

## Masks

The current test shape is exactly divisible by the block sizes, but the kernel
still uses masks:

```python
mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K)
mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N)
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
```

The masks make the same kernel work for shapes like:

```text
M = 1000
N = 1000
K = 1000
```

In those cases, some programs at the edge of the matrix would point outside the
real tensor. The mask tells Triton which lanes are valid.

## Validation

The Triton output is compared against PyTorch:

```python
expected = a @ b
print("max diff:", torch.max(torch.abs(c - expected)).item())
print("allclose:", torch.allclose(c, expected, atol=1e-2, rtol=1e-2))
```

The tolerance is looser than elementwise addition because matrix multiplication
accumulates many floating-point products. Small numeric differences are normal.

## Takeaway

The mental model for this lab is:

```text
CUDA extension baseline:
  thread -> one C[row, col] -> loop over K

Triton matmul baseline:
  program -> one C tile -> loop over K blocks -> tl.dot
```

The important thing to understand in this experiment is not peak performance.
It is how Triton maps a 2D output tile to GPU memory and uses `tl.dot` to express
the inner matrix multiplication work.
