# Triton LayerNorm

This lab implements a minimal LayerNorm kernel with Triton and compares it with
PyTorch's built-in `torch.nn.functional.layer_norm`.

The core experiment is:

```text
x.shape = [M, N]
M = 4096
N = 1000
```

LayerNorm is computed independently for each row. Triton maps this naturally as:

```text
one row -> one Triton program
```

So the launch grid is:

```python
grid = (M,)
```

This starts `M` programs. Program `pid=0` handles row 0, `pid=1` handles row 1,
and so on.

## Files

```text
02_triton_kernels/layernorm/
  layernorm.py
  README.md
```

## Run

From the repository root:

```powershell
python 02_triton_kernels\layernorm\layernorm.py
```

Expected output includes:

```text
x shape: torch.Size([4096, 1000])
y shape: torch.Size([4096, 1000])
expected shape: torch.Size([4096, 1000])
allclose: True
triton time ms: ...
torch time ms: ...
```

## Kernel Shape

The wrapper prepares the output buffer and launch configuration:

```python
y = torch.empty_like(x)
M, N = x.shape
block_size = triton.next_power_of_2(N)
grid = (M,)

layernorm_kernel[grid](x, y, N, 1e-5, BLOCK_SIZE=block_size)
```

For the current test:

```text
N = 1000
BLOCK_SIZE = 1024
```

Triton uses a power-of-two block size because reductions such as `tl.sum` are
more regular at sizes like 1024. The real row only has 1000 valid elements, so
the kernel uses a mask.

## Address Mapping

Inside the kernel:

```python
pid = tl.program_id(axis=0)
cols = tl.arange(0, BLOCK_SIZE)
mask = cols < N
offsets = pid * N + cols
```

For `pid=0`:

```text
offsets = 0 * 1000 + [0..1023]
```

Only `cols 0..999` are valid. The last 24 positions are masked out.

For `pid=1`:

```text
offsets = 1 * 1000 + [0..1023]
```

Again, only the first 1000 positions belong to row 1.

This is the key translation:

```text
row id + column vector -> flat GPU memory offsets
```

## LayerNorm Computation

Each program loads one row:

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

Then it computes the row statistics:

```python
mean = tl.sum(x, axis=0) / N
diff = tl.where(mask, x - mean, 0.0)
var = tl.sum(diff * diff, axis=0) / N
y = diff * tl.rsqrt(var + eps)
```

Finally it writes only valid elements:

```python
tl.store(y_ptr + offsets, y, mask=mask)
```

The important point is that the output element `y[row, col]` cannot be computed
from `x[row, col]` alone. It depends on the whole row's mean and variance.
Triton makes this easier to express by letting one program hold a whole row
vector and reduce it with `tl.sum`.

## Validation

The Triton result is compared against PyTorch:

```python
expected = torch.nn.functional.layer_norm(x, (N,))
print("max diff:", torch.max(torch.abs(y - expected)).item())
print("allclose:", torch.allclose(y, expected, atol=1e-5, rtol=1e-5))
```

Observed result:

```text
max diff: 9.5367431640625e-07
allclose: True
```

This confirms that the masked Triton kernel matches PyTorch for `N=1000`, where
`BLOCK_SIZE=1024`.

## Timing Snapshot

On the current setup, one observed run produced:

```text
triton time ms: 0.08718688011169434
torch time ms: 0.0865824031829834
```

The goal of this lab is not to beat PyTorch in every case. The goal is to see
how a custom Triton kernel maps a row-wise tensor operation onto GPU programs,
then verify correctness and measure the result.

## Takeaway

CUDA extension code starts from threads and elements. This Triton version starts
from a row-shaped block of data:

```text
CUDA-style thinking:
  thread -> element -> manually cooperate for reduction

Triton-style thinking:
  program -> row vector -> tl.sum for row reduction
```

For LayerNorm, the row-vector view matches the computation directly.
