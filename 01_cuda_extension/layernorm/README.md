# CUDA Extension LayerNorm

This experiment implements a minimal LayerNorm forward kernel as a PyTorch CUDA
extension.

## Files

```text
setup.py              Build script for the PyTorch CUDA extension
layernorm.cpp         Python binding and C++ entry point
layernorm_kernel.cu   CUDA kernel implementation
test_layernorm.py     Correctness test against torch.nn.functional.layer_norm
benchmark.py          Simple runtime comparison with PyTorch LayerNorm
```

## Build

Run from this directory:

```powershell
python setup.py build_ext --inplace
```

## Test

```powershell
python test_layernorm.py
```

Expected result:

```text
allclose: True
```

The latest observed correctness result:

```text
max error: 1.1920928955078125e-07
allclose: True
```

## Benchmark

```powershell
python benchmark.py
```

Latest observed result on input shape `4096 x 1024`:

```text
custom layernorm: 0.4351 ms
torch layer_norm: 0.0350 ms
```

The custom kernel is correct, but much slower than PyTorch's optimized
implementation.

## Current Kernel Mapping

The kernel launch is:

```cpp
layernorm_kernel<<<rows, 256>>>(...);
```

For an input tensor shaped:

```text
[rows, cols] = [4096, 1024]
```

the mapping is:

```text
one CUDA block -> one row
256 threads in that block -> workers for that row
```

Inside the kernel:

```cpp
int row = blockIdx.x;
int tid = threadIdx.x;
int block_size = blockDim.x;
```

Meaning:

```text
blockIdx.x   chooses the row
threadIdx.x  chooses the worker inside that row
blockDim.x   is the number of workers in the block, currently 256
```

Each thread handles columns with:

```cpp
for (int col = tid; col < cols; col += block_size)
```

For `cols = 1024` and `block_size = 256`:

```text
thread 0   handles col 0,   256, 512, 768
thread 1   handles col 1,   257, 513, 769
...
thread 255 handles col 255, 511, 767, 1023
```

## Kernel Flow

For each row:

1. Each thread computes a partial sum in `local_sum`.
2. Threads write partial sums into `shared_sum[tid]`.
3. A block-level reduction combines all partial sums into `shared_sum[0]`.
4. `mean = shared_sum[0] / cols`.
5. Each thread computes a partial variance sum in `local_var`.
6. Threads write partial variance sums into `shared_var[tid]`.
7. Another reduction combines them into `shared_var[0]`.
8. `var = shared_var[0] / cols`.
9. Each thread writes normalized output:

```cpp
y[row * cols + col] = (x[row * cols + col] - mean) / sqrtf(var + 1e-5f);
```

## Memory And Ownership

```text
x, y
  Global memory. All blocks and threads can access them through pointers.

shared_sum, shared_var
  Shared memory. One copy per block, shared by threads inside that block.

local_sum, local_var, diff
  Thread-local values. Each thread has its own copy.
```

The key mental model:

```text
math task -> CUDA logical container -> hardware execution

LayerNorm rows -> CUDA blocks -> scheduled onto SMs
row elements    -> CUDA threads inside a block
```
