# CUDA Extension Matmul

This lab builds a minimal PyTorch CUDA extension for matrix multiplication.

Current input:

```text
A: [1024, 1024] float32 CUDA tensor
B: [1024, 1024] float32 CUDA tensor
C: [1024, 1024] float32 CUDA tensor
```

The current kernel is the baseline version:

```text
one CUDA thread -> one C[row, col]
```

Each thread computes one output element:

```cpp
float sum = 0.0f;
for (int k = 0; k < K; k++) {
    sum += a[row * K + k] * b[k * N + col];
}
c[row * N + col] = sum;
```

## Why 2D Grid And Block

The output is a matrix:

```text
C[M, N]
```

One output value is not just `C[i]`. It is:

```text
C[row, col]
```

To compute it, the thread needs:

```text
row -> choose one row from A
col -> choose one column from B
```

So this kernel uses a 2D thread layout:

```cpp
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x,
          (M + block.y - 1) / block.y);
```

Think of it as:

```text
block: one 16x16 tile of output C
grid:  how many 16x16 tiles are needed to cover C
```

Inside the kernel:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Meaning:

```text
blockIdx  -> which block tile this thread is in
blockDim  -> how large each block tile is
threadIdx -> where this thread is inside its block tile
```

Example:

```text
blockDim.y = 16
blockIdx.y = 2
threadIdx.y = 3

row = 2 * 16 + 3 = 35
```

## Why Grid Is Calculated This Way

For columns:

```cpp
grid.x = (N + block.x - 1) / block.x;
```

This is integer ceil:

```text
ceil(N / block.x)
```

For rows:

```cpp
grid.y = (M + block.y - 1) / block.y;
```

This is:

```text
ceil(M / block.y)
```

If `N = 1000` and `block.x = 16`:

```text
1000 / 16 = 62 remainder 8
```

62 blocks only cover 992 columns, so we need 63 blocks.

Extra threads are stopped by:

```cpp
if (row < M && col < N) {
    ...
}
```

## Why Tensor Access Uses 1D Index

In Python, `c` looks like a 2D tensor:

```python
c[row, col]
```

But inside the CUDA kernel, we pass:

```cpp
c.data_ptr<float>()
```

So the kernel receives:

```cpp
float* c
```

That is just a pointer to continuous GPU memory.

For a contiguous `[M, N]` tensor, rows are stored one after another:

```text
c[row, col] -> c[row * N + col]
```

So this line:

```cpp
c[row * N + col] = sum;
```

means:

```text
write sum into C[row, col]
```

## Current Correctness And Speed

Reference result:

```python
ref = a @ b
```

Current observed result:

```text
max error: 0.000217437744140625
allclose: True
custom matmul ms: 1.4002415466308593
torch matmul ms: 0.15900544166564942
```

This baseline is correct, but slower than PyTorch.

The main reason is not that it launches too many threads. The main reason is that every thread repeatedly reads from global memory:

```text
A row data
B column data
```

Many nearby threads need overlapping data, but this version does not reuse it through shared memory yet.

The next optimization direction is tiled matmul:

```text
one block cooperatively loads a tile of A and a tile of B
threads reuse those values from shared memory
```
