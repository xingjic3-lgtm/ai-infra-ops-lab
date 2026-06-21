# vector_add

This experiment implements a minimal PyTorch CUDA extension for vector addition.

The goal is to understand the complete path from a Python `torch.Tensor` call to a
custom CUDA kernel launch.

## Files

- `test.py`: Python test script and entry point
- `setup.py`: build script for the PyTorch CUDA extension
- `binding.cpp`: pybind11 binding between Python and C++/CUDA
- `vector_add_cuda.cu`: CUDA kernel and C++ launcher

## Call Flow

```text
test.py
  -> import vector_add_ext
  -> vector_add_ext.vector_add(a, b)
  -> binding.cpp: vector_add(a, b)
  -> vector_add_cuda.cu: vector_add_cuda(a, b)
  -> vector_add_kernel<<<blocks, threads>>>
```

## Build

Activate the conda environment first:

```powershell
conda activate cuda-lab
```

Build and install the extension:

```powershell
cd 01_cuda_extension/vector_add
python setup.py install
```

## Run

```powershell
python .\test.py
```

Expected output:

```text
a: tensor([1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')
b: tensor([1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')
c: tensor([2., 2., 2., 2., 2., 2., 2., 2.], device='cuda:0')
expected: tensor([2., 2., 2., 2., 2., 2., 2., 2.], device='cuda:0')
allclose: True
```

`allclose: True` means the custom CUDA kernel result matches PyTorch's built-in
addition result.

## Key Ideas

### Tensor to pointer

In Python, `a` and `b` are `torch.Tensor` objects. In the CUDA launcher:

```cpp
a.data_ptr<float>()
```

gets the starting address of the tensor data in GPU memory.

### Thread index

Each CUDA thread computes one output element:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

This is a logical global thread index, not a physical GPU core index.

### Kernel work

```cpp
c[idx] = a[idx] + b[idx];
```

Each valid thread reads one element from `a`, one element from `b`, adds them, and
writes one element to `c`.

### Bounds check

```cpp
if (idx < n)
```

The last block may contain extra threads when `n` is not exactly divisible by the
block size. This check prevents out-of-bounds memory access.

## Environment Notes

This experiment was tested with:

- Windows
- Python 3.10
- PyTorch nightly CUDA 12.8
- CUDA 12.8 `nvcc` from conda
- NVIDIA GeForce RTX 5060 Ti

For the general environment setup, see the root [README.md](../../README.md).
