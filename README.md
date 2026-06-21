# AI Infra Ops Lab

This repository records my hands-on learning path for AI infrastructure engineering.

The current focus is to build small, runnable experiments around CUDA extensions,
Triton kernels, TensorRT, KV cache, vLLM internals, and distributed systems.

See [ROADMAP.md](./ROADMAP.md) for the full lab layout.

## Environment Setup

The commands below assume:

- Windows
- Anaconda or Miniconda is already installed
- No PyTorch is installed yet
- No usable CUDA `nvcc` is required globally

For compiling PyTorch CUDA extensions on Windows, Visual Studio Build Tools are also
needed. Install the C++ build tools workload if `cl.exe` or MSVC linker errors appear.

### 1. Create conda environment

```powershell
conda create -n cuda-lab python=3.10 -y
conda activate cuda-lab
```

### 2. Install PyTorch with CUDA 12.8

This repository was tested on an RTX 5060 Ti. For RTX 50-series GPUs, use a PyTorch
build that supports `sm_120`.

```powershell
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --timeout 120 --retries 10 --no-cache-dir
```

### 3. Install CUDA 12.8 compiler tools in conda

PyTorch CUDA extensions should be compiled with a CUDA toolkit version matching
`torch.version.cuda`.

```powershell
conda install -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8 -y
```

### 4. Set CUDA build paths for the current terminal

On Windows conda environments, CUDA files are under `Library`, so set these
environment variables before building:

```powershell
$env:CUDA_HOME="$env:CONDA_PREFIX\Library"
$env:CUDA_PATH="$env:CONDA_PREFIX\Library"
$env:LIB="$env:CONDA_PREFIX\Library\lib;$env:LIB"
```

### 5. Install Python build helpers

```powershell
python -m pip install setuptools wheel ninja
```

## Notes

If PyTorch reports that the GPU architecture is not supported, check:

```powershell
python -c "import torch; print(torch.cuda.get_arch_list())"
```

For RTX 50-series GPUs, the list should include:

```text
sm_120
```

If extension compilation reports a CUDA version mismatch, check:

```powershell
python -c "import torch; print(torch.version.cuda)"
nvcc --version
echo $env:CUDA_HOME
```

The CUDA version used by PyTorch and the CUDA toolkit used for extension
compilation should match, for example both `12.8`.
