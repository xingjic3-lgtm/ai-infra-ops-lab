import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idxs = k_start + offs_k

        a_offsets = offs_m[:, None] * K + k_idxs[None, :]
        b_offsets = k_idxs[:, None] * N + offs_n[None, :]

        a = tl.load(a_ptr + a_offsets, mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K), other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b, input_precision="ieee")

    c_offsets = offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)


def matmul_triton(a, b):
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert a.shape[1] == b.shape[0]

    a = a.contiguous()
    b = b.contiguous()    ## 根据b的视图创建一个新矩阵，新矩阵的内存布局是连续的，且与b的view相同

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    block_m = 16
    block_n = 16
    block_k = 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    return c


def benchmark(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat


def main():
    torch.manual_seed(0)

    M = 1024
    N = 1024
    K = 1024

    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    c = matmul_triton(a, b)
    expected = a @ b

    print("a shape:", a.shape)
    print("b shape:", b.shape)
    print("c shape:", c.shape)
    print("expected shape:", expected.shape)
    print("max diff:", torch.max(torch.abs(c - expected)).item())
    print("allclose:", torch.allclose(c, expected, atol=1e-2, rtol=1e-2))
    print("first row c:", c[0])

    triton_ms = benchmark(lambda: matmul_triton(a, b))
    torch_ms = benchmark(lambda: a @ b)

    print("triton matmul ms:", triton_ms)
    print("torch matmul ms:", torch_ms)


if __name__ == "__main__":
    main()
