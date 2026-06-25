import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(axis=0)
    pid_q = tl.program_id(axis=1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    bh_base = pid_bh * N * D
    q_offsets = bh_base + pid_q * D + offs_d
    kv_offsets = bh_base + offs_n[:, None] * D + offs_d[None, :]

    q = tl.load(q_ptr + q_offsets, mask=offs_d < D, other=0.0)
    k = tl.load(k_ptr + kv_offsets, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)

    scores = tl.sum(k * q[None, :], axis=1) * scale
    scores = tl.where(offs_n < N, scores, -float("inf"))

    scores = scores - tl.max(scores, axis=0)
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=0)

    v = tl.load(v_ptr + kv_offsets, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)
    out = tl.sum(probs[:, None] * v, axis=0)

    o_offsets = bh_base + pid_q * D + offs_d
    tl.store(o_ptr + o_offsets, out, mask=offs_d < D)


def attention_triton(q, k, v):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    assert q.dtype == torch.float32 and k.dtype == torch.float32 and v.dtype == torch.float32
    assert q.shape == k.shape == v.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, H, N, D = q.shape
    o = torch.empty_like(q)

    block_n = triton.next_power_of_2(N)
    block_d = triton.next_power_of_2(D)
    scale = 1.0 / math.sqrt(D)
    grid = (B * H, N)

    attention_kernel[grid](
        q,
        k,
        v,
        o,
        B,
        H,
        N,
        D,
        scale,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )

    return o


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

    B = 2
    H = 4
    N = 128
    D = 64

    q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float32)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float32)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float32)

    o = attention_triton(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    print("q shape:", q.shape)
    print("k shape:", k.shape)
    print("v shape:", v.shape)
    print("o shape:", o.shape)
    print("expected shape:", expected.shape)
    print("max diff:", torch.max(torch.abs(o - expected)).item())
    print("allclose:", torch.allclose(o, expected, atol=1e-4, rtol=1e-4))
    print("first output row:", o[0, 0, 0])

    triton_ms = benchmark(lambda: attention_triton(q, k, v))
    torch_ms = benchmark(lambda: F.scaled_dot_product_attention(q, k, v))

    print("triton attention ms:", triton_ms)
    print("torch attention ms:", torch_ms)


if __name__ == "__main__":
    main()
