import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(x_ptr, y_ptr, N: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offsets = pid * N + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    y = diff * tl.rsqrt(var + eps)

    tl.store(y_ptr + offsets, y, mask=mask)


def layernorm_triton(x):
    y = torch.empty_like(x)
    M, N = x.shape
    block_size = triton.next_power_of_2(N)
    grid = (M,)

    layernorm_kernel[grid](x, y, N, 1e-5, BLOCK_SIZE=block_size)

    return y


def main():
    torch.manual_seed(0)

    M = 4096
    N = 1000
    x = torch.randn((M, N), device="cuda", dtype=torch.float32)
    y = layernorm_triton(x)
    expected = torch.nn.functional.layer_norm(x, (N,))

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("expected shape:", expected.shape)
    print("max diff:", torch.max(torch.abs(y - expected)).item())
    print("allclose:", torch.allclose(y, expected, atol=1e-5, rtol=1e-5))
    print("first row x:", x[0])
    print("first row y:", y[0])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        y = layernorm_triton(x)
    end.record()
    torch.cuda.synchronize()
    print("triton time ms:", start.elapsed_time(end) / 100)

    start.record()
    for _ in range(100):
        expected = torch.nn.functional.layer_norm(x, (N,))
    end.record()
    torch.cuda.synchronize()
    print("torch time ms:", start.elapsed_time(end) / 100)


if __name__ == "__main__":
    main()
