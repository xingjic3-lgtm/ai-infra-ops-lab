import time
import torch
import layernorm_ext


def bench(name, fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iters):
        fn()

    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000 / iters
    print(f"{name}: {avg_ms:.4f} ms")


def main():
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)

    bench("custom layernorm", lambda: layernorm_ext.layernorm(x))
    bench("torch layer_norm", lambda: torch.nn.functional.layer_norm(x, (x.shape[-1],)))


if __name__ == "__main__":
    main()