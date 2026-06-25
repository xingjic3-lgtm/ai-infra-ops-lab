from pathlib import Path

import tensorrt as trt
import torch
import torch.nn as nn


ENGINE_PATH = Path("03_tensorrt/build_engine/model.engine")
WARMUP_ITERS = 20
BENCH_ITERS = 2000


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


def load_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine_bytes = ENGINE_PATH.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("failed to load TensorRT engine")
    return engine


def time_cuda(fn, warmup_iters=WARMUP_ITERS, bench_iters=BENCH_ITERS):
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(bench_iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / bench_iters


def main():
    torch.manual_seed(0)

    engine = load_engine()
    context = engine.create_execution_context()

    input_name = "input"
    output_name = "output"
    batch_size = 4

    x = torch.randn(batch_size, 16, device="cuda")
    y_trt = torch.empty(batch_size, 32, device="cuda")

    context.set_input_shape(input_name, tuple(x.shape))
    context.set_tensor_address(input_name, x.data_ptr())
    context.set_tensor_address(output_name, y_trt.data_ptr())

    torch.manual_seed(0)
    model = TinyModel().eval().cuda()
    stream = torch.cuda.Stream()

    def run_trt():
        context.execute_async_v3(stream.cuda_stream)

    def run_torch():
        with torch.no_grad():
            return model(x)

    with torch.cuda.stream(stream):
        run_trt()
        y_torch = run_torch()
    stream.synchronize()

    max_diff = (y_trt - y_torch).abs().max().item()
    trt_ms = time_cuda(run_trt)
    torch_ms = time_cuda(run_torch)

    print("input shape:", tuple(x.shape))
    print("TensorRT output shape:", tuple(y_trt.shape))
    print("PyTorch output shape:", tuple(y_torch.shape))
    print("max diff:", max_diff)
    print("allclose:", torch.allclose(y_trt, y_torch, atol=1e-3, rtol=1e-3))
    print(f"TensorRT latency: {trt_ms:.6f} ms")
    print(f"PyTorch latency: {torch_ms:.6f} ms")
    print(f"speedup: {torch_ms / trt_ms:.2f}x")


if __name__ == "__main__":
    main()
