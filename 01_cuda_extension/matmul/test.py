import torch
import matmul_ext

a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

c = matmul_ext.forward(a, b)
ref = a @ b

max_error = (c - ref).abs().max()

print("a shape:", a.shape)
print("b shape:", b.shape)
print("c shape:", c.shape)
print("ref shape:", ref.shape)
print("max error:", max_error.item())
print("allclose:", torch.allclose(c, ref, atol=1e-3))

iters = 100

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(iters):
    c = matmul_ext.forward(a, b)
end.record()
torch.cuda.synchronize()
custom_ms = start.elapsed_time(end) / iters

start.record()
for _ in range(iters):
    ref = a @ b
end.record()
torch.cuda.synchronize()
torch_ms = start.elapsed_time(end) / iters

print("custom matmul ms:", custom_ms)
print("torch matmul ms:", torch_ms)