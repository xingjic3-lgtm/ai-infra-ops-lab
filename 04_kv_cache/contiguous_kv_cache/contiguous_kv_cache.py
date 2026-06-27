import math

import torch
import torch.nn.functional as F


class ContiguousKVCache:
    def __init__(self, batch_size, num_heads, max_seq_len, head_dim, device, dtype):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        self.k_cache = torch.empty(
            (batch_size, num_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.empty(
            (batch_size, num_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.seq_len = 0

    def append(self, k_new, v_new):
        assert k_new.shape == (self.batch_size, self.num_heads, 1, self.head_dim)
        assert v_new.shape == (self.batch_size, self.num_heads, 1, self.head_dim)
        assert self.seq_len < self.max_seq_len

        token_pos = self.seq_len
        self.k_cache[:, :, token_pos : token_pos + 1, :] = k_new
        self.v_cache[:, :, token_pos : token_pos + 1, :] = v_new
        self.seq_len += 1

    def current_kv(self):
        return (
            self.k_cache[:, :, : self.seq_len, :],
            self.v_cache[:, :, : self.seq_len, :],
        )


def decode_attention(q_new, k_cache_view, v_cache_view):
    scale = 1.0 / math.sqrt(q_new.shape[-1])
    scores = torch.matmul(q_new, k_cache_view.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_cache_view)


def main():
    assert torch.cuda.is_available(), "This lab expects a CUDA-capable PyTorch setup."
    torch.manual_seed(0)

    B = 2
    H = 4
    T = 8
    D = 16
    MAX_SEQ_LEN = 16

    q = torch.randn((B, H, T, D), device="cuda", dtype=torch.float32)
    k = torch.randn((B, H, T, D), device="cuda", dtype=torch.float32)
    v = torch.randn((B, H, T, D), device="cuda", dtype=torch.float32)

    cache = ContiguousKVCache(
        batch_size=B,
        num_heads=H,
        max_seq_len=MAX_SEQ_LEN,
        head_dim=D,
        device="cuda",
        dtype=torch.float32,
    )

    decode_outputs = []
    for token_idx in range(T):
        q_new = q[:, :, token_idx : token_idx + 1, :]
        k_new = k[:, :, token_idx : token_idx + 1, :]
        v_new = v[:, :, token_idx : token_idx + 1, :]

        cache.append(k_new, v_new)
        k_view, v_view = cache.current_kv()
        out_new = decode_attention(q_new, k_view, v_view)
        decode_outputs.append(out_new)

        print(
            f"step {token_idx}: "
            f"write cache[:, :, {token_idx}:{token_idx + 1}, :], "
            f"read cache[:, :, :{cache.seq_len}, :]"
        )

    cached_out = torch.cat(decode_outputs, dim=2)

    causal_mask = torch.ones((T, T), device="cuda", dtype=torch.bool).tril()
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)

    print("q shape:", q.shape)
    print("k_cache shape:", cache.k_cache.shape)
    print("v_cache shape:", cache.v_cache.shape)
    print("cached_out shape:", cached_out.shape)
    print("expected shape:", expected.shape)
    print("max diff:", torch.max(torch.abs(cached_out - expected)).item())
    print("allclose:", torch.allclose(cached_out, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    main()
