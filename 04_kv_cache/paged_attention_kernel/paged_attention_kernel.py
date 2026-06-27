import math

import torch


def paged_attention(q_new, k_blocks, v_blocks, block_table, seq_len, block_size):
    keys = []
    values = []

    for token_pos in range(seq_len):
        block_slot = token_pos // block_size
        offset_in_block = token_pos % block_size
        physical_block = block_table[block_slot]

        keys.append(k_blocks[physical_block, offset_in_block])
        values.append(v_blocks[physical_block, offset_in_block])

    k = torch.stack(keys, dim=0).permute(1, 0, 2)
    v = torch.stack(values, dim=0).permute(1, 0, 2)

    scale = 1.0 / math.sqrt(q_new.shape[-1])
    scores = torch.matmul(q_new, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v), probs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    num_heads = 2
    head_dim = 4
    block_size = 4
    seq_len = 6
    block_table = [2, 0]

    k_blocks = torch.zeros((3, block_size, num_heads, head_dim), device=device)
    v_blocks = torch.zeros((3, block_size, num_heads, head_dim), device=device)

    logical_k = torch.randn((seq_len, num_heads, head_dim), device=device)
    logical_v = torch.randn((seq_len, num_heads, head_dim), device=device)
    q_new = torch.randn((num_heads, 1, head_dim), device=device)

    for token_pos in range(seq_len):
        block_slot = token_pos // block_size
        offset_in_block = token_pos % block_size
        physical_block = block_table[block_slot]
        k_blocks[physical_block, offset_in_block] = logical_k[token_pos]
        v_blocks[physical_block, offset_in_block] = logical_v[token_pos]
        print(f"fill token {token_pos}: block {physical_block}, offset {offset_in_block}")

    out, probs = paged_attention(q_new, k_blocks, v_blocks, block_table, seq_len, block_size)

    logical_k_by_head = logical_k.permute(1, 0, 2)
    logical_v_by_head = logical_v.permute(1, 0, 2)
    expected_scores = torch.matmul(q_new, logical_k_by_head.transpose(-2, -1)) / math.sqrt(head_dim)
    expected_probs = torch.softmax(expected_scores, dim=-1)
    expected = torch.matmul(expected_probs, logical_v_by_head)

    print("q_new shape:", q_new.shape)
    print("paged output shape:", out.shape)
    print("attention probs shape:", probs.shape)
    print("max diff:", torch.max(torch.abs(out - expected)).item())
    print("allclose:", torch.allclose(out, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    main()
