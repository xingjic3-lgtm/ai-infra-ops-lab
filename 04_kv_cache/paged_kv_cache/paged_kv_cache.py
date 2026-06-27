import torch


class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_heads, head_dim, device, dtype):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.k_blocks = torch.empty((num_blocks, block_size, num_heads, head_dim), device=device, dtype=dtype)
        self.v_blocks = torch.empty((num_blocks, block_size, num_heads, head_dim), device=device, dtype=dtype)
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}
        self.seq_lens = {}

    def allocate_sequence(self, seq_id):
        self.block_tables[seq_id] = []
        self.seq_lens[seq_id] = 0

    def append(self, seq_id, k_new, v_new):
        assert k_new.shape == (self.num_heads, self.head_dim)
        assert v_new.shape == (self.num_heads, self.head_dim)

        token_pos = self.seq_lens[seq_id]
        block_slot = token_pos // self.block_size
        offset_in_block = token_pos % self.block_size

        if block_slot == len(self.block_tables[seq_id]):
            physical_block = self.free_blocks.pop(0)
            self.block_tables[seq_id].append(physical_block)
        else:
            physical_block = self.block_tables[seq_id][block_slot]

        self.k_blocks[physical_block, offset_in_block] = k_new
        self.v_blocks[physical_block, offset_in_block] = v_new
        self.seq_lens[seq_id] += 1

        return physical_block, offset_in_block

    def gather_sequence(self, seq_id):
        seq_len = self.seq_lens[seq_id]
        k_tokens = []
        v_tokens = []

        for token_pos in range(seq_len):
            block_slot = token_pos // self.block_size
            offset_in_block = token_pos % self.block_size
            physical_block = self.block_tables[seq_id][block_slot]

            k_tokens.append(self.k_blocks[physical_block, offset_in_block])
            v_tokens.append(self.v_blocks[physical_block, offset_in_block])

        return torch.stack(k_tokens, dim=0), torch.stack(v_tokens, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    num_heads = 2
    head_dim = 4
    block_size = 4

    cache = PagedKVCache(
        num_blocks=4,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
        dtype=torch.float32,
    )

    seq_a = "request_a"
    seq_b = "request_b"
    cache.allocate_sequence(seq_a)
    cache.allocate_sequence(seq_b)

    tokens = [
        (seq_a, 0),
        (seq_a, 1),
        (seq_b, 0),
        (seq_a, 2),
        (seq_b, 1),
        (seq_a, 3),
        (seq_a, 4),
    ]

    for seq_id, token_value in tokens:
        k_new = torch.full((num_heads, head_dim), float(token_value), device=device)
        v_new = torch.full((num_heads, head_dim), float(token_value + 100), device=device)
        block_id, offset = cache.append(seq_id, k_new, v_new)
        print(f"{seq_id}: token {token_value} -> block {block_id}, offset {offset}")
        print(f"{seq_id}: block table = {cache.block_tables[seq_id]}")

    k_a, v_a = cache.gather_sequence(seq_a)
    k_b, v_b = cache.gather_sequence(seq_b)

    print("request_a k shape:", k_a.shape)
    print("request_a v shape:", v_a.shape)
    print("request_a k first head:")
    print(k_a[:, 0, :])
    print("request_b k first head:")
    print(k_b[:, 0, :])


if __name__ == "__main__":
    main()
