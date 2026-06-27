class BlockAllocator:
    def __init__(self, num_blocks):
        self.free = list(range(num_blocks))
        self.owned = {}

    def allocate(self, request_id):
        block_id = self.free.pop(0)
        self.owned.setdefault(request_id, []).append(block_id)
        return block_id

    def free_request(self, request_id):
        blocks = self.owned.pop(request_id)
        self.free.extend(blocks)
        self.free.sort()
        return blocks


def main():
    allocator = BlockAllocator(num_blocks=6)
    block_size = 4
    seq_lens = {"req_a": 0, "req_b": 0}
    block_tables = {"req_a": [], "req_b": []}

    events = [
        ("append", "req_a"),
        ("append", "req_a"),
        ("append", "req_a"),
        ("append", "req_a"),
        ("append", "req_a"),
        ("append", "req_b"),
        ("finish", "req_a"),
        ("append", "req_b"),
    ]

    for event, request_id in events:
        if event == "append":
            token_pos = seq_lens[request_id]
            block_slot = token_pos // block_size

            if block_slot == len(block_tables[request_id]):
                block_id = allocator.allocate(request_id)
                block_tables[request_id].append(block_id)

            seq_lens[request_id] += 1
            print(f"{request_id}: append token {token_pos}, block table = {block_tables[request_id]}")

        if event == "finish":
            released = allocator.free_request(request_id)
            block_tables[request_id] = []
            print(f"{request_id}: finish, released blocks = {released}")

        print(f"free blocks = {allocator.free}")


if __name__ == "__main__":
    main()
