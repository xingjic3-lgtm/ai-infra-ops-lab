def estimate_blocks(num_requests, tokens_per_request, block_size):
    blocks_per_request = (tokens_per_request + block_size - 1) // block_size
    total_blocks = num_requests * blocks_per_request
    used_tokens = num_requests * tokens_per_request
    reserved_tokens = total_blocks * block_size
    waste_tokens = reserved_tokens - used_tokens
    return blocks_per_request, total_blocks, waste_tokens


def main():
    num_requests = 8
    tokens_per_request = 17
    block_sizes = [4, 8, 16]

    print("num_requests:", num_requests)
    print("tokens_per_request:", tokens_per_request)

    for block_size in block_sizes:
        blocks_per_request, total_blocks, waste_tokens = estimate_blocks(
            num_requests,
            tokens_per_request,
            block_size,
        )
        print(
            f"block_size={block_size}: "
            f"blocks/request={blocks_per_request}, "
            f"total_blocks={total_blocks}, "
            f"waste_tokens={waste_tokens}"
        )


if __name__ == "__main__":
    main()
