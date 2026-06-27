import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)

    batch = 2
    in_features = 4
    out_features = 6
    out_per_rank = out_features // world_size

    x = torch.arange(batch * in_features, dtype=torch.float32).reshape(batch, in_features)
    full_weight = torch.arange(out_features * in_features, dtype=torch.float32).reshape(out_features, in_features)

    start = rank * out_per_rank
    end = start + out_per_rank
    local_weight = full_weight[start:end]
    local_out = torch.matmul(x, local_weight.t())

    gathered = [torch.empty_like(local_out) for _ in range(world_size)]
    dist.all_gather(gathered, local_out)
    parallel_out = torch.cat(gathered, dim=1)

    if rank == 0:
        expected = torch.matmul(x, full_weight.t())
        print("x shape:", x.shape)
        print("local_out shape:", local_out.shape)
        print("parallel_out shape:", parallel_out.shape)
        print("expected shape:", expected.shape)
        print("allclose:", torch.allclose(parallel_out, expected))

    dist.destroy_process_group()


def main():
    world_size = 2
    port = find_free_port()
    mp.spawn(worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
