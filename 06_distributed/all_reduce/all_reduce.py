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

    x = torch.tensor([float(rank + 1)])
    print(f"rank {rank}: before all_reduce x = {x.item()}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"rank {rank}: after all_reduce x = {x.item()}")

    dist.destroy_process_group()


def main():
    world_size = 2
    port = find_free_port()
    mp.spawn(worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
