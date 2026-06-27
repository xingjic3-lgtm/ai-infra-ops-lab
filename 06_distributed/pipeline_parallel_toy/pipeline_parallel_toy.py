import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stage0(x):
    return x * 2.0 + 1.0


def stage1(hidden):
    return hidden - 3.0


def worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if rank == 0:
        for microbatch_id in range(3):
            x = torch.tensor([float(microbatch_id + 1)])
            hidden = stage0(x)
            dist.send(hidden, dst=1)
            print(f"rank 0: microbatch {microbatch_id}, x = {x.item()}, hidden = {hidden.item()}")

    if rank == 1:
        for microbatch_id in range(3):
            hidden = torch.empty(1)
            dist.recv(hidden, src=0)
            out = stage1(hidden)
            print(f"rank 1: microbatch {microbatch_id}, hidden = {hidden.item()}, out = {out.item()}")

    dist.destroy_process_group()


def main():
    world_size = 2
    port = find_free_port()
    mp.spawn(worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
