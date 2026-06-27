import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)

    model = torch.nn.Linear(1, 1, bias=False)
    ddp_model = DDP(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    x = torch.tensor([[float(rank + 1)]])
    y = torch.tensor([[2.0 * float(rank + 1)]])

    pred = ddp_model(x)
    loss = (pred - y).pow(2).mean()
    loss.backward()

    grad = ddp_model.module.weight.grad.item()
    optimizer.step()

    print(f"rank {rank}: averaged grad = {grad:.4f}")
    print(f"rank {rank}: updated weight = {ddp_model.module.weight.item():.4f}")

    dist.destroy_process_group()


def main():
    world_size = 2
    port = find_free_port()
    mp.spawn(worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
