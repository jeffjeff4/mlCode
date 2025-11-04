import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def example(rank, world_size):
    # init process group (Gloo works for CPU)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # put model on CPU explicitly
    device = torch.device("cpu")
    model = nn.Linear(10, 10).to(device)

    # For CPU, DO NOT pass device_ids
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # create CPU tensors (no .to(rank)!)
    inputs = torch.randn(20, 10, device=device)
    labels = torch.randn(20, 10, device=device)

    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()



#--------------------------------------------
#wrong
####import torch
####import os
#####import distutils
#####import dist
####import torch.distributed as dist
####import torch.multiprocessing as mp
####import torch.nn as nn
####import torch.optim as optim
####from torch.nn.parallel import DistributedDataParallel as DDP
####
####import os
####os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#####device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
####device = torch.device("cpu")
####
####
####def example(rank, world_size):
####    # create default process group
####    dist.init_process_group("gloo", rank=rank, world_size=world_size)
####    # create local model
####    #model = nn.Linear(10, 10).to(rank)
####    model = nn.Linear(10, 10).to(device)
####    # construct DDP model
####    ddp_model = DDP(model, device_ids=[rank])
####    # define loss function and optimizer
####    loss_fn = nn.MSELoss()
####    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
####
####    # forward pass
####    outputs = ddp_model(torch.randn(20, 10).to(rank))
####    labels = torch.randn(20, 10).to(rank)
####    # backward pass
####    loss_fn(outputs, labels).backward()
####    # update parameters
####    optimizer.step()
####
####def main():
####    world_size = 2
####    mp.spawn(example,
####        args=(world_size,),
####        nprocs=world_size,
####        join=True)
####
####if __name__=="__main__":
####    # Environment variables which need to be
####    # set when using c10d's default "env"
####    # initialization mode.
####    os.environ["MASTER_ADDR"] = "localhost"
####    os.environ["MASTER_PORT"] = "29500"
####    main()





