import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class RandomDataset(Dataset):
    def __init__(self, length: int):
        self.len = length
        self.data = [(torch.randn(20), torch.randn(1)) for _ in range(length)]

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.len
    
def ddp_setup(rank: int, world_size: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)

    model = torch.nn.Linear(20, 1)
    model.to(rank)
    dataset = RandomDataset(100)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, shuffle=False)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    model = DDP(model, device_ids=[rank])

    for epoch in range(10):
        dataloader.sampler.set_epoch(epoch)
        for src, tgt in dataloader:
            print(src.shape)
            src = src.to(rank)
            tgt = tgt.to(rank)
            outputs = model(src)
            loss = F.mse_loss(outputs, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        print(f"Rank {rank}, Epoch {epoch}, Loss {loss.item()}")

        if rank == 0 and (epoch + 1) % 5 == 0:
            torch.save(model.module.state_dict(), f"model_{epoch}.pt")
            print(f"Rank {rank}, Epoch {epoch}, Saved model to model_{epoch}.pt")

    if rank == 0:
        print("Done!")

    cleanup()
            

if __name__ == "__main__":
    devices = torch.cuda.device_count()
    mp.spawn(main, args=(devices,), nprocs=devices)