import torch
import torch.nn as nn # nn 模块通常会用到
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os

# --- MNIST specific imports ---
from torchvision import datasets, transforms

# 注意：MyTrainDataset 类不再需要，可以删除或注释掉
# class MyTrainDataset(Dataset):
#     ...

def ddp_setup():
    # LOCAL_RANK 由 torchrun 或 torch.distributed.launch 自动设置
    rank = int(os.environ["RANK"]) # 全局 rank
    local_rank = int(os.environ["LOCAL_RANK"]) # 单机内的 rank
    world_size = int(os.environ["WORLD_SIZE"]) # 总进程数

    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Initialized DDP for rank {rank} (local_rank {local_rank}) on device cuda:{local_rank} in world_size {world_size}")


# --- 定义适用于 MNIST 的模型 ---
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: (batch_size, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # -> (batch, 32, 28, 28)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    # -> (batch, 32, 14, 14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (batch, 64, 14, 14)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)    # -> (batch, 64, 7, 7)
        
        # 全连接层
        # 展平后的维度: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10个类别

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # 输出 logits，CrossEntropyLoss 会处理 softmax
        return x

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"]) # 使用 local_rank 作为 gpu_id
        self.global_rank = int(os.environ["RANK"])     # 全局 rank

        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        
        # 确保快照目录存在 (仅主进程创建)
        if self.global_rank == 0 and not os.path.exists(os.path.dirname(snapshot_path)):
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        dist.barrier() # 等待所有进程，确保目录已创建

        if os.path.exists(snapshot_path):
            print(f"GPU {self.local_rank}: Loading snapshot from {snapshot_path}")
            self._load_snapshot(snapshot_path)
        else:
            print(f"GPU {self.local_rank}: No snapshot found at {snapshot_path}, starting from scratch.")

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        # 在加载前，确保所有进程都看到文件（如果文件系统有延迟）
        dist.barrier() 
        try:
            snapshot = torch.load(snapshot_path, map_location=loc)
            # 确保模型结构匹配
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.epochs_run = snapshot["EPOCHS_RUN"]
            print(f"GPU {self.local_rank}: Resuming training from snapshot at Epoch {self.epochs_run}")
        except Exception as e:
            print(f"GPU {self.local_rank}: Failed to load snapshot {snapshot_path}. Error: {e}. Training from scratch.")
            self.epochs_run = 0 # 重置 epoch

    def _run_batch(self, source, targets, epoch, step):
        self.optimizer.zero_grad()
        output = self.model(source)
        # MNIST 目标是类别索引，直接用于 cross_entropy
        loss = F.cross_entropy(output, targets) 
        loss.backward()
        self.optimizer.step()

        # 仅在需要全局聚合的 loss 时进行 all_reduce (例如用于打印)
        # DDP 会自动处理梯度的 all-reduce
        reduced_loss = loss.clone() # 克隆以避免修改原始 loss
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
        
        if self.global_rank == 0 and step % 50 == 0: # 主进程打印，减少打印频率
            print(f"Epoch {epoch}, Step {step}, Loss {reduced_loss.item():.4f}")

    def _run_epoch(self, epoch):
        # b_sz = len(next(iter(self.train_data))[0]) # 这行在迭代器空的时候会报错
        # print(f"[GPU{self.local_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        print(f"[GPU{self.local_rank} | Global Rank {self.global_rank}] Epoch {epoch} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch) # 关键：确保 shuffle 在每个 epoch 不同
        for step, (source, targets) in enumerate(self.train_data):
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank) # MNIST target 是 LongTensor
            self._run_batch(source, targets, epoch, step)

    def _save_snapshot(self, epoch):
        # 确保只有主进程保存快照
        if self.global_rank == 0:
            snapshot = {
                "MODEL_STATE": self.model.module.state_dict(), # DDP 模型需要 .module
                "EPOCHS_RUN": epoch + 1, # 保存的是已完成的 epoch 数，下次从 epoch+1 开始
            }
            torch.save(snapshot, self.snapshot_path)
            print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path} by GPU {self.local_rank}")
        # 使用 barrier 确保所有进程在主进程完成保存前不会继续，避免潜在的竞争条件
        dist.barrier()


    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # 保存快照的逻辑
            if epoch % self.save_every == 0 or epoch == max_epochs -1 : # 在每个 save_every epoch 或最后一个 epoch 保存
                 self._save_snapshot(epoch)


def load_train_objs(data_root: str = "./data"):
    # --- MNIST 数据预处理 ---
    transform = transforms.Compose([
        transforms.ToTensor(), # 将 PIL 图像或 numpy.ndarray 转换为FloatTensor，并将像素强度范围从 [0, 255] 缩放到 [0.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 数据集的均值和标准差
    ])

    # --- 加载 MNIST 数据集 ---
    # download=True 如果数据集不存在，则会自动下载 (仅需一个进程执行下载)
    # 为了避免多个进程同时下载，通常会在这里加一个 rank == 0 的判断，或者让 torchvison 自己处理
    # torchvision datasets 应该能处理好并发下载的问题，但保险起见可以加 barrier
    
    # 推荐做法：让 rank 0 下载，其他 rank 等待
    # if int(os.environ.get("RANK", "0")) == 0: # get("RANK", "0") 避免在非DDP环境报错
    #     train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    # dist.barrier() # 所有进程在此等待，直到 rank 0 完成下载
    # if int(os.environ.get("RANK", "0")) != 0:
    #     train_set = datasets.MNIST(root=data_root, train=True, download=False, transform=transform) # 其他进程不重复下载

    # 更简单的方式，datasets.MNIST 内部应该有对并发下载的处理
    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)

    model = MNIST_CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adam 通常对 CNN 效果不错
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9) # 或者 SGD
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False, # 注意：shuffle=False，因为 DistributedSampler 会处理 shuffle
        sampler=DistributedSampler(dataset, shuffle=True) # shuffle 在这里设置
    )


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "./snapshots/mnist_snapshot.pt"):
    ddp_setup()
    dataset, model, optimizer = load_train_objs(data_root="./data_mnist") # 指定 MNIST 数据保存路径
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch MNIST Distributed Training')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot (in epochs)')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each_device (default: 64)')
    parser.add_argument('--snapshot_path', default="./snapshots/mnist_snapshot.pt", type=str, help='Path to save/load snapshots')
    # DDP 会自动注入 LOCAL_RANK, RANK, WORLD_SIZE 等环境变量，这里不需要 argparse 处理
    args = parser.parse_args()
    
    # 创建快照目录（如果不存在）
    snapshot_dir = os.path.dirname(args.snapshot_path)
    if not os.path.exists(snapshot_dir) and int(os.environ.get("RANK", "0")) == 0 : # 主进程创建
        os.makedirs(snapshot_dir, exist_ok=True)

    main(args.save_every, args.total_epochs, args.batch_size, snapshot_path=args.snapshot_path)