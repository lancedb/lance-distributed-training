import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets, models
import os
import wandb
import time
from tqdm import tqdm
from itertools import islice

from modelling.get_model_and_loss import get_model_and_loss

class TorchvisionIterableDataset(IterableDataset):
    def __init__(self, root, split, batch_size, transform=None, download=False, shuffle=True):
        super().__init__()
        self.map_dataset = datasets.Food101(root=root, split=split, download=download, transform=transform)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        sampler = DistributedSampler(
            self.map_dataset,
            num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            shuffle=self.shuffle
        )
        sampler.set_epoch(self.epoch)
        
        iterator = iter(sampler)
        
        while True:
            batch_indices = list(islice(iterator, self.batch_size))
            if not batch_indices:
                break
            
            items = [self.map_dataset[i] for i in batch_indices]
            images, labels = zip(*items)
            
            yield {
                "image": torch.stack(images),
                "label": torch.tensor(labels, dtype=torch.long)
            }

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_dataloaders(args, rank, world_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if rank != 0 and not args.no_ddp:
        dist.barrier()
    
    _ = datasets.Food101(root=args.dataset_path, split='train', download=(rank == 0))
    
    if rank == 0 and not args.no_ddp:
        dist.barrier()

    train_dataset = TorchvisionIterableDataset(
        root=args.dataset_path,
        split='train',
        batch_size=args.batch_size,
        transform=transform,
        shuffle=True
    )
    
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=None)
    
    return train_loader, train_dataset


def train(rank, local_rank, world_size, args):
    is_distributed = not getattr(args, "no_ddp", False)
    if is_distributed:
        args.num_workers = 0
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")

    train_loader, train_dataset = get_dataloaders(args, rank, world_size)
    print(f"Rank {rank} (GPU {local_rank}) initialized with iterable-style torchvision dataset.")

    model, loss_fn, evaluate = get_model_and_loss("classification", args.num_classes)
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if rank == 0 and not args.no_wandb:
        wandb.init(project="lance-dist-training", config=vars(args), name=f"{'DDP' if world_size > 1 else 'single'}-iterable-torch")
    
    total_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        if is_distributed:
            train_dataset.set_epoch(epoch)

        total_loss = 0.0
        
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        for batch in batch_iter:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if rank == 0:
                batch_iter.set_postfix(loss=loss.item())
        
        epoch_time = time.time() - epoch_start_time
        log_dict = {"epoch": epoch, "loss": total_loss, "epoch_time": epoch_time}

        
        if rank == 0:
            print(f"[Epoch {epoch}] Loss: {log_dict['loss']:.4f}" + (f", Val Acc: {log_dict.get('val_acc', 0):.4f}" if 'val_acc' in log_dict else ""))
            if not args.no_wandb:
                wandb.log(log_dict)
                
    total_time = time.time() - total_start_time
    if rank == 0:
        eval_loader = DataLoader(train_dataset.map_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        val_acc = evaluate(model, eval_loader, device)
        log_dict["val_acc"] = val_acc
        print(f"Total training time: {total_time:.2f} seconds")
        if not args.no_wandb:
            wandb.log(log_dict)

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/FOOD101", help="Directory to download/load datasets")
    parser.add_argument("--num_classes", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_ddp", action="store_true", help="Run in non-distributed (debug) mode")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    if args.no_ddp:
        train(rank=0, local_rank=0, world_size=1, args=args)
    else:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train(rank, local_rank, world_size, args)
