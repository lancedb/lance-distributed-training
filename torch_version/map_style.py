import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets, models
import os
import wandb
import time
from tqdm import tqdm


def get_model_and_loss(num_classes):
    """
    Loads a pre-trained ResNet50 model and adapts its final layer.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn

def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    model_to_eval = model.module if isinstance(model, DDP) else model
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_to_eval(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
    return total_correct / total_samples


def get_dataloaders(args, rank, world_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if rank != 0 and not args.no_ddp:
        dist.barrier()
    
    train_dataset = datasets.Food101(root=args.dataset_path, split='train', transform=transform, download=(rank == 0))
    
    if rank == 0 and not args.no_ddp:
        dist.barrier()

    val_dataset = datasets.Food101(root=args.dataset_path, split='test', transform=transform, download=False)

    train_sampler = None
    if not args.no_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    ctx = torch.multiprocessing.get_context("spawn")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
        multiprocessing_context=ctx
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        multiprocessing_context=ctx
    )
    
    return train_loader, val_loader, train_sampler


def train(rank, local_rank, world_size, args):
    is_distributed = not getattr(args, "no_ddp", False)
    if is_distributed:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")

    train_loader, val_loader, train_sampler = get_dataloaders(args, rank, world_size)
    print(f"Rank {rank} (GPU {local_rank}) initialized with {len(train_loader)} batches per epoch (workers={args.num_workers}).")

    model, loss_fn = get_model_and_loss(args.num_classes)
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if rank == 0 and not args.no_wandb:
        wandb.init(project="lance-dist-training", config=vars(args), name=f"DDP-map-style-torchvision")
    
    total_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        if is_distributed:
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        for images, labels in batch_iter:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if rank == 0:
                batch_iter.set_postfix(loss=loss.item())
        
        epoch_time = time.time() - epoch_start_time
        log_dict = {"epoch": epoch, "loss": total_loss / len(train_loader), "epoch_time": epoch_time}

        if (epoch + 1) % 5 == 0 and rank == 0:
            val_acc = evaluate(model, val_loader, device)
            log_dict["val_acc"] = val_acc
        
        if rank == 0:
            print(f"[Epoch {epoch}] Loss: {log_dict['loss']:.4f}, Epoch Time: {epoch_time:.2f}s" + (f", Val Acc: {log_dict.get('val_acc', 0):.4f}" if 'val_acc' in log_dict else ""))
            if not args.no_wandb:
                wandb.log(log_dict)
                
    total_time = time.time() - total_start_time
    if rank == 0:
        print(f"Total training time: {total_time:.2f} seconds")
        if not args.no_wandb:
            wandb.log({"total_training_time": total_time})

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
