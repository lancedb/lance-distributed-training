import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
import os
import wandb
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
import io
import lance

from lance.torch.data import SafeLanceDataset, get_safe_loader
from modelling.get_model_and_loss import get_model_and_loss

def collate_fn(batch_of_dicts):
    """
    Collates a list of dictionaries from SafeLanceDataset into a single batch.
    This function handles decoding the image bytes and applying transforms.
    """
    images = []
    labels = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for item in batch_of_dicts:
        image_bytes = item["image"]
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)
        labels.append(item["label"])
        
    return {
        "image": torch.stack(images),
        "label": torch.tensor(labels, dtype=torch.long)
    }

def train(rank, local_rank, world_size, args):
    is_distributed = not getattr(args, "no_ddp", False)
    if is_distributed:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")

    # Use the official SafeLanceDataset
    dataset = SafeLanceDataset(uri=args.dataset_path)
    
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader = get_safe_loader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    print(f"Rank {rank} (GPU {local_rank}) initialized with {len(loader)} batches per epoch (workers={args.num_workers}).")

    model, loss_fn, eval_fn = get_model_and_loss(args.task_type, args.num_classes)
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if rank == 0 and not args.no_wandb:
        wandb.init(project="lance-dist-training", config=vars(args), name=f"DDP-SafeLanceDataset-lance")

    total_start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        if is_distributed:
            sampler.set_epoch(epoch)
        
        total_loss = 0.0
        epoch_start_time = time.time()
        
        batch_iter = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        for batch in batch_iter:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if rank == 0:
                batch_iter.set_postfix(loss=loss.item())

        epoch_time = time.time() - epoch_start_time
        log_dict = {"epoch": epoch, "loss": total_loss / len(loader), "epoch_time": epoch_time}
        
        if (epoch + 1) % 5 == 0 and rank == 0:
            eval_loader = get_safe_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
            val_acc = eval_fn(model, eval_loader, device)
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
    parser.add_argument("--dataset_path", type=str, default="data/FOOD101.lance")
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--num_classes", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_ddp", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    if args.no_ddp:
        train(rank=0, local_rank=0, world_size=1, args=args)
    else:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train(rank, local_rank, world_size, args)
