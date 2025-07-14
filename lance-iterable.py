import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import decode_image

import os
import io
import cv2
import wandb
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from lance.torch.data import (
    LanceDataset,
    
    ShardedBatchSampler,
    ShardedFragmentSampler,
    FullScanSampler
)
from modelling.get_model_and_loss import get_model_and_loss

_food101_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


ctx = torch.multiprocessing.get_context("spawn")


def decode_tensor_image(batch, **kwargs):
    images = []
    labels = []
    for item in batch.to_pylist():
        img = Image.open(io.BytesIO(item["image"])).convert("RGB")
        img = _food101_transform(img)
        images.append(img)
        labels.append(item["label"])
    batch = {
        "image": torch.stack(images),
        "label": torch.tensor(labels, dtype=torch.long)
    }
    return batch


def get_dataset(dataset_path, batch_size, sampler=None):
    return LanceDataset(
        dataset_path,
        to_tensor_fn=decode_tensor_image,
        batch_size=batch_size,
        sampler=sampler
    )

def get_sampler(sampler_type, rank, world_size):
    if sampler_type == "sharded_batch":
        return ShardedBatchSampler(rank=rank, world_size=world_size)
    elif sampler_type == "sharded_fragment":
        return ShardedFragmentSampler(rank=rank, world_size=world_size, pad=True)
    elif sampler_type == "full_scan":
        return FullScanSampler()
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")

def get_loader(dataset, num_workers):
    return DataLoader(dataset, num_workers=num_workers, batch_size=None)

def train(rank, local_rank, world_size, args):
    is_distributed = not getattr(args, "no_ddp", False)
    if is_distributed:
        args.num_workers = 0
    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        print("Warning: CUDA not available. Running on CPU. This will be slow.")
        device = torch.device("cpu")
    sampler = get_sampler(args.sampler_type, rank, world_size)
    dataset = get_dataset(args.dataset_path, args.batch_size, sampler=sampler)
    loader = get_loader(dataset, args.num_workers)
    print(f"Rank {rank} (GPU {local_rank}) initialized with {sampler} sampler")

    model, loss_fn, eval_fn = get_model_and_loss(args.task_type, args.num_classes)
    model = model.to(device)
    if is_distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if rank == 0 and not args.no_wandb:
        wandb.init(project="lance-dist-training", config=vars(args), name=f"{'DDP' if world_size>1 else 'single'}-{type(sampler).__name__}-lance")
    total_start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        batch_iter = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        for batch in batch_iter:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
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
            print(f"[Epoch {epoch}] Loss: {total_loss:.4f}, Epoch Time: {epoch_time:.2f}s" + (f", Val Acc: {log_dict['val_acc']:.4f}" if 'val_acc' in log_dict else ""))
            if not args.no_wandb:
                wandb.log(log_dict)
    total_time = time.time() - total_start_time
    if rank == 0:
        val_acc = eval_fn(model, loader, device) if rank == 0 else None
        log_dict["val_acc"] = val_acc
        print(f"Total training time: {total_time:.2f} seconds")
        if not args.no_wandb:
            wandb.log(log_dict)
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/FOOD101.lance")
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--num_classes", type=int, default=101)
    parser.add_argument("--sampler_type", type=str, default="sharded_batch")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_ddp", action="store_true", help="Run in non-distributed (debug) mode")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    if args.no_ddp:
        # Run in single-process mode
        train(rank=0, local_rank=0, world_size=1, args=args)
    else:
        # This is the standard torchrun setup
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train(rank, local_rank, world_size, args)