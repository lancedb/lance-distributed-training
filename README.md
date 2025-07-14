# Distributed training with Lance

## Examples
### classification
1. Create lance dataset of FOOD101:
```
python create_datasets/classficiation.py
```
2. train using map-style dataset:
```
torchrun --nproc-per-node=2  lance_map_style.py --batch_size 128
```

3. train using iterable dataset:
```
torchrun --nproc-per-node=2  lance_iterable.py --batch_size 128
```

## Docs

There are 2 ways to load data for training models using Lance’s pytorch integration. 

1. Iterable style dataset (`LanceDataset`) - Suitable for streaming. Works with inbuilt distributed samplers
2. Map-style dataset (`SafeLanceDataset`) - Suitable for

A key difference in working with both is that:

- In the **iterable-style** (`LanceDataset`), the data transformation (decoding bytes, applying transforms, stacking) **must happen before the `DataLoader` receives the data**. This is done inside the `to_tensor_fn` (`decode_tensor_image` ).

If your dataset contains a binray feild, it can't be converted to tensor directly, so you need to handle it appropriately in a custom
`to_tensor_fn`. This is similar to `collate_fn` when using map-style dataset
<details>
<summary>Example: Decoding images from LanceDatase using a custom `to_tensor_fn` </summary>

```python
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

  ds = LanceDataset(
        dataset_path,
        to_tensor_fn=decode_tensor_image,
        batch_size=batch_size,
        sampler=sampler
    )
```

</details>

In the **map-style** (`SafeLanceDataset`), the `DataLoader`'s workers fetch the raw data, and the transformation happens later in the `collate_fn`

<details>
<summary>Example: Decoding images from SafeLanceDataset using `collate_fn` </summary>

```python
from lance.torch.data import SafeLanceDataset, get_safe_loader

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
```
</details>


## When to use Map-style or Iterable dataset
These are some rules of thumb to decide it you should use Map-style or Iterable dataset

### When to Use Map-Style(`torch.utils.data.Dataset` \ `SafeLanceDataset`):

**Standard Datasets (Default Choice)**: Use this for any dataset where you have a finite collection of data points that can be indexed. This covers almost all standard use cases like image classification (ImageNet, CIFAR) or text classification where each file/line is a sample.

When You Need High Performance: This is the only way to get the full performance benefit of PyTorch's DataLoader with num_workers > 0. The DataLoader can give each worker a list of indices to fetch in parallel, which is extremely efficient.
In short this should be your default choice unless you have a specific reason to use an iterable dataset.

### Iterable-Style Dataset (`torch.utils.data.IterableDataset` \ `LanceDataset` )
This type of dataset works like a Python generator or a data stream.

Core Idea: It only knows how to give you the next item when you iterate over it (__iter__). It often has no known length and you cannot ask for the N-th item directly.

** When to Use Iterable-Style: **
When your data is coming from a source that is a stream and cannot be easily indexed, like a database cursor, or a log that is being written to in real-time.

For Extremely Large Datasets: When your dataset is so massive that even creating an index of all the file paths would be too slow or consume too much memory.

For Custom, Complex Sampling: When you need to implement your own highly specialized sampling or batching logic that PyTorch's standard samplers cannot handle. This is exactly why LanceDataset is iterable. It uses its own custom, highly optimized ShardedBatchSampler or ShardedFragmentSampler that understands the internal structure of .lance files, which a generic PyTorch sampler would not.

## Lance's sampling guide

- `FullScanSampler`: **Not DDP-aware**. Intentionally designed for each process to scan the entire dataset. It is useful for single-GPU evaluation or when you need every process to see all the data for some reason (which is rare in ddp training).
- `ShardedBatchSampler`: **DDP-aware**. Splits the total set of **batches** evenly among GPUs. Provides perfect workload balance.
- `ShardedFragmentSampler`: **DDP-aware**. Splits the list of **fragments** among GPUs. Can result in an unbalanced workload if fragments have uneven sizes. This needs to be handled to prevent synchornization errors,

## Full Scan Sampler

This is the simplest sampler. It inherits from `FragmentSampler` and implements the `iter_fragments` method. Its implementation is a single loop that gets all fragments from the dataset and yields each one sequentially.

### Behavior in DDP

The `FullScanSampler` is **not DDP-aware**. It contains no logic that checks for a `rank` or `world_size`. Consequently, when used in a distributed setting, **every single process (GPU) will scan the entire dataset**.

- **Use Case:** This sampler is primarily intended for single-GPU scenarios, such as validation, inference, or debugging, where you need one process to read all the data. It is not suitable for distributed training.

## ShardedFragmentSampler

This sampler also inherits from `FragmentSampler` and works by dividing the **list of fragments** among the available processes. Its `iter_fragments` method gets the full list of fragments and then yields only the ones corresponding to its assigned `rank`.

- **Rank 0** gets fragments 0, 2, 4, ...
- **Rank 1** gets fragments 1, 3, 5, ...

and so on

### Behavior in DDP

This sampler is **DDP-aware**, but it operates at the fragment level.

- **Pros:** It can be very I/O efficient. Since each process is assigned whole fragments, it can read them in long, sequential blocks. The documentation notes this is more efficient for large datasets.
- **Cons:** It can lead to **workload imbalance**. Lance datasets can have fragments of varying sizes (e.g., the last fragment is often smaller). If one rank is assigned fragments that have more total rows than another rank, it will have more batches to process. This imbalance can lead to DDP deadlocks if not handled with padding.


can lead to **workload imbalance, and eventually error out.** 

<details>
<summary>Example: DDP error due to imbalanced fragment sampler </summary>

```python

Epoch 1/10: 300it [07:12,  1.44s/it, loss=1.07]  
[Epoch 0] Loss: 980.4352, Epoch Time: 432.61s
Epoch 2/10: 133it [03:17,  1.48s/it, loss=5.98]
Epoch 2/10: 300it [07:24,  1.48s/it, loss=2.49]
[Epoch 1] Loss: 1200.9648, Epoch Time: 444.51s
Epoch 3/10: 300it [07:22,  1.48s/it, loss=3.24]
[Epoch 2] Loss: 1324.9992, Epoch Time: 442.84s
Epoch 4/10: 300it [07:23,  1.48s/it, loss=3.69]
[Epoch 3] Loss: 1371.6891, Epoch Time: 443.10s
Epoch 5/10: 300it [07:23,  1.48s/it, loss=3.91]
[Epoch 4] Loss: 1384.9732, Epoch Time: 443.12s, Val Acc: 0.0196
Epoch 6/10: 300it [07:24,  1.48s/it, loss=3.94]
[Epoch 5] Loss: 1388.0216, Epoch Time: 444.14s
Epoch 7/10: 300it [07:24,  1.48s/it, loss=4]   
[Epoch 6] Loss: 1388.9526, Epoch Time: 444.02s
Epoch 8/10: 300it [07:24,  1.48s/it, loss=3.99]
[Epoch 7] Loss: 1388.8115, Epoch Time: 444.43s
Epoch 9/10: 300it [07:24,  1.48s/it, loss=2.29]
[Epoch 8] Loss: 1314.3089, Epoch Time: 444.65s
Epoch 9/10: 300it [07:24,  1.48s/it, loss=2.29]]
[Epoch 8] Loss: 1314.3089, Epoch Time: 444.65s
Epoch 10/10: 240it [05:55,  1.47s/it, loss=5.46][rank0]:[E709 17:05:38.162555850 ProcessGroupNCCL.cpp:632] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=20585, OpType=ALLREDUCE, NumelIn=1259621, NumelOut=1259621, Timeout(ms)=600000) ran for 600000 milliseconds before timing out.
[rank0]:[E709 17:05:38.162814866 ProcessGroupNCCL.cpp:2271] [PG ID 0 PG GUID 0(default_pg) Rank 0]  failure detected by watchdog at work sequence id: 20585 PG status: last enqueued work: 20589, last completed work: 20584
[rank0]:[E709 17:05:38.162832798 ProcessGroupNCCL.cpp:670] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[rank0]:[E709 17:05:38.162895613 ProcessGroupNCCL.cpp:2106] [PG ID 0 PG GUID 0(default_pg) Rank 0] First PG on this rank to signal dumping.
[rank0]:[E709 17:05:38.482119928 ProcessGroupNCCL.cpp:1746] [PG ID 0 PG GUID 0(default_pg) Rank 0] Received a dump signal due to a collective timeout from this local rank and we will try our best to dump the debug info. Last enqueued NCCL work: 20589, last completed NCCL work: 20584.This is most likely caused by incorrect usages of collectives, e.g., wrong sizes used across ranks, the order of collectives is not same for all ranks or the scheduled collective, for some reason, didn't run. Additionally, this can be caused by GIL deadlock or other reasons such as network errors or bugs in the communications library (e.g. NCCL), etc. 
[rank0]:[E709 17:05:38.482326987 ProcessGroupNCCL.cpp:1536] [PG ID 0 PG GUID 0(default_pg) Rank 0] ProcessGroupNCCL preparing to dump debug info. Include stack trace: 1
Epoch 10/10: 241it [15:55, 181.19s/it, loss=5.09][rank0]:[E709 17:05:39.081662161 ProcessGroupNCCL.cpp:684] [Rank 0] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank0]:[E709 17:05:39.081690629 ProcessGroupNCCL.cpp:698] [Rank 0] To avoid data inconsistency, we are taking the entire process down.
[rank0]:[E709 17:05:39.083402482 ProcessGroupNCCL.cpp:1899] [PG ID 0 PG GUID 0(default_pg) Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=20585, OpType=ALLREDUCE, NumelIn=1259621, NumelOut=1259621, Timeout(ms)=600000) ran for 600000 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:635 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f92e62535e8 in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x23d (0x7f92e756ea6d in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0xc80 (0x7f92e75707f0 in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7f92e7571efd in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0xd8198 (0x7f92d7559198 in /opt/conda/bin/../lib/libstdc++.so.6)
frame #5: <unknown function> + 0x7ea7 (0x7f933d48dea7 in /usr/lib/x86_64-linux-gnu/libpthread.so.0)
frame #6: clone + 0x3f (0x7f933d25eadf in /usr/lib/x86_64-linux-gnu/libc.so.6)

terminate called after throwing an instance of 'c10::DistBackendError'
  what():  [PG ID 0 PG GUID 0(default_pg) Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=20585, OpType=ALLREDUCE, NumelIn=1259621, NumelOut=1259621, Timeout(ms)=600000) ran for 600000 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:635 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f92e62535e8 in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x23d (0x7f92e756ea6d in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0xc80 (0x7f92e75707f0 in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7f92e7571efd in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0xd8198 (0x7f92d7559198 in /opt/conda/bin/../lib/libstdc++.so.6)
frame #5: <unknown function> + 0x7ea7 (0x7f933d48dea7 in /usr/lib/x86_64-linux-gnu/libpthread.so.0)
frame #6: clone + 0x3f (0x7f933d25eadf in /usr/lib/x86_64-linux-gnu/libc.so.6)

Exception raised from ncclCommWatchdog at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1905 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f92e62535e8 in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x11b4abe (0x7f92e7540abe in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #2: <unknown function> + 0xe07bed (0x7f92e7193bed in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0xd8198 (0x7f92d7559198 in /opt/conda/bin/../lib/libstdc++.so.6)
frame #4: <unknown function> + 0x7ea7 (0x7f933d48dea7 in /usr/lib/x86_64-linux-gnu/libpthread.so.0)
frame #5: clone + 0x3f (0x7f933d25eadf in /usr/lib/x86_64-linux-gnu/libc.so.6)

E0709 17:05:39.816000 56204 site-packages/torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: -6) local_rank: 0 (pid: 56213) of binary: /opt/conda/bin/python3.10
Traceback (most recent call last):
  File "/opt/conda/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/opt/conda/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/distributed/run.py", line 892, in main
    run(args)
  File "/opt/conda/lib/python3.10/site-packages/torch/distributed/run.py", line 883, in run
    elastic_launch(
  File "/opt/conda/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 139, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/opt/conda/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 270, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
trainer.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-07-09_17:05:39
  host      : distributed-training.us-central1-a.c.lance-dev-ayush.internal
  rank      : 0 (local_rank: 0)
  exitcode  : -6 (pid: 56213)
  error_file: <N/A>
  traceback : Signal 6 (SIGABRT) received by PID 56213
============================================================
(base) jupyter@distributed-training:~/lance-dist-training$ 
(base) jupyter@distributed-training:~/lance-dist-training$ python
```
</details>

## ShardedBatchSampler

This sampler provides perfectly balanced sharding by operating at the **batch level**, not the fragment level. Calculates row ranges for each batch and deals those ranges out to the different ranks.

This logic gives interleaved batches to each process:

- **Rank 0** gets row ranges for Batch 0, Batch 2, Batch 4, ...
- **Rank 1** gets row ranges for Batch 1, Batch 3, Batch 5, ...

### Behavior in DDP

This sampler is **DDP-aware** and is the safest choice for balanced distributed training.

- **Pros:** It guarantees that every process receives almost the exact same number of batches, preventing workload imbalance and DDP deadlocks.
- **Cons:** It can be slightly less I/O efficient than `ShardedFragmentSampler`. To construct a batch, it may need to perform a specific range read from a fragment, which can be less optimal than reading the entire fragment at once.


you cannot use the `lance` samplers (like `ShardedBatchSampler` or `ShardedFragmentSampler`) with a map-style dataset.

The two systems are fundamentally incompatible by design:

1. **Lance Samplers** are designed to work *inside* the iterable `LanceDataset`. They don't generate indices. Instead, they directly control how the `lance` file scanner reads and yields entire batches of data. They are tightly coupled to the `LanceDataset`'s streaming (`__iter__`) mechanism.
2. **PyTorch's `DistributedSampler`** works by generating a list of **indices** (e.g., `[10, 5, 22]`). The `DataLoader` then takes these indices and fetches each item individually from a map-style dataset using its `__getitem__` method (e.g., `dataset[10]`).

Because the `lance` samplers don't produce the indices that a map-style `DataLoader` needs, you cannot use them together. You have to choose one of the two paths:

- **Path A (Lance Control):** Use the iterable `LanceDataset` with a `lance` sampler. **Benefit:** Uses `lance`'s native, optimized sampling. **Limitation:** Must use `num_workers=0`.
- **Path B (PyTorch Control):** Use a map-style dataset (like the `LanceMapDataset` we built, or `torchvision`'s) with PyTorch's `DistributedSampler`. **Benefit:** Allows for high-performance parallel data loading with `num_workers > 0`. **Limitation:** Does not use `lance`'s specific sampling logic.


