import os
import pyarrow as pa
import lance
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import io
from torchvision.datasets import ImageFolder
from torchvision.datasets import Food101


def create_lance_from_classification_dataset(root_path="data/FOOD101", 
                                             output_path="data/FOOD101.lance", 
                                             dataset_name="FOOD101", 
                                             fragment_size=12500, 
                                             batch_size=1024):

    if dataset_name == "FOOD101":
        dataset = Food101(root=root_path, split="train", download=True)
    else:
        raise ValueError("Unsupported dataset")

    def record_batch_generator():
        images, labels = [], []
        for idx, (img, label) in enumerate(tqdm(dataset, desc="Creating Lance Dataset")):
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            
            images.append(image_bytes)
            labels.append(label)

            if (idx + 1) % batch_size == 0:
                table = pa.Table.from_arrays(
                    [pa.array(images, type=pa.binary()), pa.array(labels, type=pa.int64())],
                    names=["image", "label"]
                )
                yield from table.to_batches(max_chunksize=fragment_size)
                images, labels = [], []

        if images:
            table = pa.Table.from_arrays(
                [pa.array(images, type=pa.binary()), pa.array(labels, type=pa.int64())],
                names=["image", "label"]
            )
            yield from table.to_batches(max_chunksize=fragment_size)


    schema = pa.schema([
        ("image", pa.binary()),
        ("label", pa.int64())
    ])

    ds = lance.write_dataset(
        record_batch_generator(),
        schema=schema,
        uri=output_path,
        mode="overwrite",
        max_rows_per_file=fragment_size
    )

    print(f"Lance dataset written to {output_path} with fragment size {fragment_size}. total fragments {len(ds.get_fragments())}")

# Usage:
# For Food101, run with:
#   create_lance_from_classification_dataset(dataset_name="FOOD101", root_path="data/food101", output_path="data/food101.lance")

if __name__ == "__main__":
    create_lance_from_classification_dataset()