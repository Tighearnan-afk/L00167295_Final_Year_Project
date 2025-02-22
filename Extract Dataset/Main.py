import io
import os
from os import path

import pandas as pd
from PIL import Image
from PIL.Image import Resampling

test_parquet = pd.read_parquet("Extract Dataset/Parquet/test-00000-of-00001.parquet")

train_parquet = pd.read_parquet("Extract Dataset/Parquet/train-00000-of-00001.parquet")

SIZE = (900,900)

if not os.path.exists("Testing_images"):
    print("Creating Testing_images directory...")
    os.mkdir("Testing_images")

for index, row in test_parquet.iterrows():
    if path.isfile(f"Testing_images/{index}.jpg"):
        print(f"Testing_images/{index}.jpg already exists")
        continue

    imageBytes = row["pixel_values"]
    imageStream = io.BytesIO(imageBytes.get("bytes"))
    image = Image.open(imageStream).convert('L').resize(SIZE, Resampling.LANCZOS)
    image.save(f"Testing_images/{index}.jpg",quality=95)
    print(f"Testing Image {index} saved")

if not os.path.exists("Training_images"):
    print("Creating Training_images directory...")
    os.mkdir("Training_images")

for index, row in train_parquet.iterrows():
    if path.isfile(f"Training_images/{index}.jpg"):
        print(f"Training_images/{index}.jpg already exists")
        continue

    imageBytes = row["pixel_values"]
    imageStream = io.BytesIO(imageBytes.get("bytes"))
    image = Image.open(imageStream).convert('L').resize(SIZE, Resampling.LANCZOS)
    image.save(f"Training_images/{index}.jpg",quality=95)
    print(f"Training Image {index} saved")