import io
import os
from os import path

import pandas as pd
from PIL import Image
from PIL.Image import Resampling

test_parquet = pd.read_parquet("Parquet/ReceiptsEvenMore/test-00000-of-00001.parquet")

train_parquet = pd.read_parquet("Parquet/ReceiptsEvenMore/train-00000-of-00001.parquet")

SIZE = (400,400)

if not os.path.exists("Testing_images"):
    print("Creating Testing_images directory...")
    os.mkdir("Testing_images")

for index, row in test_parquet.iterrows():
    if path.isfile(f"Testing_images/{index}.png"):
        print(f"Testing_images/{index}.png already exists")
        continue

    imageBytes = row["pixel_values"]
    imageStream = io.BytesIO(imageBytes.get("bytes"))
    image = Image.open(imageStream).convert('L').resize(SIZE)
    image.save(f"Testing_images/{index}.png")
    print(f"Testing Image {index} saved")

if not os.path.exists("Training_images"):
    print("Creating Training_images directory...")
    os.mkdir("Training_images")

for index, row in train_parquet.iterrows():
    if path.isfile(f"Training_images/{index}.png"):
        print(f"Training_images/{index}.png already exists")
        continue

    imageBytes = row["pixel_values"]
    imageStream = io.BytesIO(imageBytes.get("bytes"))
    image = Image.open(imageStream).convert('L').resize(SIZE)
    image.save(f"Training_images/{index}.png")
    print(f"Training Image {index} saved")