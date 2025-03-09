import os
from os import path
import skimage as ski
from PIL import Image


if not os.path.exists("Noisy_testing_images"):
    print("Creating Noisy_testing_images directory...")
    os.mkdir("Noisy_testing_images")

for testing_Image in os.listdir("Testing_images"):
    if path.isfile(f"Noisy_testing_images/{testing_Image}.png"):
        print(f"Testing_images/{testing_Image}.png already exists")
        continue

    image = ski.io.imread(os.path.join("Testing_images", testing_Image))
    image = ski.util.random_noise(image, mode='gaussian', var=0.001)
    image = ski.util.img_as_ubyte(image)
    image = Image.fromarray(image).convert("L")
    image.save(f"Noisy_testing_images/{testing_Image}.png")
    print(f"Noisy Testing Image {testing_Image} saved")

if not os.path.exists("Noisy_training_images"):
    print("Creating Noisy_training_images directory...")
    os.mkdir("Noisy_training_images")

for training_image in os.listdir("Training_images"):
    if path.isfile(f"Noisy_training_images/{training_image}.png"):
        print(f"Testing_images/{training_image}.png already exists")
        continue

    image = ski.io.imread(os.path.join("Training_images", training_image))
    image = ski.util.random_noise(image, mode='gaussian', var=0.001)
    image = ski.util.img_as_ubyte(image)
    image = Image.fromarray(image).convert("L")
    image.save(f"Noisy_training_images/{training_image}.png")
    print(f"Noisy Training Image {training_image} saved")