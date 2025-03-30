import os
from os import path
import skimage as ski
from PIL import Image

testImagesPath = "Test Images"
noisyImagesPath = "Noisy Test Images"


for testImage in os.listdir(testImagesPath):
    #Check if noisy image already exists in the noisy images directory
    if path.isfile(f"{noisyImagesPath}/{testImage}"):
        print(f"Noisy Testing Images/{testImage}.jpg already exists")
        continue
    #Read the image data using Skimage
    image = ski.io.imread(os.path.join(testImagesPath, testImage))
    #Add noise to the image
    image = ski.util.random_noise(image, mode='gaussian', var=0.001)
    #Convert the image to ubyte to allow PIL to read the data
    image = ski.util.img_as_ubyte(image)
    #Cover the numpy array to a PIL object to allow it to be saved
    image = Image.fromarray(image)
    #Save the noisy image to the noisy images directory
    image.save(f"{noisyImagesPath}/{testImage}")
    print(f"Noisy Testing Image {testImage} saved")