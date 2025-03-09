import tensorflow as tf
import numpy as np
from PIL import Image
import skimage as ski
import jiwer as jw
from PIL.Image import Resampling
import pytesseract
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('../Model/Receipts400x400px_8layers_105epochs_shuffle_32_64filters.keras')

image1 = Image.open('Test Images/1000-receipt.jpg').convert('L').resize((400, 400), Resampling.LANCZOS)
image1 = np.array(image1)
image1 = image1.astype('float32') / 255
image1 = np.reshape(image1, (400, 400))

noisy_image1 = Image.open('Noisy Test Images/1000-receipt.jpg').convert('L').resize((400, 400), Resampling.LANCZOS)
noisy_image1 = np.array(noisy_image1)
noisy_image1 = noisy_image1.astype('float32') / 255
noisy_image1 = np.reshape(noisy_image1, (1, 400, 400, 1))

reconstructed_image1 = model.predict(noisy_image1)

reconstructed_image1 = np.reshape(reconstructed_image1, (400, 400))

psnr = ski.metrics.peak_signal_noise_ratio(image1, reconstructed_image1 )

ssim = ski.metrics.structural_similarity(image1, reconstructed_image1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

print(psnr)
print(ssim)

#Pixel values have to be multiplied else the conversion to uint8 produces a totally black image
reconstructed_image1 = reconstructed_image1 * 255
reconstructed_image1 = reconstructed_image1.astype('uint8')

plt.imshow(reconstructed_image1, cmap="grey")
plt.show()

prediction = pytesseract.image_to_string(reconstructed_image1)
text = "abc"

if prediction == "":
    prediction = "There is no output"

wer = jw.wer(prediction, text)
cer = jw.cer(prediction, text)

print(wer)
print(cer)
print(prediction)