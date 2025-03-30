import tensorflow as tf
import numpy as np
from PIL import Image
import skimage as ski
import jiwer as jw
from PIL.Image import Resampling
import pytesseract
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('../Model/Receipts+invoicesandReceiptsv2400x400px_7layers_50epochs_shuffle_200_100filters_learning-rate=0.01_0.01_noise.keras')

image1 = Image.open('Test Images/1000-receipt.jpg').convert('L').resize((400, 400), Resampling.LANCZOS)
image1 = np.array(image1)
image1 = image1.astype('float32') / 255
image1 = np.reshape(image1, (400, 400))

noisy_image1 = Image.open('Noisy Test Images/1000-receipt.jpg').convert('L').resize((400, 400), Resampling.LANCZOS)
noisy_image1 = np.array(noisy_image1)
noisy_image1 = noisy_image1.astype('float32') / 255
noisy_image1 = np.reshape(noisy_image1, (1, 400, 400, 1))

reconstructed_image1 = model.predict(noisy_image1)

noisy_image1 = np.reshape(noisy_image1, (400, 400))
reconstructed_image1 = np.reshape(reconstructed_image1, (400, 400))

noisy_psnr = ski.metrics.peak_signal_noise_ratio(image1, noisy_image1 )

noisy_ssim = ski.metrics.structural_similarity(image1, noisy_image1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

reconstructed_psnr = ski.metrics.peak_signal_noise_ratio(image1, reconstructed_image1 )

reconstructed_ssim = ski.metrics.structural_similarity(image1, reconstructed_image1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

print("PSNR for noisy image: ", noisy_psnr)
print("SSIM for noisy image: ", noisy_ssim)
print("PSNR for reconstructed image: ", reconstructed_psnr)
print("SSIM for reconstructed image: ", reconstructed_ssim)

#Pixel values have to be multiplied else the conversion to uint8 produces a totally black image
noisy_image1 = noisy_image1 * 255
noisy_image1 = noisy_image1.astype('uint8')

reconstructed_image1 = reconstructed_image1 * 255
reconstructed_image1 = reconstructed_image1.astype('uint8')

# Used to debug image conversion
# plt.imshow(reconstructed_image1, cmap="grey")
# plt.show()
noisy_prediction1 = pytesseract.image_to_string(noisy_image1)
reconstructed_prediction1 = pytesseract.image_to_string(reconstructed_image1)
image1Text = open("Ground Truths/1000-receipt-ground-truth.txt","r").read()

if not(reconstructed_prediction1 == ""):
    noisy_wer = jw.wer(image1Text, noisy_prediction1)
    noisy_cer = jw.cer(image1Text, noisy_prediction1)
    print("WER for noisy image: ", noisy_wer)
    print("CER for noisy image: ", noisy_cer)
    print("OCR Output for noisy image: ", noisy_prediction1)
    reconstructed_wer = jw.wer(image1Text, reconstructed_prediction1)
    reconstructed_cer = jw.cer(image1Text, reconstructed_prediction1)
    print("WER for reconstructed image: ", reconstructed_wer)
    print("CER for reconstructed image: ", reconstructed_cer)
    print("OCR Output for reconstructed image: ", reconstructed_prediction1)
else:
    print("There is no output")
