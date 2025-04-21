import tensorflow as tf
import numpy as np
from PIL import Image
import skimage as ski
import jiwer as jw
from PIL.Image import Resampling
import pytesseract
import matplotlib.pyplot as plt

#Load model
model = tf.keras.models.load_model('../Model/Receipts_even_more+invoicesandreciptsv2400x400px_8layers_150epochs_shuffle_200_100filters_learning-rate=0.01_0.01_noise.keras')

#Paths for images
image1Path = "Test Images/1000-receipt.jpg"
image2Path = "Test Images/1005-receipt.jpg"
image3Path = "Test Images/1010-receipt.jpg"
image4Path = "Test Images/1060-receipt.jpg"
image5Path = "Test Images/1122-receipt.jpg"

noisyImage1Path = "Noisy Test Images/1000-receipt.jpg"
noisyImage2Path = "Noisy Test Images/1005-receipt.jpg"
noisyImage3Path = "Noisy Test Images/1010-receipt.jpg"
noisyImage4Path = "Noisy Test Images/1060-receipt.jpg"
noisyImage5Path = "Noisy Test Images/1122-receipt.jpg"

#Paths for ground truth text files
image1GroundTruthPath = "Ground Truths/1000-receipt-ground-truth.txt"
image2GroundTruthPath = "Ground Truths/1005-receipt-ground-truth.txt"
image3GroundTruthPath = "Ground Truths/1010-receipt-ground-truth.txt"
image4GroundTruthPath = "Ground Truths/1060-receipt-ground-truth.txt"
image5GroundTruthPath = "Ground Truths/1122-receipt-ground-truth.txt"

#Load testing images

#Image 1

#Load original testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
image1 = Image.open(image1Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
image1 = np.array(image1)
#Convert to float32 and normalise pixel values
image1 = image1.astype('float32') / 255
#Reshape to 400x400
image1 = np.reshape(image1, (400, 400))

#Load noisy testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
noisy_image1 = Image.open(noisyImage1Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
noisy_image1 = np.array(noisy_image1)
#Convert to float32 and normalise pixel values
noisy_image1 = noisy_image1.astype('float32') / 255
#Reshape to 400x400
noisy_image1 = np.reshape(noisy_image1, (1, 400, 400, 1))

#Image 2

#Load original testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
image2 = Image.open(image2Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
image2 = np.array(image2)
#Convert to float32 and normalise pixel values
image2 = image2.astype('float32') / 255
#Reshape to 400x400
image2 = np.reshape(image2, (400, 400))

#Load noisy testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
noisy_image2 = Image.open(noisyImage2Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
noisy_image2 = np.array(noisy_image2)
#Convert to float32 and normalise pixel values
noisy_image2 = noisy_image2.astype('float32') / 255
#Reshape to 400x400
noisy_image2 = np.reshape(noisy_image2, (1, 400, 400, 1))

#Image 3

#Load original testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
image3 = Image.open(image3Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
image3 = np.array(image3)
#Convert to float32 and normalise pixel values
image3 = image3.astype('float32') / 255
#Reshape to 400x400
image3 = np.reshape(image3, (400, 400))

#Load noisy testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
noisy_image3 = Image.open(noisyImage3Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
noisy_image3 = np.array(noisy_image3)
#Convert to float32 and normalise pixel values
noisy_image3 = noisy_image3.astype('float32') / 255
#Reshape to 400x400
noisy_image3 = np.reshape(noisy_image3, (1, 400, 400, 1))

#Image 4

#Load original testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
image4 = Image.open(image4Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
image4 = np.array(image4)
#Convert to float32 and normalise pixel values
image4 = image4.astype('float32') / 255
#Reshape to 400x400
image4 = np.reshape(image4, (400, 400))

#Load noisy testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
noisy_image4 = Image.open(noisyImage4Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
noisy_image4 = np.array(noisy_image4)
#Convert to float32 and normalise pixel values
noisy_image4 = noisy_image4.astype('float32') / 255
#Reshape to 400x400
noisy_image4 = np.reshape(noisy_image4, (1, 400, 400, 1))

#Image 5

#Load original testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
image5 = Image.open(image5Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
image5 = np.array(image5)
#Convert to float32 and normalise pixel values
image5 = image5.astype('float32') / 255
#Reshape to 400x400
image5 = np.reshape(image5, (400, 400))

#Load noisy testing image, convert to greyscale, and resize to 400x400 pixels using the LANCZOS(Anti-aliasing) algorithm
noisy_image5 = Image.open(noisyImage5Path).convert('L').resize((400, 400), Resampling.LANCZOS)
#Create NumPy array
noisy_image5 = np.array(noisy_image5)
#Convert to float32 and normalise pixel values
noisy_image5 = noisy_image5.astype('float32') / 255
#Reshape to 400x400
noisy_image5 = np.reshape(noisy_image5, (1, 400, 400, 1))

#Reconstruct noisy images with CAE model

#Image 1
reconstructed_image1 = model.predict(noisy_image1)
#Image 2
reconstructed_image2 = model.predict(noisy_image2)
#Image 3
reconstructed_image3 = model.predict(noisy_image3)
#Image 4
reconstructed_image4 = model.predict(noisy_image4)
#Image 5
reconstructed_image5 = model.predict(noisy_image5)

#Reshape noisy images and the reconstructed images to allow metrics to be calculated

#Image 1
noisy_image1 = np.reshape(noisy_image1, (400, 400))
reconstructed_image1 = np.reshape(reconstructed_image1, (400, 400))

#Image 2
noisy_image2 = np.reshape(noisy_image2, (400, 400))
reconstructed_image2 = np.reshape(reconstructed_image2, (400, 400))

#Image 3
noisy_image3 = np.reshape(noisy_image3, (400, 400))
reconstructed_image3 = np.reshape(reconstructed_image3, (400, 400))

#Image 4
noisy_image4 = np.reshape(noisy_image4, (400, 400))
reconstructed_image4 = np.reshape(reconstructed_image4, (400, 400))

#Image 5
noisy_image5 = np.reshape(noisy_image5, (400, 400))
reconstructed_image5 = np.reshape(reconstructed_image5, (400, 400))

#Calculate metrics for each image

#Image 1 Metrics
noisy_psnr1 = ski.metrics.peak_signal_noise_ratio(image1, noisy_image1 )

noisy_ssim1 = ski.metrics.structural_similarity(image1, noisy_image1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

reconstructed_psnr1 = ski.metrics.peak_signal_noise_ratio(image1, reconstructed_image1 )

reconstructed_ssim1 = ski.metrics.structural_similarity(image1, reconstructed_image1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

#Image 2 Metrics
noisy_psnr2 = ski.metrics.peak_signal_noise_ratio(image2, noisy_image2 )

noisy_ssim2 = ski.metrics.structural_similarity(image2, noisy_image2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

reconstructed_psnr2 = ski.metrics.peak_signal_noise_ratio(image2, reconstructed_image2 )

reconstructed_ssim2 = ski.metrics.structural_similarity(image2, reconstructed_image2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

#Image 3 Metrics
noisy_psnr3 = ski.metrics.peak_signal_noise_ratio(image3, noisy_image3 )

noisy_ssim3 = ski.metrics.structural_similarity(image3, noisy_image3, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

reconstructed_psnr3 = ski.metrics.peak_signal_noise_ratio(image3, reconstructed_image3 )

reconstructed_ssim3 = ski.metrics.structural_similarity(image3, reconstructed_image3, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

#Image 4 Metrics
noisy_psnr4 = ski.metrics.peak_signal_noise_ratio(image4, noisy_image4 )

noisy_ssim4 = ski.metrics.structural_similarity(image4, noisy_image4, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

reconstructed_psnr4 = ski.metrics.peak_signal_noise_ratio(image4, reconstructed_image4 )

reconstructed_ssim4 = ski.metrics.structural_similarity(image4, reconstructed_image4, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

#Image 3 Metrics
noisy_psnr5 = ski.metrics.peak_signal_noise_ratio(image5, noisy_image5 )

noisy_ssim5 = ski.metrics.structural_similarity(image5, noisy_image5, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

reconstructed_psnr5 = ski.metrics.peak_signal_noise_ratio(image5, reconstructed_image5 )

reconstructed_ssim5 = ski.metrics.structural_similarity(image5, reconstructed_image5, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

#Display Image 1 Metrics

print("***************")
print("Image metrics for image 1")
print("PSNR for noisy image: ", noisy_psnr1)
print("SSIM for noisy image: ", noisy_ssim1)
print("PSNR for reconstructed image: ", reconstructed_psnr1)
print("SSIM for reconstructed image: ", reconstructed_ssim1)
print("*************** \n")

#Display Image 2 Metrics

print("***************")
print("Image metrics for image 2")
print("PSNR for noisy image: ", noisy_psnr2)
print("SSIM for noisy image: ", noisy_ssim2)
print("PSNR for reconstructed image: ", reconstructed_psnr2)
print("SSIM for reconstructed image: ", reconstructed_ssim2)
print("*************** \n")

#Display Image 3 Metrics

print("***************")
print("Image metrics for image 3")
print("PSNR for noisy image: ", noisy_psnr3)
print("SSIM for noisy image: ", noisy_ssim3)
print("PSNR for reconstructed image: ", reconstructed_psnr3)
print("SSIM for reconstructed image: ", reconstructed_ssim3)
print("*************** \n")

#Display Image 4 Metrics

print("***************")
print("Image metrics for image 4")
print("PSNR for noisy image: ", noisy_psnr4)
print("SSIM for noisy image: ", noisy_ssim4)
print("PSNR for reconstructed image: ", reconstructed_psnr4)
print("SSIM for reconstructed image: ", reconstructed_ssim4)
print("*************** \n")

#Display Image 5 Metrics

print("***************")
print("Image metrics for image 5")
print("PSNR for noisy image: ", noisy_psnr5)
print("SSIM for noisy image: ", noisy_ssim5)
print("PSNR for reconstructed image: ", reconstructed_psnr5)
print("SSIM for reconstructed image: ", reconstructed_ssim5)
print("*************** \n")

############
##Image 1##
###########

#Multiply pixel values and convert to uint8 to allow pyTesseract to extract text and matplotlib to display the noisy and reconstructed images
noisy_image1 = noisy_image1 * 255
noisy_image1 = noisy_image1.astype('uint8')

reconstructed_image1 = reconstructed_image1 * 255
reconstructed_image1 = reconstructed_image1.astype('uint8')

#Display both the noisy and reconstructed images
plt.figure(figsize=(10,10))

plt.subplot(1,2,1).set_title("Noisy Image 1")
plt.imshow(noisy_image1, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2).set_title("Reconstructed Image 1")
plt.imshow(reconstructed_image1, cmap='gray')
plt.axis('off')

plt.show()

#Extract text from noisy image
noisy_prediction1 = pytesseract.image_to_string(noisy_image1)
#Extract text from reconstructed image
reconstructed_prediction1 = pytesseract.image_to_string(reconstructed_image1)
#Read ground truth from text file
image1Text = open(image1GroundTruthPath,"r").read()

#Check if the reconstructed image has an output from pyTesseract
if not(reconstructed_prediction1 == ""):
    #If it does then display metrics
    print("***************")
    print("OCR output metrics for noisy image 1")
    noisy_wer1 = jw.wer(image1Text, noisy_prediction1)
    noisy_cer1 = jw.cer(image1Text, noisy_prediction1)
    print("WER for noisy image: ", noisy_wer1)
    print("CER for noisy image: ", noisy_cer1)
    print("OCR Output for noisy image: ", noisy_prediction1)
    reconstructed_wer1 = jw.wer(image1Text, reconstructed_prediction1)
    reconstructed_cer1 = jw.cer(image1Text, reconstructed_prediction1)
    print("OCR output metrics for reconstructed image 1")
    print("WER for reconstructed image: ", reconstructed_wer1)
    print("CER for reconstructed image: ", reconstructed_cer1)
    print("OCR Output for reconstructed image: ", reconstructed_prediction1)
    print("*************** \n")
else:
    #Else display that there is no output
    print("***************")
    print("OCR output metrics for image 1")
    print("There is no output")
    print("*************** \n")

############
##Image 2##
###########

#Multiply pixel values and convert to uint8 to allow pyTesseract to extract text and matplotlib to display the noisy and reconstructed images
noisy_image2 = noisy_image2 * 255
noisy_image2 = noisy_image2.astype('uint8')

reconstructed_image2 = reconstructed_image2 * 255
reconstructed_image2 = reconstructed_image2.astype('uint8')

#Display both the noisy and reconstructed images
plt.figure(figsize=(10,10))

plt.subplot(1,2,1).set_title("Noisy Image 2")
plt.imshow(noisy_image2, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2).set_title("Reconstructed Image 2")
plt.imshow(reconstructed_image2, cmap='gray')
plt.axis('off')

plt.show()

#Extract text from noisy image
noisy_prediction2 = pytesseract.image_to_string(noisy_image2)
#Extract text from reconstructed image
reconstructed_prediction2 = pytesseract.image_to_string(reconstructed_image2)
#Read ground truth from text file
image2Text = open(image2GroundTruthPath,"r").read()

#Check if the reconstructed image has an output from pyTesseract
if not(reconstructed_prediction2 == ""):
    #If it does then display metrics
    print("***************")
    print("OCR output metrics for noisy image 2")
    noisy_wer2 = jw.wer(image2Text, noisy_prediction2)
    noisy_cer2 = jw.cer(image2Text, noisy_prediction2)
    print("WER for noisy image: ", noisy_wer2)
    print("CER for noisy image: ", noisy_cer2)
    print("OCR Output for noisy image: ", noisy_prediction2)
    reconstructed_wer2 = jw.wer(image2Text, reconstructed_prediction2)
    reconstructed_cer2 = jw.cer(image2Text, reconstructed_prediction2)
    print("OCR output metrics for reconstructed image 2")
    print("WER for reconstructed image: ", reconstructed_wer2)
    print("CER for reconstructed image: ", reconstructed_cer2)
    print("OCR Output for reconstructed image: ", reconstructed_prediction2)
    print("*************** \n")
else:
    #Else display that there is no output
    print("***************")
    print("OCR output metrics for image 2")
    print("There is no output")
    print("*************** \n")

############
##Image 3##
###########

#Multiply pixel values and convert to uint8 to allow pyTesseract to extract text and matplotlib to display the noisy and reconstructed images
noisy_image3 = noisy_image3 * 255
noisy_image3 = noisy_image3.astype('uint8')

reconstructed_image3 = reconstructed_image3 * 255
reconstructed_image3 = reconstructed_image3.astype('uint8')

#Display both the noisy and reconstructed images
plt.figure(figsize=(10,10))

plt.subplot(1,2,1).set_title("Noisy Image 3")
plt.imshow(noisy_image3, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2).set_title("Reconstructed Image 3")
plt.imshow(reconstructed_image3, cmap='gray')
plt.axis('off')

plt.show()

#Extract text from noisy image
noisy_prediction3 = pytesseract.image_to_string(noisy_image3)
#Extract text from reconstructed image
reconstructed_prediction3 = pytesseract.image_to_string(reconstructed_image3)
#Read ground truth from text file
image3Text = open(image3GroundTruthPath,"r").read()

#Check if the reconstructed image has an output from pyTesseract
if not(reconstructed_prediction3 == ""):
    #If it does then display metrics
    print("***************")
    print("OCR output metrics for noisy image 3")
    noisy_wer3 = jw.wer(image3Text, noisy_prediction3)
    noisy_cer3 = jw.cer(image3Text, noisy_prediction3)
    print("WER for noisy image: ", noisy_wer3)
    print("CER for noisy image: ", noisy_cer3)
    print("OCR Output for noisy image: ", noisy_prediction3)
    reconstructed_wer3 = jw.wer(image3Text, reconstructed_prediction3)
    reconstructed_cer3 = jw.cer(image3Text, reconstructed_prediction3)
    print("OCR output metrics for reconstructed image 3")
    print("WER for reconstructed image: ", reconstructed_wer3)
    print("CER for reconstructed image: ", reconstructed_cer3)
    print("OCR Output for reconstructed image: ", reconstructed_prediction3)
    print("*************** \n")
else:
    #Else display that there is no output
    print("***************")
    print("OCR output metrics for image 3")
    print("There is no output")
    print("*************** \n")

############
##Image 4##
###########

#Multiply pixel values and convert to uint8 to allow pyTesseract to extract text and matplotlib to display the noisy and reconstructed images
noisy_image4 = noisy_image4 * 255
noisy_image4 = noisy_image4.astype('uint8')

reconstructed_image4 = reconstructed_image4 * 255
reconstructed_image4 = reconstructed_image4.astype('uint8')

#Display both the noisy and reconstructed images
plt.figure(figsize=(10,10))

plt.subplot(1,2,1).set_title("Noisy Image 4")
plt.imshow(noisy_image4, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2).set_title("Reconstructed Image 4")
plt.imshow(reconstructed_image4, cmap='gray')
plt.axis('off')

plt.show()

#Extract text from noisy image
noisy_prediction4 = pytesseract.image_to_string(noisy_image4)
#Extract text from reconstructed image
reconstructed_prediction4 = pytesseract.image_to_string(reconstructed_image4)
#Read ground truth from text file
image4Text = open(image4GroundTruthPath,"r").read()

#Check if the reconstructed image has an output from pyTesseract
if not(reconstructed_prediction4 == ""):
    #If it does then display metrics
    print("***************")
    print("OCR output metrics for noisy image 4")
    noisy_wer4 = jw.wer(image4Text, noisy_prediction4)
    noisy_cer4 = jw.cer(image4Text, noisy_prediction4)
    print("WER for noisy image: ", noisy_wer4)
    print("CER for noisy image: ", noisy_cer4)
    print("OCR Output for noisy image: ", noisy_prediction4)
    reconstructed_wer4 = jw.wer(image4Text, reconstructed_prediction4)
    reconstructed_cer4 = jw.cer(image4Text, reconstructed_prediction4)
    print("OCR output metrics for reconstructed image 4")
    print("WER for reconstructed image: ", reconstructed_wer4)
    print("CER for reconstructed image: ", reconstructed_cer4)
    print("OCR Output for reconstructed image: ", reconstructed_prediction4)
    print("*************** \n")
else:
    #Else display that there is no output
    print("***************")
    print("OCR output metrics for image 4")
    print("There is no output")
    print("*************** \n")

############
##Image 5##
###########

#Multiply pixel values and convert to uint8 to allow pyTesseract to extract text and matplotlib to display the noisy and reconstructed images
noisy_image5 = noisy_image5 * 255
noisy_image5 = noisy_image5.astype('uint8')

reconstructed_image5 = reconstructed_image5 * 255
reconstructed_image5 = reconstructed_image5.astype('uint8')

#Display both the noisy and reconstructed images
plt.figure(figsize=(10,10))

plt.subplot(1,2,1).set_title("Noisy Image 5")
plt.imshow(noisy_image5, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2).set_title("Reconstructed Image 5")
plt.imshow(reconstructed_image5, cmap='gray')
plt.axis('off')

plt.show()

#Extract text from noisy image
noisy_prediction5 = pytesseract.image_to_string(noisy_image5)
#Extract text from reconstructed image
reconstructed_prediction5 = pytesseract.image_to_string(reconstructed_image5)
#Read ground truth from text file
image5Text = open(image5GroundTruthPath,"r").read()

#Check if the reconstructed image has an output from pyTesseract
if not(reconstructed_prediction4 == ""):
    #If it does then display metrics
    print("***************")
    print("OCR output metrics for noisy image 5")
    noisy_wer5 = jw.wer(image5Text, noisy_prediction5)
    noisy_cer5 = jw.cer(image5Text, noisy_prediction5)
    print("WER for noisy image: ", noisy_wer5)
    print("CER for noisy image: ", noisy_cer5)
    print("OCR Output for noisy image: ", noisy_prediction5)
    reconstructed_wer5 = jw.wer(image5Text, reconstructed_prediction5)
    reconstructed_cer5 = jw.cer(image5Text, reconstructed_prediction5)
    print("OCR output metrics for reconstructed image 5")
    print("WER for reconstructed image: ", reconstructed_wer5)
    print("CER for reconstructed image: ", reconstructed_cer5)
    print("OCR Output for reconstructed image: ", reconstructed_prediction5)
    print("*************** \n")
else:
    #Else display that there is no output
    print("***************")
    print("OCR output metrics for image 5")
    print("There is no output")
    print("*************** \n")

