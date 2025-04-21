from PIL.Image import Resampling
from fastapi import FastAPI, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image
import pytesseract
import tensorflow as tf
import numpy as np
import skimage as ski
import jiwer as jw
app = FastAPI()

#Index route that responds with a message verifying the API is running
@app.get("/")
def index():
    return "CAE-OCR API running"

#Upload route for receipts that returns a plain text response from the pyTesseract output
@app.post("/ocr/upload", response_class=PlainTextResponse)
async def upload_file(file: UploadFile):
    #Read the contents of the file using Pillow, extract the text using pyTesseract and return the output
    image = Image.open(file.file).resize((400, 400), Resampling.LANCZOS).convert('L')
    #Create NumPy array
    image = np.array(image)
    #Convert to float32 and normalise pixel values
    image = image.astype('float32') / 255
    #Reshape to 400x400 with a batch size of 1
    image = np.reshape(image, (1,400, 400))
    #Create noisy image with a variance of 0.001
    noisy_image = ski.util.random_noise(image, mode='gaussian', var=0.001)
    #Load model
    model = tf.keras.models.load_model('../Model/Receipts_even_more+invoicesandreciptsv2400x400px_8layers_150epochs_shuffle_200_100filters_learning-rate=0.01_0.01_noise.keras')
    #Reconstruct noisy image
    reconstructed_image = model.predict(noisy_image)
    #Reshape reconstructed image to 400x400, removing the batch size dimension(incompatible with pyTesseract)
    reconstructed_image = np.reshape(reconstructed_image, (400, 400))
    #Muliply pixel values by 255 (necessary for conversion to uint8)
    reconstructed_image = reconstructed_image * 255
    #Convert to uint8
    reconstructed_image = reconstructed_image.astype('uint8')
    #Extract text from reconstructed image and return to user
    return pytesseract.image_to_string(reconstructed_image)

#This is really messy and is only for demonstration purposes during the Viva presentation to showcase the metrics
@app.post("/ocr/uploadWithMetrics", response_class=PlainTextResponse)
async def upload_file(file: UploadFile):
    #Read the contents of the file using Pillow, extract the text using pyTesseract and return the output
    image = Image.open(file.file).resize((400, 400), Resampling.LANCZOS).convert('L')
    #Extract text from image to be used a ground truth text string for calculating WER and CER metrics
    groundTruth = pytesseract.image_to_string(image)
    #Create NumPy array
    image = np.array(image)
    #Convert to float32 and normalise pixel values
    image = image.astype('float32') / 255
    #Reshape to 400x400 with a batch size of 1
    image = np.reshape(image, (1,400, 400))
    #Create noisy image with a variance of 0.001
    noisy_image = ski.util.random_noise(image, mode='gaussian', var=0.001)
    #Load model
    model = tf.keras.models.load_model('../Model/Receipts_even_more+invoicesandreciptsv2400x400px_8layers_150epochs_shuffle_200_100filters_learning-rate=0.01_0.01_noise.keras')
    #Reconstruct noisy image
    reconstructed_image = model.predict(noisy_image)
    #Reshape reconstructed image to 400x400, removing the batch size dimension(incompatible with pyTesseract)
    reconstructed_image = np.reshape(reconstructed_image, (400, 400))
    #Muliply pixel values by 255 (necessary for conversion to uint8)
    reconstructed_image = reconstructed_image * 255
    #Convert to uint8
    reconstructed_image = reconstructed_image.astype('uint8')
    #Reshape reconstructed image to 400x400, removing the batch size dimension(incompatible with jiwer)
    image = np.reshape(image, (400, 400))
    #Muliply pixel values by 255 (necessary for conversion to uint8)
    image = image * 255
    #Convert to uint8
    image = image.astype('uint8')

    # Reshape reconstructed image to 400x400, removing the batch size dimension(incompatible with pyTesseract and jiwer)
    noisy_image = np.reshape(noisy_image, (400, 400))
    #Muliply pixel values by 255 (necessary for conversion to uint8)
    noisy_image = noisy_image * 255
    # Convert to uint8
    noisy_image = noisy_image.astype('uint8')

    #Extract text from noisy and reconstructed images
    noisy_image_ocr_output = pytesseract.image_to_string(noisy_image)
    reconstructed_ocr_output = pytesseract.image_to_string(reconstructed_image)

    #Calculate PSNR and SSIM metrics for both noisy and reconstructed images
    noisy_psnr = ski.metrics.peak_signal_noise_ratio(image, noisy_image)

    noisy_ssim = ski.metrics.structural_similarity(image, noisy_image, gaussian_weights=True, sigma=1.5,
                                                    use_sample_covariance=False, data_range=1.0)

    reconstructed_psnr = ski.metrics.peak_signal_noise_ratio(image, reconstructed_image)

    reconstructed_ssim = ski.metrics.structural_similarity(image, reconstructed_image, gaussian_weights=True, sigma=1.5,
                                                   use_sample_covariance=False, data_range=1.0)

    #Calculate the WER and CER values for both the noisy image and reconstructed image using the OCR output of the original, non-noisy image as a ground truth text string
    #Check if there is an output
    if not (noisy_image_ocr_output == ""):
        noisy_wer = jw.wer(groundTruth, noisy_image_ocr_output)
        noisy_cer = jw.cer(groundTruth, noisy_image_ocr_output)
        reconstructed_wer = jw.wer(groundTruth, reconstructed_ocr_output)
        reconstructed_cer = jw.cer(groundTruth, reconstructed_ocr_output)
    else:
    #If there is no output for the reconstructed model then set its OCR output string to N/A
        noisy_wer = jw.wer(groundTruth, noisy_image_ocr_output)
        noisy_cer = jw.cer(groundTruth, noisy_image_ocr_output)
        reconstructed_wer = "N/A"
        reconstructed_cer = "N/A"

    #Format metrics for legibility
    output = (f'**********\nNoisy Image PSNR: {noisy_psnr} \nNoisy Image SSIM: {noisy_ssim} \nNoisy Image WER: {noisy_wer} \nNoisy Image CER: {noisy_cer} \n**********\n\n'
              f'**********\nReconstructed Image PSNR: {reconstructed_psnr} \nReconstructed SSIM: {reconstructed_ssim} \nReconstructed WER: {reconstructed_wer} \nReconstructed CER: {reconstructed_cer} \n**********\n\n'
              f'**********\nNoisy Image OCR Output: {noisy_image_ocr_output} \n**********\nReconstructed Image OCR Output: {reconstructed_ocr_output}\n**********')
    return output
