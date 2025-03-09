from PIL import Image
import pytesseract
from PIL.Image import Resampling

#normalImage = Image.open("Testing Pipeline/Test Images/1000-receipt.jpg")
#print(pytesseract.image_to_string(normalImage))

resizedImage = Image.open("Testing Pipeline/Test Images/1000-receipt.jpg").resize((400,400), Resampling.LANCZOS)
print(pytesseract.image_to_string(resizedImage))