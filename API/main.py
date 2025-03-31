from PIL.Image import Resampling
from fastapi import FastAPI, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image
import pytesseract

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
    return pytesseract.image_to_string(image)
