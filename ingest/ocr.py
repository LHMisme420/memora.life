# ingest/ocr.py
import pytesseract
from PIL import Image

def ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img)
