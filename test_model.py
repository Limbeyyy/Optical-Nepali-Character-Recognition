import cv2
import pytesseract
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_image(img, lang=None, psm=None, clean_text=True):
    """
    Run Tesseract OCR on an image and return cleaned text.
    
    Parameters:
        img : str or np.array
            - Path to image OR numpy array (BGR/gray)
        lang : str
            - Language code for Tesseract
        psm : int
            - Page segmentation mode (Tesseract)
        clean_text : bool
            - Whether to apply text cleanup (remove symbols, prefixes, extra whitespace)
    
    Returns:
        str : OCR extracted text
    """
    # Load image if path provided
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            print("❌ Failed to read image from path")
            return ""
    
    # Convert to grayscale if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    
    # Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR
    text = pytesseract.image_to_string(
        thresh,
        lang=lang,
        config=f"--oem 1 --psm {psm}"
    ).strip()
    
    if not clean_text:
        return text
    
    # ----------------- CLEANING -----------------
    # Remove unwanted symbols and multiple spaces
    text = re.sub(r'[^\w\s\-\/\+\:\u0900-\u097F]', '', text)  # keep letters, numbers, nepali, basic symbols
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

img_path = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\OCR_Plate_Processor\Scanned_Plates\roi\1770358443_r5_City.png"
text = ocr_image(img_path, lang='hin', psm= 8)
print("OCR Result:", text)
