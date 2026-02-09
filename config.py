# OCR CONFIGURATION
IMG_SIZE = 32  # default resize for CNN, not used in Tesseract but can keep for future
LANG = 'nep'   # Nepali traineddata, or 'eng' for English
DEFAULT_OEM = 1
DEFAULT_PSM = 6


# KATAHO API AUTHENTICATION CREDENTIALS
username = "kataho_developer"
password = "Hello@world123"

#OR VISUALIZATION DIRECTORY
DEBUG_DIR = "QR_visualizations"

# Plate-specific OCR configuration
FIELD_OCR_CONFIG = {
    "KID_No": {"lang": "eng", "psm": 6, "oem": 3},
    "Plus_Code": {"lang": "eng", "psm": 6, "oem": 3},
    "Local_Address": {"lang": "hin", "psm": 11, "oem": 3},
    "City": {"lang": "hin", "psm": 11, "oem": 3},
    "Ward_Address": {"lang": "nep", "psm": 11, "oem": 3},
}

# TESSERACT OCR MODEL PATH
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path as needed

# API ENDPOINTS CONFIG
LOGIN_URL = "https://kataho.app/api/login"
DATA_URL = "https://kataho.app/api/plate-status-check"

# PLATE TEMPLATES / COORDINATES DIRECTORY
COORDINATES_OUTPUT_DIR = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\Plate_Templates\Templates"

# YOLO DETECTION MODEL PATH
YOLO_MODEL = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\models\plate_yolo.pt"

# TEMPELATE PATH
JSON_PATH = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\Plate_Templates\Templates\default_plate_template.json"

# PLATE SCAN AFTER DETECTION IN 4K (SNAPSHOT DIRECTORY)
OUTPUT_DIR = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\Scan_Images\Plate_Scans"
ROI_DIR = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\Scan_Images\ROI"

#LIVE MAIN OCR CONFIGURATION
LIVE_CAMERA = 0
HOLD_SECONDS = 5          # each round = 5 seconds
MAX_OCR_ROUNDS = 5        # total rounds
