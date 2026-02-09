# Optical Nepali OCR System

> **Real-time Nepali Address Plate Detection & OCR System with API Validation**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-green.svg)](https://github.com/tesseract-ocr/tesseract)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web-009688.svg)](https://fastapi.tiangolo.com/)

## 🎯 Overview

An advanced computer vision system for detecting and extracting information from Nepali Kataho address plates. Combines YOLOv8 object detection, multi-language Tesseract OCR (Nepali/Hindi/English), QR code reading, and backend API validation for high-accuracy results.

### Key Features

- ✅ **Real-time Detection**: 4K camera feed processing with YOLOv8
- ✅ **Multi-language OCR**: Support for Nepali, Hindi, and English scripts
- ✅ **QR Code Reading**: Robust multi-strategy QR detection
- ✅ **API Validation**: Progressive field verification with Kataho database
- ✅ **Web Interface**: FastAPI service for on-demand processing
- ✅ **Smart Post-processing**: Field-specific text cleaning and normalization

## 📚 Documentation

For complete technical documentation, architecture diagrams, and implementation details, see:
- **[DOCUMENTATION.md](./DOCUMENTATION.md)** - Comprehensive system documentation

## 🚀 Quick Start

### Prerequisites

1. **Install Tesseract OCR**:
   - Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install traineddata: `eng.traineddata`, `hin.traineddata`, `nep.traineddata`

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure System**:
   - Edit [`config.py`](./config.py) with your paths
   - Update `TESSERACT_PATH`
   - Set API credentials

### Run Live Camera OCR

```bash
python ocr_main_live.py
```

- Opens 4K camera feed with live plate detection
- Automatically captures and processes plates
- Validates results with Kataho API
- Progressive multi-round verification

### Run Web API

```bash
cd web/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access API at: `http://localhost:8000`

## 🏗️ Architecture

```
Input (Camera/Web) → YOLOv8 Detection → ROI Extraction 
→ Tesseract OCR + QR Reader → Post-processing 
→ API Validation → Verified Results
```

### Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Plate Detector** | Detect plate regions | YOLOv8 (Ultralytics) |
| **OCR Engine** | Extract text from fields | Tesseract (multi-lang) |
| **QR Processor** | Read QR codes | OpenCV + pyzbar |
| **Backend Validator** | Verify with API | Kataho REST API |
| **Web Service** | HTTP interface | FastAPI |

## 📁 Project Structure

```
Optical_Nepali_OCR/
├── config.py                    # Configuration
├── ocr_main_live.py             # Main live system
├── test_model.py                # Testing script
├── requirements.txt             # Dependencies
├── models/
│   └── plate_yolo.pt            # YOLOv8 model
├── Plate_Detector/              # Detection module
├── OCR_Engine/                  # OCR module
├── QR_Processor/                # QR reading module
├── Plate_Templates/             # ROI templates
├── web/                         # Web interface
└── Scan_Images/                 # Output directory
```

## 🔧 Configuration

Key settings in [`config.py`](./config.py):

```python
# OCR Configuration
FIELD_OCR_CONFIG = {
    "KID_No": {"lang": "eng", "psm": 6, "oem": 3},
    "Plus_Code": {"lang": "eng", "psm": 6, "oem": 3},
    "Local_Address": {"lang": "hin", "psm": 11, "oem": 3},
    "Ward_Address": {"lang": "nep", "psm": 11, "oem": 3},
    "City": {"lang": "hin", "psm": 11, "oem": 3},
}

# Camera Settings
LIVE_CAMERA = 0
HOLD_SECONDS = 5
MAX_OCR_ROUNDS = 5

# API Endpoints
LOGIN_URL = "https://kataho.app/api/login"
DATA_URL = "https://kataho.app/api/plate-status-check"
```

## 📊 Extracted Fields

The system extracts and validates:

1. **KID_No** - Unique identifier (format: `XX-XXX-XXXX-XXXX`)
2. **Plus_Code** - Google Plus Code location
3. **Local_Address** - Devanagari local address
4. **Ward_Address** - Ward information with number
5. **City** - City name in Nepali
6. **QR_Code** - Embedded QR data
7. **Kataho_Address** - Full Kataho format address

## 🔄 Progressive Validation

The system uses a multi-round approach:

1. **Round 1**: Initial OCR + API validation
2. **Lock verified fields** (never re-OCR)
3. **Round 2-5**: Re-OCR only unverified fields
4. **Exit** when all fields verified or max rounds reached

## 🌐 Web API Usage

### POST /ocr

**Request**:
```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "plate_type=default" \
  -F "image=@plate.jpg"
```

**Response**:
```json
{
  "status": "success",
  "plate_type": "default",
  "ocr_result": {
    "KID_No": "01-234-5678-9012",
    "Plus_Code": "7MHM+WX76",
    "Local_Address": "काठमाडौं",
    "Ward_Address": "काठमाडौं महानगरपालिका, वडा नं २६",
    "City": "काठमाडौं",
    "QR_Code": "http://kataho.app/KATAHO_12345"
  }
}
```

## 🎨 Advanced Features

### 1. Smart Post-processing
- Field-specific regex patterns
- Devanagari normalization
- Common OCR error correction

### 2. Multi-Strategy QR Detection
- 5 preprocessing strategies
- Fallback to pyzbar library
- Handles rotated/inverted codes

### 3. API Response Caching
- Reduces redundant API calls
- Improves performance
- Cache by QR+KID combination

### 4. ROI Template System
- Normalized coordinate templates
- Easy plate type customization
- Manual ROI selection support

## 📖 Dependencies

```
opencv-python       # Image processing
pytesseract        # OCR wrapper
numpy              # Arrays
pyzbar             # QR fallback
uvicorn            # Web server
fastapi            # Web framework
python-multipart   # File uploads
ultralytics        # YOLOv8
torch              # Deep learning
```

## 🔮 Future Enhancements

-  Multi-plate simultaneous processing
-  Field confidence scoring
-  OCR history database
-  ML-based post-correction
-  Mobile app integration
-  Offline validation mode
-  Multi-camera support

## 📝 License

This project is developed for Kataho address plate automation.

## 🤝 Contributing

For detailed technical information and architecture details, please refer to [DOCUMENTATION.md](./DOCUMENTATION.md).

---

**Project Status**: Production  
**Version**: 1.0  
**Last Updated**: 2026-02-09
