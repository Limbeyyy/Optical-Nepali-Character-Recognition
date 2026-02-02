from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import json
import pytesseract
from fastapi.middleware.cors import CORSMiddleware
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from OCR_Engine.ocr_engine import run_plate_ocr


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

app = FastAPI(title="Plate OCR API")

# MiddleWares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "images"
TEMPLATE_DIR = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\OCR_Plate_Processor\Plate_Templates"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)


@app.post("/ocr")
async def ocr(
    plate_type: str = Form(...),               # "default" | "manual"
    image: UploadFile = File(...),
    rois: str = Form(None)                      # JSON string (manual only)
):
    # --- Save image ---
    img_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(img_path, "wb") as f:
        f.write(await image.read())

    # --- Choose template ---
    if plate_type == "default":
        rois_json_path = os.path.join(
            TEMPLATE_DIR, "default_plate_template.json"
        )

        if not os.path.exists(rois_json_path):
            return JSONResponse(
                status_code=400,
                content={"error": "Default template not found"}
            )

    elif plate_type == "manual":
        if not rois:
            return JSONResponse(
                status_code=400,
                content={"error": "ROIs required for manual mode"}
            )

        rois_data = json.loads(rois)
        rois_json_path = os.path.join(
            TEMPLATE_DIR, "selected_plate_template.json"
        )

        with open(rois_json_path, "w") as f:
            json.dump(rois_data, f, indent=4)

    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid plate_type"}
        )

    # --- Run OCR ---
    result = run_plate_ocr(img_path, rois_json_path)

    return {
        "status": "success",
        "plate_type": plate_type,
        "ocr_result": result
    }
