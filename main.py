import argparse
from OCR_Engine.ocr_engine import run_plate_ocr

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--template", required=True)
args = parser.parse_args()

results = run_plate_ocr(args.image, args.template)

for k, v in results.items():
    print(f"{k}: {v}")
