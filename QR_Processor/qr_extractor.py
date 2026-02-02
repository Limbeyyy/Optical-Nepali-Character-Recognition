# qr_extractor.py
import cv2
import argparse
import os

DEBUG_DIR = "QR_visualizations"
os.makedirs(DEBUG_DIR, exist_ok=True)


def crop_roi(img, roi, expand=0.15):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, int((x1 - expand) * w))
    y1 = max(0, int((y1 - expand) * h))
    x2 = min(w, int((x2 + expand) * w))
    y2 = min(h, int((y2 + expand) * h))
    return img[y1:y2, x1:x2]


def upscale(img, scale=2):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input plate image")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("x1", "y1", "x2", "y2"),
        help="Normalized ROI coordinates"
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("Could not read image")
        return

    # ---- Crop ROI ----
    if args.roi:
        img = crop_roi(img, args.roi)

    if img.size == 0:
        print("Empty ROI")
        return

    cv2.imwrite(f"{DEBUG_DIR}/01_roi.png", img)

    # ---- Upscale ----
    up = upscale(img, 2)
    cv2.imwrite(f"{DEBUG_DIR}/02_upscaled.png", up)

    # ---- Grayscale ----
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{DEBUG_DIR}/03_gray.png", gray)

    # ---- Grayscale inverted ----
    gray_inv = cv2.bitwise_not(gray)
    cv2.imwrite(f"{DEBUG_DIR}/03_gray_inverted.png", gray_inv)

    # ---- Adaptive Threshold ----
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )
    cv2.imwrite(f"{DEBUG_DIR}/04_thresh.png", thresh)

    # ---- Threshold inverted ----
    thresh_inv = cv2.bitwise_not(thresh)
    cv2.imwrite(f"{DEBUG_DIR}/05_thresh_inverted.png", thresh_inv)

    print(f"QR preprocessing done. Check folder '{DEBUG_DIR}' for images.")


if __name__ == "__main__":
    main()
