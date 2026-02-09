import cv2

try:
    from pyzbar.pyzbar import decode as zbar_decode
    PYZBAR_AVAILABLE = True
except:
    PYZBAR_AVAILABLE = False


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


def read_qr_from_image(image_or_path, verbose=False):
    """
    image_or_path: image path OR numpy image
    returns: decoded QR string or ""
    """

    # --- Load image if path ---
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
        if img is None:
            if verbose:
                print(" Could not read image from path")
            return ""
    else:
        img = image_or_path

    detector = cv2.QRCodeDetector()

    def try_decode(image, tag=""):
        data, _, _ = detector.detectAndDecode(image)
        if data:
            if verbose:
                print(f" QR ({tag}): {data}")
            return data
        return ""

    # --- Raw ---
    data = try_decode(img, "raw")
    if data:
        return data

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img


    # --- Gray ---
    data = try_decode(gray, "gray")
    if data:
        return data

    # --- Inverted ---
    gray_inv = cv2.bitwise_not(gray)
    data = try_decode(gray_inv, "gray_inverted")
    if data:
        return data

    # --- Threshold ---
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )
    data = try_decode(thresh, "thresh")
    if data:
        return data

    # --- pyzbar fallback ---
    if PYZBAR_AVAILABLE:
        codes = zbar_decode(img)
        if codes:
            qr = codes[0].data.decode("utf-8")
            if verbose:
                print(f" QR (pyzbar): {qr}")
            return qr

    if verbose:
        print("No QR detected")

    return ""
