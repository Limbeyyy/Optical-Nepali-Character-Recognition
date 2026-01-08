import cv2
import numpy as np

def segment_characters(binary_img, boxes, pad=2, min_w=5, min_h=10):
    """
    Segment characters from binary image using bounding boxes.

    Args:
        binary_img: np.array, binary image
        boxes: list of tuples (x, y, w, h)
        pad: int, padding around each character
        min_w: int, minimum width to consider a character
        min_h: int, minimum height to consider a character

    Returns:
        List of np.array, each character in BGR format
    """
    chars = []
    for x, y, w, h in boxes:
        if w < min_w or h < min_h:
            continue

        # Add padding while keeping within image bounds
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, binary_img.shape[1])
        y2 = min(y + h + pad, binary_img.shape[0])

        char = binary_img[y1:y2, x1:x2]

        # Convert to BGR (needed for preprocess_char)
        char_bgr = cv2.cvtColor(char, cv2.COLOR_GRAY2BGR)

        chars.append(char_bgr)

    return chars
