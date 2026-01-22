
import cv2
import numpy as np

def extract_characters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hist = th.sum(axis=1)
    y = hist.argmax()
    th[y-1:y+2, :] = 0

    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h > 20 and w > 10:
            boxes.append((x,y,w,h))

    return sorted(boxes, key=lambda b:b[0])
