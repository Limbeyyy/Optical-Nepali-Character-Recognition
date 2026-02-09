import cv2

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="QR Code Reader")
    parser.add_argument("--image", required=True, help="Path to QR image")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("Could not read image")
        sys.exit(1)

    qr_data = read_qr_from_image(img, verbose=True)

    if qr_data:
        print("\n FINAL QR VALUE:")
        print(qr_data)
    else:
        print("\n No QR detected")
