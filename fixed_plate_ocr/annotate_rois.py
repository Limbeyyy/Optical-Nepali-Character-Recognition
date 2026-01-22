
import cv2

img = cv2.imread("/home/kataho/Downloads/mallanet_ocr/data/test_images/plate_1.jpeg")
rois = {}
FIELDS = ["kataho_address","KID_No","Plus_Code","Address_Name"]
pt1 = None
current = None

def mouse(event,x,y,flags,param):
    global pt1
    if event == cv2.EVENT_LBUTTONDOWN:
        pt1 = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        rois[current] = (*pt1, x, y)
        cv2.rectangle(img, pt1, (x,y), (0,255,0), 2)
        cv2.imshow("img", img)

cv2.namedWindow("img")
cv2.setMouseCallback("img", mouse)

for f in FIELDS:
    current = f
    print("Draw ROI for:", f)
    cv2.imshow("img", img)
    cv2.waitKey(0)

print(rois)
cv2.destroyAllWindows()
