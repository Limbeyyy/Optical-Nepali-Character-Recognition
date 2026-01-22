
def normalize_rois(rois, shape):
    h, w = shape[:2]
    return {k:(x1/w,y1/h,x2/w,y2/h) for k,(x1,y1,x2,y2) in rois.items()}

def denormalize_rois(rois, shape):
    h, w = shape[:2]
    return {k:(int(x1*w),int(y1*h),int(x2*w),int(y2*h)) for k,(x1,y1,x2,y2) in rois.items()}
