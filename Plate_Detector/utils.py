

def shrink_bbox(x1, y1, x2, y2, shrink_w_ratio=0.015, shrink_h_ratio= 0.01):
    """
    Shrink bounding box from all sides by a percentage.
    shrink_ratio = 0.02 means 2% from each side
    """
    w = x2 - x1
    h = y2 - y1

    dx = int(w * shrink_w_ratio)
    dy = int(h * shrink_h_ratio)

    nx1 = x1 + dx
    ny1 = y1 + dy
    nx2 = x2 - dx
    ny2 = y2 - dy

    return nx1, ny1, nx2, ny2


def shrink_bbox_vertical(x1, y1, x2, y2, shrink_ratio=0.02):
    """
    Shrink bounding box only from top and bottom by a percentage.
    Horizontal (x) remains unchanged.
    """
    h = y2 - y1
    dy = int(h * shrink_ratio)

    ny1 = y1 + dy
    ny2 = y2 - dy

    return x1, ny1, x2, 
    

def shrink_vertical(y1, y2, max_h, ratio):
    dy = int((y2 - y1) * ratio)
    y1 = max(0, y1 + dy)
    y2 = min(max_h, y2 - dy)
    return y1, y2

