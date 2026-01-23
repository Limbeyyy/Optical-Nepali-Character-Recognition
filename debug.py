KID
# ---------------- CHARACTER RECOGNITION ----------------
def recognize_word(gray_img, field="global", strict_char=False, threshold_ratio=0.77, min_gap_pixels=2):
    """
    OCR for a single word/line using histogram thresholding.
    strict_char: if True, segment each character individually (for KID, Plus Code)
    threshold_ratio: fraction of max column height used to define splits
    min_gap_pixels: minimum number of pixels between splits to consider as new character
    """
    img = binarize(gray_img)
    img = crop_text(img)
    save_debug(img, "char_bin.png", field)

    char_images = []

    if strict_char:
        # ------------------- Horizontal histogram -------------------
        col_sum = np.sum(img > 0, axis=0)
        threshold = threshold_ratio * col_sum.max()

        # Find candidate split positions where histogram is below threshold
        below_thresh = col_sum < threshold
        splits = []
        start = None
        for i, val in enumerate(below_thresh):
            if val:
                if start is None:
                    start = i
            else:
                if start is not None:
                    splits.append((start, i))
                    start = None
        if start is not None:
            splits.append((start, len(col_sum)))

        # Merge splits that are too close (tiny gaps)
        merged_splits = []
        for s, e in splits:
            if not merged_splits:
                merged_splits.append((s, e))
            else:
                prev_s, prev_e = merged_splits[-1]
                if s - prev_e <= min_gap_pixels:
                    merged_splits[-1] = (prev_s, e)
                else:
                    merged_splits.append((s, e))

        # ------------------- Extract character images -------------------
        for i, (s, e) in enumerate(merged_splits):
            char_img = img[:, s:e]
            if np.count_nonzero(char_img) < 10:
                continue
            char_images.append(char_img)
            save_debug(char_img, f"char_{i}.png", field)

        # ------------------- Visualize histogram + threshold -------------------
        hist_vis = np.zeros((100, img.shape[1]), np.uint8)
        col_vis = (col_sum / col_sum.max() * 100).astype(np.int32)
        for x, h in enumerate(col_vis):
            cv2.line(hist_vis, (x, 100), (x, 100 - h), 255, 1)
        thresh_h = int(threshold / col_sum.max() * 100)
        cv2.line(hist_vis, (0, 100 - thresh_h), (img.shape[1]-1, 100 - thresh_h), 128, 1)
        # Draw vertical lines for splits
        for s, e in merged_splits:
            cv2.line(hist_vis, (s, 0), (s, 100), 0, 1)
            cv2.line(hist_vis, (e, 0), (e, 100), 0, 1)
        save_debug(hist_vis, "hist_threshold_splits.png", field)

    else:
        # Normal connected components (old logic)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 20 or w < 3 or h < 3:
                continue
            char_img = img[y:y+h, x:x+w]
            char_images.append(char_img)
            save_debug(char_img, f"char_{i}.png", field)

        # Sort left to right
        if char_images:
            char_images = [img for _, img in sorted(zip([s[0] for s in stats[1:]], char_images))]

    # ------------------- CNN inference -------------------
    result = ""
    for char_img in char_images:
        result += infer_base(char_img)

    # ------------------- Visualization bounding boxes -------------------
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, char_img in enumerate(char_images):
        coords = cv2.findNonZero(char_img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cv2.rectangle(vis, (x, 0), (x+w, img.shape[0]), (0, 255, 0), 1)
    save_debug(vis, "char_boxes.png", field)

    return result




Kataho Adddress
def recognize_word(gray_img, field="global", strict_char=False, threshold_ratio=0.8, min_gap_pixels=2):
    """
    OCR for a single word/line using histogram thresholding with Shirorekha removal.

    strict_char: if True, segment each character individually (for KID, Plus Code, Kataho addresses)
    threshold_ratio: fraction of max column height used to define splits
    min_gap_pixels: minimum number of pixels between splits to consider as new character
    """
    img = binarize(gray_img)
    img = crop_text(img)
    save_debug(img, "char_bin.png", field)

    # ------------------- Shirorekha removal -------------------
    # Vertical projection histogram
    row_sum = np.sum(img > 0, axis=1)
    shiro_threshold = 0.7 * row_sum.max()  # rows above this are considered Shirorekha
    shiro_rows = np.where(row_sum > shiro_threshold)[0]

    if len(shiro_rows) > 0:
        img[shiro_rows, :] = 0  # remove Shirorekha
        save_debug(img, "char_no_shirorekha.png", field)

    char_images = []

    if strict_char:
        # ------------------- Horizontal histogram segmentation -------------------
        col_sum = np.sum(img > 0, axis=0)
        threshold = threshold_ratio * col_sum.max()

        below_thresh = col_sum < threshold
        splits = []
        start = None
        for i, val in enumerate(below_thresh):
            if val:
                if start is None:
                    start = i
            else:
                if start is not None:
                    splits.append((start, i))
                    start = None
        if start is not None:
            splits.append((start, len(col_sum)))

        # Merge splits that are too close
        merged_splits = []
        for s, e in splits:
            if not merged_splits:
                merged_splits.append((s, e))
            else:
                prev_s, prev_e = merged_splits[-1]
                if s - prev_e <= min_gap_pixels:
                    merged_splits[-1] = (prev_s, e)
                else:
                    merged_splits.append((s, e))

        # Extract character images
        for i, (s, e) in enumerate(merged_splits):
            char_img = img[:, s:e]
            if np.count_nonzero(char_img) < 10:
                continue
            char_images.append(char_img)
            save_debug(char_img, f"char_{i}.png", field)

        # Visualize histogram + threshold
        hist_vis = np.zeros((100, img.shape[1]), np.uint8)
        col_vis = (col_sum / col_sum.max() * 100).astype(np.int32)
        for x, h in enumerate(col_vis):
            cv2.line(hist_vis, (x, 100), (x, 100 - h), 255, 1)
        thresh_h = int(threshold / col_sum.max() * 100)
        cv2.line(hist_vis, (0, 100 - thresh_h), (img.shape[1]-1, 100 - thresh_h), 128, 1)
        for s, e in merged_splits:
            cv2.line(hist_vis, (s, 0), (s, 100), 0, 1)
            cv2.line(hist_vis, (e, 0), (e, 100), 0, 1)
        save_debug(hist_vis, "hist_threshold_splits.png", field)

    else:
        # ------------------- Connected components (normal text) -------------------
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        components = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 20 or w < 3 or h < 3:
                continue
            char_img = img[y:y+h, x:x+w]
            components.append((x, char_img))
            save_debug(char_img, f"char_{i}.png", field)

        # Sort left to right safely
        char_images = [img for x, img in sorted(components, key=lambda t: t[0])]

    # ------------------- CNN inference -------------------
    result = ""
    for char_img in char_images:
        result += infer_base(char_img)

    # ------------------- Visualization bounding boxes -------------------
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, char_img in enumerate(char_images):
        coords = cv2.findNonZero(char_img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cv2.rectangle(vis, (x, 0), (x+w, img.shape[0]), (0, 255, 0), 1)
    save_debug(vis, "char_boxes.png", field)

    return result


Ward Address
def recognize_word(gray_img, field="global", strict_char=False, threshold_ratio=0.65, min_gap_pixels=0.2):
    """
    OCR for a single word/line using histogram thresholding with Shirorekha removal.

    strict_char: if True, segment each character individually (for KID, Plus Code, Kataho addresses)
    threshold_ratio: fraction of max column height used to define splits
    min_gap_pixels: minimum number of pixels between splits to consider as new character
    """
    img = binarize(gray_img)
    img = crop_text(img)
    save_debug(img, "char_bin.png", field)

    # ------------------- Shirorekha removal -------------------
    # Vertical projection histogram
    row_sum = np.sum(img > 0, axis=1)
    shiro_threshold = 0.8 * row_sum.max()  # rows above this are considered Shirorekha
    shiro_rows = np.where(row_sum > shiro_threshold)[0]

    if len(shiro_rows) > 0:
        img[shiro_rows, :] = 0  # remove Shirorekha
        save_debug(img, "char_no_shirorekha.png", field)

    char_images = []

    if strict_char:
        # ------------------- Horizontal histogram segmentation -------------------
        col_sum = np.sum(img > 0, axis=0)
        threshold = threshold_ratio * col_sum.max()

        below_thresh = col_sum < threshold
        splits = []
        start = None
        for i, val in enumerate(below_thresh):
            if val:
                if start is None:
                    start = i
            else:
                if start is not None:
                    splits.append((start, i))
                    start = None
        if start is not None:
            splits.append((start, len(col_sum)))

        # Merge splits that are too close
        merged_splits = []
        for s, e in splits:
            if not merged_splits:
                merged_splits.append((s, e))
            else:
                prev_s, prev_e = merged_splits[-1]
                if s - prev_e <= min_gap_pixels:
                    merged_splits[-1] = (prev_s, e)
                else:
                    merged_splits.append((s, e))

        # Extract character images
        for i, (s, e) in enumerate(merged_splits):
            char_img = img[:, s:e]
            if np.count_nonzero(char_img) < 10:
                continue
            char_images.append(char_img)
            save_debug(char_img, f"char_{i}.png", field)

        # Visualize histogram + threshold
        hist_vis = np.zeros((100, img.shape[1]), np.uint8)
        col_vis = (col_sum / col_sum.max() * 100).astype(np.int32)
        for x, h in enumerate(col_vis):
            cv2.line(hist_vis, (x, 100), (x, 100 - h), 255, 1)
        thresh_h = int(threshold / col_sum.max() * 100)
        cv2.line(hist_vis, (0, 100 - thresh_h), (img.shape[1]-1, 100 - thresh_h), 128, 1)
        for s, e in merged_splits:
            cv2.line(hist_vis, (s, 0), (s, 100), 0, 1)
            cv2.line(hist_vis, (e, 0), (e, 100), 0, 1)
        save_debug(hist_vis, "hist_threshold_splits.png", field)

    else:
        # ------------------- Connected components (normal text) -------------------
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        components = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 20 or w < 3 or h < 3:
                continue
            char_img = img[y:y+h, x:x+w]
            components.append((x, char_img))
            save_debug(char_img, f"char_{i}.png", field)

        # Sort left to right safely
        char_images = [img for x, img in sorted(components, key=lambda t: t[0])]

    # ------------------- CNN inference -------------------
    result = ""
    for char_img in char_images:
        result += infer_base(char_img)

    # ------------------- Visualization bounding boxes -------------------
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, char_img in enumerate(char_images):
        coords = cv2.findNonZero(char_img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cv2.rectangle(vis, (x, 0), (x+w, img.shape[0]), (0, 255, 0), 1)
    save_debug(vis, "char_boxes.png", field)

    return result


Local Address
def recognize_word(gray_img, field="global", strict_char=False, threshold_ratio=0.82, min_gap_pixels=2):
    """
    OCR for a single word/line using histogram thresholding with Shirorekha removal.

    strict_char: if True, segment each character individually (for KID, Plus Code, Kataho addresses)
    threshold_ratio: fraction of max column height used to define splits
    min_gap_pixels: minimum number of pixels between splits to consider as new character
    """
    img = binarize(gray_img)
    img = crop_text(img)
    save_debug(img, "char_bin.png", field)

    # ------------------- Shirorekha removal -------------------
    # Vertical projection histogram
    row_sum = np.sum(img > 0, axis=1)
    shiro_threshold = 0.8 * row_sum.max()  # rows above this are considered Shirorekha
    shiro_rows = np.where(row_sum > shiro_threshold)[0]

    if len(shiro_rows) > 0:
        img[shiro_rows, :] = 0  # remove Shirorekha
        save_debug(img, "char_no_shirorekha.png", field)

    char_images = []

    if strict_char:
        # ------------------- Horizontal histogram segmentation -------------------
        col_sum = np.sum(img > 0, axis=0)
        threshold = threshold_ratio * col_sum.max()

        below_thresh = col_sum < threshold
        splits = []
        start = None
        for i, val in enumerate(below_thresh):
            if val:
                if start is None:
                    start = i
            else:
                if start is not None:
                    splits.append((start, i))
                    start = None
        if start is not None:
            splits.append((start, len(col_sum)))

        # Merge splits that are too close
        merged_splits = []
        for s, e in splits:
            if not merged_splits:
                merged_splits.append((s, e))
            else:
                prev_s, prev_e = merged_splits[-1]
                if s - prev_e <= min_gap_pixels:
                    merged_splits[-1] = (prev_s, e)
                else:
                    merged_splits.append((s, e))

        # Extract character images
        for i, (s, e) in enumerate(merged_splits):
            char_img = img[:, s:e]
            if np.count_nonzero(char_img) < 10:
                continue
            char_images.append(char_img)
            save_debug(char_img, f"char_{i}.png", field)

        # Visualize histogram + threshold
        hist_vis = np.zeros((100, img.shape[1]), np.uint8)
        col_vis = (col_sum / col_sum.max() * 100).astype(np.int32)
        for x, h in enumerate(col_vis):
            cv2.line(hist_vis, (x, 100), (x, 100 - h), 255, 1)
        thresh_h = int(threshold / col_sum.max() * 100)
        cv2.line(hist_vis, (0, 100 - thresh_h), (img.shape[1]-1, 100 - thresh_h), 128, 1)
        for s, e in merged_splits:
            cv2.line(hist_vis, (s, 0), (s, 100), 0, 1)
            cv2.line(hist_vis, (e, 0), (e, 100), 0, 1)
        save_debug(hist_vis, "hist_threshold_splits.png", field)

    else:
        # ------------------- Connected components (normal text) -------------------
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        components = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 20 or w < 3 or h < 3:
                continue
            char_img = img[y:y+h, x:x+w]
            components.append((x, char_img))
            save_debug(char_img, f"char_{i}.png", field)

        # Sort left to right safely
        char_images = [img for x, img in sorted(components, key=lambda t: t[0])]

    # ------------------- CNN inference -------------------
    result = ""
    for char_img in char_images:
        result += infer_base(char_img)

    # ------------------- Visualization bounding boxes -------------------
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, char_img in enumerate(char_images):
        coords = cv2.findNonZero(char_img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cv2.rectangle(vis, (x, 0), (x+w, img.shape[0]), (0, 255, 0), 1)
    save_debug(vis, "char_boxes.png", field)

    return result



