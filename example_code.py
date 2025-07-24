#!/usr/bin/env python3
#DONT TOUCH THIS FILE JUST READ FOR REFERENCE
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# --- Setup paths & environment ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)

# --- Load & resize image ---
def load_and_resize(path, max_dim=1024):
    img = Image.open(path)
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return np.array(img.convert("RGB"))

image = load_and_resize(os.path.join(BASE_DIR, "demo/arac2.jpg"))
H, W = image.shape[:2]

# --- Initialize SAM2 ---
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
checkpoint = os.path.join(BASE_DIR, "checkpoints", "sam2.1_hiera_large.pt")
config = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
sam2 = build_sam2(config, checkpoint, device=device, apply_postprocessing=False)
mask_gen = SAM2AutomaticMaskGenerator(sam2)

# --- Stage 1: Coarse rim ROI detection via SAM masks ---
masks = mask_gen.generate(image)
print(f"Generated {len(masks)} masks on full image")

def filter_rims(masks, min_area=1200, circ_thresh=0.6):
    left, right = [], []
    for ann in masks:
        seg = ann['segmentation'].astype(np.uint8)
        area = ann['area']
        if area < min_area:
            continue
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        circ = 4*np.pi*area/(cv2.arcLength(cnt, True)**2 + 1e-6)
        if circ < circ_thresh:
            continue
        ys, xs = np.where(seg)
        if ys.mean() < H*0.4:
            continue
        w_box, h_box = xs.max()-xs.min(), ys.max()-ys.min()
        if w_box == 0 or h_box == 0:
            continue
        ar = w_box/h_box
        if not (0.8 <= ar <= 1.2):
            continue
        cx = xs.mean()
        if cx < W*0.5:
            left.append((circ, seg))
        else:
            right.append((circ, seg))
    rims = []
    if left:
        rims.append(max(left, key=lambda x: x[0])[1])
    if right:
        rims.append(max(right, key=lambda x: x[0])[1])
    return rims

coarse_rims = filter_rims(masks)
rois = []
for seg in coarse_rims:
    ys, xs = np.where(seg)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    rois.append((box, seg))
print(f"Detected ROIs: {[box for box,_ in rois]}")

# --- Debug: visualize ROIs ---
for idx, (box, seg) in enumerate(rois):
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    plt.figure(figsize=(4,4)); plt.title(f"ROI {idx}")
    plt.imshow(crop)
    overlay = np.zeros((y1-y0, x1-x0, 4))
    mask_crop = seg[y0:y1, x0:x1].astype(bool)
    overlay[mask_crop] = [1,0,0,0.3]
    plt.imshow(overlay)
    plt.axis('off')

# --- Stage 2: Combined Hough + ROI-based SAM refinement ---
final_masks = []
for idx, (box, seg) in enumerate(rois):
    x0, y0, x1, y1 = box
    # Crop ROI
    crop = image[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    # First, attempt HoughCircles on ROI
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)
    edges = cv2.Canny(blur, 50, 150)
    min_r = int(0.3 * min(w, h))
    max_r = int(0.6 * min(w, h))
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h/2,
                               param1=50, param2=30,
                               minRadius=min_r, maxRadius=max_r)
    use_hough = False
    if circles is not None:
        circs = np.round(circles[0]).astype(int)
        cx0, cy0 = w//2, h//2
        cx, cy, r = min(circs, key=lambda c: np.hypot(c[0]-cx0, c[1]-cy0))
        Y, X = np.ogrid[:h, :w]
        circ_mask = (X-cx)**2 + (Y-cy)**2 <= r*r
        # validate area ratio
        coarse_area = seg[y0:y1, x0:x1].astype(bool).sum()
        if coarse_area>0 and circ_mask.sum() >= 0.5 * coarse_area:
            mask_final = circ_mask
            use_hough = True
    if not use_hough:
        # ROI-based SAM refinement
        masks_roi = mask_gen.generate(crop)
        best_mask = None
        best_score = 0
        for ann in masks_roi:
            seg_roi = ann['segmentation'].astype(np.uint8)
            area = ann['area']
            cnts, _ = cv2.findContours(seg_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            circ = 4*np.pi*area/(peri*peri + 1e-6)
            score = circ * area
            if score > best_score:
                best_score = score
                best_mask = seg_roi.astype(bool)
        if best_mask is not None:
            mask_final = best_mask
        else:
            # fallback to coarse ROI mask
            mask_final = seg[y0:y1, x0:x1].astype(bool)
    final_masks.append((mask_final, box))
print(f"Final rims after combined refinement: {len(final_masks)}")

# --- Debug: visualize final rims ---
for idx, (mask, (x0, y0, x1, y1)) in enumerate(final_masks):
    crop = image[y0:y1, x0:x1]
    plt.figure(figsize=(4,4)); plt.title(f"Hough Rim {idx}")
    plt.imshow(crop)
    overlay = np.zeros((y1-y0, x1-x0, 4))
    overlay[mask] = [0,1,0,0.5]
    plt.imshow(overlay)
    plt.axis('off')

# --- Final overlay on full image ---
overlay_full = np.zeros((H, W, 4))
for mask, (x0, y0, x1, y1) in final_masks:
    ys, xs = np.where(mask)
    overlay_full[ys+y0, xs+x0] = [0,1,0,0.5]
plt.figure(figsize=(12,6))
plt.imshow(image)
plt.imshow(overlay_full)
plt.axis('off')
plt.show()
