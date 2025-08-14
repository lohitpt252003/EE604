#!/usr/bin/env python3
"""
make_mask_fixed.py

Cleaned / corrected version of your mask generation code.
- Fixed syntax/indentation errors
- Clamped writes to image bounds
- Added proof output (pixel count, bounding box, coverage)
- Uses cv2 to save, falls back to Pillow if cv2 unavailable

Usage:
    python make_mask_fixed.py         # uses default r=50
    python make_mask_fixed.py --r 80  # change scale
"""
import argparse
import math
import numpy as np

# optional image saving libraries
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    try:
        from PIL import Image
        _HAS_PIL = True
    except Exception:
        _HAS_PIL = False


def clamp(val, lo, hi):
    return max(lo, min(val, hi))


def make_mask(r=50, img_size=None):
    if img_size is None:
        img_size = 8 * r

    # ensure img_size is at least something reasonable
    img_size = max(img_size, 4 * r + 10)

    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cx, cy = img_size // 2, img_size // 2

    # rectangle params
    outer_x1, outer_x2 = cx - r, cx + r
    outer_y1, outer_y2 = cy - r, cy + r

    inner_x1, inner_x2 = cx - r // 2, cx + r // 2
    inner_y1, inner_y2 = cy - r // 2, cy + r // 2

    # clamp helper for inclusive loops
    x0 = clamp(outer_x1, 0, img_size - 1)
    x1 = clamp(outer_x2, 0, img_size - 1)
    y0 = clamp(outer_y1, 0, img_size - 1)
    y1 = clamp(outer_y2, 0, img_size - 1)

    # Fill outer rectangle (inclusive)
    for y in range(y0, y1 + 1):
        mask[y, x0:x1 + 1] = 255

    # Cut out inner rectangle (inclusive)
    ix0 = clamp(inner_x1, 0, img_size - 1)
    ix1 = clamp(inner_x2, 0, img_size - 1)
    iy0 = clamp(inner_y1, 0, img_size - 1)
    iy1 = clamp(inner_y2, 0, img_size - 1)
    for y in range(iy0, iy1 + 1):
        mask[y, ix0:ix1 + 1] = 0

    # Left triangle "ear"
    for y in range(y0, y1 + 1):
        if y <= cy:
            dy = y - outer_y1
        else:
            dy = outer_y2 - y
        # dy in [0, r]; use it as dx
        dx = int(round((dy / max(1, r)) * r))
        sx = clamp(outer_x1 - dx, 0, img_size - 1)
        ex = clamp(outer_x1, 0, img_size - 1)
        mask[y, sx:ex + 1] = 255

    # Right triangle "ear"
    for y in range(y0, y1 + 1):
        if y <= cy:
            dy = y - outer_y1
        else:
            dy = outer_y2 - y
        dx = int(round((dy / max(1, r)) * r))
        sx = clamp(outer_x2, 0, img_size - 1)
        ex = clamp(outer_x2 + dx, 0, img_size - 1)
        mask[y, sx:ex + 1] = 255

    # Top hollow semicircle (centered at (cx, outer_y1))
    rad = r
    rad_inner = max(1, r // 2)
    ty0 = clamp(outer_y1 - rad, 0, img_size - 1)
    ty1 = clamp(outer_y1, 0, img_size - 1)
    tx0 = clamp(outer_x1, 0, img_size - 1)
    tx1 = clamp(outer_x2, 0, img_size - 1)
    for y in range(ty0, ty1 + 1):
        for x in range(tx0, tx1 + 1):
            eq_outer = ((x - cx) ** 2) / (rad ** 2) + ((y - outer_y1) ** 2) / (rad ** 2)
            eq_inner = ((x - cx) ** 2) / (rad_inner ** 2) + ((y - outer_y1) ** 2) / (rad_inner ** 2)
            if eq_outer <= 1.0 and eq_inner >= 1.0:
                mask[y, x] = 255

    # Bottom hollow semicircle (centered at (cx, outer_y2))
    by0 = clamp(outer_y2, 0, img_size - 1)
    by1 = clamp(outer_y2 + rad, 0, img_size - 1)
    bx0 = clamp(outer_x1, 0, img_size - 1)
    bx1 = clamp(outer_x2, 0, img_size - 1)
    for y in range(by0, by1 + 1):
        for x in range(bx0, bx1 + 1):
            eq_outer = ((x - cx) ** 2) / (rad ** 2) + ((y - outer_y2) ** 2) / (rad ** 2)
            eq_inner = ((x - cx) ** 2) / (rad_inner ** 2) + ((y - outer_y2) ** 2) / (rad_inner ** 2)
            if eq_outer <= 1.0 and eq_inner >= 1.0:
                mask[y, x] = 255

    return mask


def save_image(mask, fname="mask_binary.png"):
    if _HAS_CV2:
        # cv2.imwrite expects BGR for color; for grayscale single channel is fine
        cv2.imwrite(fname, mask)
    elif 'Image' in globals() or _HAS_PIL:
        # use Pillow fallback
        from PIL import Image
        im = Image.fromarray(mask)
        im.save(fname)
    else:
        raise RuntimeError("No supported image library found to save the mask (install opencv-python or pillow).")


def print_proof(mask):
    fg = int(np.count_nonzero(mask))
    total = mask.size
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("No foreground pixels found.")
        return
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    width = int(xmax - xmin + 1)
    height = int(ymax - ymin + 1)
    coverage = 100.0 * fg / total
    print("=== Proof / Verification ===")
    print(f"Image size: {mask.shape[1]} x {mask.shape[0]} (w x h)")
    print(f"Foreground pixels (value=255): {fg}")
    print(f"Bounding box (xmin, ymin, width, height): ({xmin}, {ymin}, {width}, {height})")
    print(f"Coverage: {coverage:.3f}%")
    print("=============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=50, help="scale parameter r (default 50)")
    parser.add_argument("--out", type=str, default="mask_binary.png", help="output filename")
    args = parser.parse_args()

    r = max(1, args.r)
    mask = make_mask(r=r)
    save_image(mask, fname=args.out)
    print(f"Saved mask to: {args.out}")
    print_proof(mask)
