"""
process_redlogo_no_builtin_ops.py

Performs required image operations WITHOUT using high-level built-in array ops.
Allowed libs: numpy (for allocation and dtype), PIL only for reading/writing images.
All image transformations are implemented with explicit loops.

Outputs (saved in 'outputs/' folder):
 - redlogo_transpose.png
 - redlogo_rot90_cw.png
 - redlogo_grayscale_avg.png
 - redlogo_flip_horizontal.png
 - redlogo_green_logo.png
 - redlogo_2x2_grid.png
 - redlogo_binary.png
 - redlogo_no_trishul.png
"""

import os
import numpy as np
from PIL import Image

# ---------- Settings ----------
INPUT_PATH = "redlogo.jpg"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# color thresholds for detecting red logo (tweak if needed)
RED_MIN = 100
GREEN_MAX = 100
BLUE_MAX = 100

# Trishul removal params (tweak to match your image)
TRISHUL_CENTER_X = 115  # x coordinate (columns)
TRISHUL_CENTER_Y = 109  # y coordinate (rows)
TRISHUL_RADIUS = 45

# ---------- Helpers ----------
def load_image_as_array(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return arr

def save_array_as_image(arr, path, mode="RGB"):
    # arr shape: HxW or HxWx3
    Image.fromarray(arr, mode=mode).save(path)
    print("Saved", path)

# ---------- Implementations without high-level built-ins ----------

def transpose_image_manual(arr):
    H, W = arr.shape[0], arr.shape[1]
    out = np.zeros((W, H, 3), dtype=np.uint8)  # transposed shape
    for i in range(H):
        for j in range(W):
            # place pixel (i,j) -> (j,i)
            out[j, i, 0] = arr[i, j, 0]
            out[j, i, 1] = arr[i, j, 1]
            out[j, i, 2] = arr[i, j, 2]
    return out

def rotate_90_cw_manual(arr):
    # rotation: (i, j) -> (j, H-1-i)
    H, W = arr.shape[0], arr.shape[1]
    out = np.zeros((W, H, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            ni = j
            nj = H - 1 - i
            out[ni, nj, 0] = arr[i, j, 0]
            out[ni, nj, 1] = arr[i, j, 1]
            out[ni, nj, 2] = arr[i, j, 2]
    return out

def grayscale_avg_manual(arr):
    H, W = arr.shape[0], arr.shape[1]
    out = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            r = int(arr[i,j,0])
            g = int(arr[i,j,1])
            b = int(arr[i,j,2])
            s = r + g + b
            avg = s // 3   # integer average
            out[i,j] = avg
    return out

def flip_horizontal_manual(arr):
    H, W = arr.shape[0], arr.shape[1]
    out = np.zeros_like(arr)
    for i in range(H):
        for j in range(W):
            out[i, W - 1 - j, 0] = arr[i, j, 0]
            out[i, W - 1 - j, 1] = arr[i, j, 1]
            out[i, W - 1 - j, 2] = arr[i, j, 2]
    return out

def logo_mask_manual(arr, rmin=RED_MIN, gmax=GREEN_MAX, bmax=BLUE_MAX):
    H, W = arr.shape[0], arr.shape[1]
    mask = np.zeros((H, W), dtype=np.bool_)
    for i in range(H):
        for j in range(W):
            r = int(arr[i,j,0])
            g = int(arr[i,j,1])
            b = int(arr[i,j,2])
            if (r >= rmin) and (g <= gmax) and (b <= bmax):
                mask[i,j] = True
    return mask

def recolor_to_green_manual(arr, mask):
    H, W = arr.shape[0], arr.shape[1]
    out = np.zeros_like(arr)
    for i in range(H):
        for j in range(W):
            if mask[i,j]:
                out[i,j,0] = 0
                out[i,j,1] = 255
                out[i,j,2] = 0
            else:
                out[i,j,0] = arr[i,j,0]
                out[i,j,1] = arr[i,j,1]
                out[i,j,2] = arr[i,j,2]
    return out

def tile_2x2_manual(arr):
    H, W = arr.shape[0], arr.shape[1]
    out = np.zeros((H*2, W*2, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            # top-left
            out[i, j, :] = arr[i,j,:]
            # top-right
            out[i, j + W, :] = arr[i,j,:]
            # bottom-left
            out[i + H, j, :] = arr[i,j,:]
            # bottom-right
            out[i + H, j + W, :] = arr[i,j,:]
    return out

def binary_logo_manual(mask):
    # mask: boolean HxW
    H, W = mask.shape
    out = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            out[i,j] = 255 if mask[i,j] else 0
    return out

def remove_trishul_manual(arr, center_x=TRISHUL_CENTER_X, center_y=TRISHUL_CENTER_Y, radius=TRISHUL_RADIUS):
    H, W = arr.shape[0], arr.shape[1]
    out = np.zeros_like(arr)
    # start by copying original
    for i in range(H):
        for j in range(W):
            out[i,j,0] = arr[i,j,0]
            out[i,j,1] = arr[i,j,1]
            out[i,j,2] = arr[i,j,2]
    radius2 = radius * radius
    # remove main circular region
    for i in range(H):
        for j in range(W):
            dx = j - center_x
            dy = i - center_y
            dist2 = dx*dx + dy*dy
            if dist2 <= radius2:
                out[i,j,0] = 255
                out[i,j,1] = 255
                out[i,j,2] = 255
    # remove three small eyes by small circular wipes (offsets relative to center)
    eye_offsets = [(-8, -6), (0, -9), (8, -6)]   # tweak if necessary
    eye_radius = max(3, radius // 8)
    eye_r2 = eye_radius * eye_radius
    for ex_off, ey_off in eye_offsets:
        ex = center_x + ex_off
        ey = center_y + ey_off
        for i in range(H):
            for j in range(W):
                dx = j - ex
                dy = i - ey
                if dx*dx + dy*dy <= eye_r2:
                    out[i,j,0] = 255
                    out[i,j,1] = 255
                    out[i,j,2] = 255
    return out

# ---------- Main ----------
def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("Place 'redlogo.jpg' in the same folder as this script or update INPUT_PATH.")
    arr = load_image_as_array(INPUT_PATH)
    H, W = arr.shape[0], arr.shape[1]
    print("Loaded image:", INPUT_PATH, "size:", W, "x", H)

    # 1) Transpose
    trans = transpose_image_manual(arr)
    save_array_as_image(trans, os.path.join(OUT_DIR, "redlogo_transpose.png"))

    # 2) Rotate 90 CW
    rot90 = rotate_90_cw_manual(arr)
    save_array_as_image(rot90, os.path.join(OUT_DIR, "redlogo_rot90_cw.png"))

    # 3) Grayscale (average)
    gray = grayscale_avg_manual(arr)
    save_array_as_image(gray, os.path.join(OUT_DIR, "redlogo_grayscale_avg.png"), mode="L")

    # 4) Flip horizontally
    flipped = flip_horizontal_manual(arr)
    save_array_as_image(flipped, os.path.join(OUT_DIR, "redlogo_flip_horizontal.png"))

    # 5) Change logo color to green
    mask = logo_mask_manual(arr)
    green = recolor_to_green_manual(arr, mask)
    save_array_as_image(green, os.path.join(OUT_DIR, "redlogo_green_logo.png"))

    # 6) 2x2 grid
    tiled = tile_2x2_manual(arr)
    save_array_as_image(tiled, os.path.join(OUT_DIR, "redlogo_2x2_grid.png"))

    # 7) Binary version
    binary = binary_logo_manual(mask)
    save_array_as_image(binary, os.path.join(OUT_DIR, "redlogo_binary.png"), mode="L")

    # 8) Remove trishul
    removed = remove_trishul_manual(arr)
    save_array_as_image(removed, os.path.join(OUT_DIR, "redlogo_no_trishul.png"))

    print("All done. Outputs are in:", OUT_DIR)

if __name__ == "__main__":
    main()
