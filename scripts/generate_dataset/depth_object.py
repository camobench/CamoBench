import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

# ================= Path Configuration =================
INPUT_DIR = './data/processed/camouflaged_object_image_black_background'
OUT_DIR_DEPTH = './data/processed/camouflaged_object_image_depth'

# ================= Model Configuration =================
MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"

def main():
    if not os.path.exists(OUT_DIR_DEPTH):
        os.makedirs(OUT_DIR_DEPTH)

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model {MODEL_ID} to {device} ...")
    depth_pipe = pipeline("depth-estimation", model=MODEL_ID, device=device)
    print("Model loaded successfully.")

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Starting depth map generation, total {len(files)} images...")

    for i, filename in enumerate(files):
        try:
            prefix = int(filename[:3])
            if not (65 <= prefix <= 128):
                continue
        except:
            continue

        file_path = os.path.join(INPUT_DIR, filename)

        # --- A. Read image ---
        img_bgr = cv2.imread(file_path)
        if img_bgr is None: continue

        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # --- B. Inference ---
        pipe_out = depth_pipe(img_pil)
        depth_pil = pipe_out["depth"]

        # --- C. Post-processing ---
        depth_map = np.array(depth_pil)

        # 1. Size alignment
        h, w = img_bgr.shape[:2]
        if depth_map.shape[:2] != (h, w):
            depth_map = cv2.resize(depth_map, (w, h))

        # 2. Normalize (0-255)
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 3. Smart Masking - Fixed version
        # Goal: only blacken the outer background, preserving black regions inside the object

        gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Step 1: Binarize — background becomes 0, object (and non-black interior) becomes 255
        # At this point, pure black pixels inside the object are also 0 (this needs to be fixed)
        _, binary_base = cv2.threshold(gray_src, 1, 255, cv2.THRESH_BINARY)

        # Step 2: Find contours
        # RETR_EXTERNAL only finds outer contours, ignoring internal holes
        contours, _ = cv2.findContours(binary_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 3: Draw solid mask (Hole Filling)
        # Create a pure black canvas
        solid_mask = np.zeros_like(gray_src)
        # Fill all found outer contours with white (255)
        # This way, originally black eyes/spots inside the object now become part of the white mask
        cv2.drawContours(solid_mask, contours, -1, 255, thickness=cv2.FILLED)

        # Step 4: Apply mask
        # In solid_mask, white represents "entire object area", black represents "true background"
        depth_clean = cv2.bitwise_and(depth_map, depth_map, mask=solid_mask)

        # 4. Light Gaussian blur
        depth_final = cv2.GaussianBlur(depth_clean, (5, 5), 0)

        # --- D. Save ---
        cv2.imwrite(os.path.join(OUT_DIR_DEPTH, filename), depth_final)

        print(f"[{i+1}/{len(files)}] Depth (Large): {filename} -> done")

    print("\nDepth map generation complete!")

if __name__ == "__main__":
    main()