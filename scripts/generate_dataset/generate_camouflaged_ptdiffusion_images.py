import os
import cv2
import numpy as np

# ================= Configuration =================
INPUT_DIR = "data\images\camouflaged_object_image\camouflaged_object_image"
OUTPUT_DIR = "data\images\camouflaged_object_image\camouflaged_object_image_ptdiffusion"

# Noise intensity (determines texture roughness)
# Around 20 works well — adds texture without destroying original content
NOISE_SIGMA = 25
# =======================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Processing images for Phase-Transfer (Texturization)...")

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue
    
    file_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(file_path)
    if img is None: continue
    
    # 1. Compress dynamic range (Contrast Compression)
    # This step is critical! Compress 0-255 down to 10-245.
    # Goal: turn pure black into dark gray, pure white into light gray.
    # This prevents noise from being erased by clipping after addition.
    img_float = img.astype(np.float32)
    img_compressed = 20 + (img_float / 255.0) * 215.0
    
    # 2. Generate fixed texture noise (Baked-in Texture)
    noise = np.random.normal(0, NOISE_SIGMA, img.shape).astype(np.float32)
    
    # 3. Overlay
    final_float = img_compressed + noise
    
    # 4. Clip back to 0-255 and save
    final_img = np.clip(final_float, 0, 255).astype(np.uint8)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, final_img)
    print(f"Processed: {filename}")

print(f"Done! Use '{OUTPUT_DIR}' to run your search_d script.")