import os
import cv2
import numpy as np
from rembg import remove, new_session

INPUT_DIR = './data/processed/camouflaged_object_image'
OUT_DIR_PT = './data/processed/camouflaged_object_image_black_background'

# ================= Configuration =================
# On first run the 'isnet-general-use' model (~170MB) will be auto-downloaded; keep your network connected
# This model handles general objects and edge details better than the default u2net, reducing false removals
MODEL_NAME = "isnet-general-use" 
PADDING_SIZE = 50  # Padding size to add a border around the image, preventing edge-clinging objects from being truncated

# Initialize session (outside the loop to avoid reloading the model)
print(f"Loading model {MODEL_NAME} ...")
session = new_session(MODEL_NAME)
print("Model loaded.")

def process_type_b_rembg_robust(img_bgr):
    """
    More robust background removal method:
    1. Padding: add white border to prevent edge-clinging objects from being truncated
    2. Model: use the stronger isnet model
    3. Composite: composite onto a pure black background
    """
    h, w = img_bgr.shape[:2]
    
    # --- 1. Add border padding ---
    # Use white fill (255,255,255) to give the AI “breathing room”
    img_padded = cv2.copyMakeBorder(
        img_bgr, 
        PADDING_SIZE, PADDING_SIZE, PADDING_SIZE, PADDING_SIZE, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )
    
    # --- 2. Use AI model to remove background ---
    # alpha_matting=True enables edge feathering for softer edges
    # If you find small holes have been punched inside the object, set alpha_matting to False
    result_rgba = remove(
        img_padded, 
        session=session,
        alpha_matting=True, 
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=0 # Set to 0 to prevent eroding object edges
    )

    # --- 3. Crop off the border ---
    # Restore to original image dimensions
    result_cropped = result_rgba[PADDING_SIZE:h+PADDING_SIZE, PADDING_SIZE:w+PADDING_SIZE]
    
    # --- 4. Composite onto black background ---
    # Extract alpha channel
    alpha = result_cropped[:, :, 3]
    
    # Normalize alpha (0.0 - 1.0)
    alpha_factor = alpha[:, :, np.newaxis] / 255.0
    
    # Extract foreground RGB
    foreground = result_cropped[:, :, :3]

    # Blending formula: foreground * alpha + background(black 0) * (1-alpha)
    # Since the background is black, the second half is 0, so just compute the first half
    img_final = (foreground * alpha_factor).astype(np.uint8)
    
    return img_final

def main():
    if not os.path.exists(OUT_DIR_PT):
        os.makedirs(OUT_DIR_PT)

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Starting processing, {len(files)} images total...")

    for i, filename in enumerate(files):
        file_path = os.path.join(INPUT_DIR, filename)
        
        # Parse prefix
        try:
            prefix = int(filename[:3])
        except:
            continue
            
        img_bgr = cv2.imread(file_path)
        if img_bgr is None: continue

        # ================= Type A: Text/Geometry (001-064) =================
        if 1 <= prefix <= 64:
            # Logic: Type A keeps the original OpenCV approach (results are usually good enough)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Binarize to enhance contrast
            _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_invert = 255 - img_thresh
            img_final = cv2.cvtColor(img_invert, cv2.COLOR_GRAY2BGR)
            
            print(f"[{i+1}/{len(files)}] Type A: {filename} -> White text on black background conversion")
            cv2.imwrite(os.path.join(OUT_DIR_PT, filename), img_final)

        # ================= Type B: Objects (065-128) =================
        elif 65 <= prefix <= 128:
            print(f"[{i+1}/{len(files)}] Type B: {filename} -> AI intelligent background removal (Padding+ISNet)")

            try:
                # Use enhanced AI method
                img_final = process_type_b_rembg_robust(img_bgr)
                cv2.imwrite(os.path.join(OUT_DIR_PT, filename), img_final)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("\nPTDiffusion data preparation complete! Check folder:", OUT_DIR_PT)

if __name__ == "__main__":
    main()