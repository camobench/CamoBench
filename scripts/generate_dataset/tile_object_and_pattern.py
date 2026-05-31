import cv2
import numpy as np
import os
import glob

def process_tile_images(input_dir, output_dir):
    # 1. Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"Found {len(files)} images to process...")

    for file_path in files:
        filename = os.path.basename(file_path)

        try:
            file_id_str = filename[:3]
            if not file_id_str.isdigit():
                print(f"Skipping {filename}: Does not start with 3 digits.")
                continue
            file_id = int(file_id_str)
        except Exception as e:
            print(f"Skipping {filename}: Error parsing ID - {e}")
            continue

        # Read image (force grayscale)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error reading {filename}")
            continue

        output_img = None

        # =========================================================
        # Strategy A: ID 001 - 064 (Text/Symbol/Pattern)
        # Logic unchanged: linear remap to mid-gray region + strong blur
        # =========================================================
        if 1 <= file_id <= 64:
            target_min = 60
            target_max = 190

            # Linear stretch: 0->60, 255->190
            normalized = cv2.normalize(img, None, alpha=target_min, beta=target_max,
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Strong blur (Sigma=5.0) -> create a "cloud-like" feel
            output_img = cv2.GaussianBlur(normalized, (0, 0), sigmaX=5.0)

            print(f"Processed Group A (Text): {filename}")

        # =========================================================
        # Strategy B: ID 065 - 128 (Objects)
        # Modified logic: globally lift dead-black pixels + light blur
        # =========================================================
        elif 65 <= file_id <= 128:
            # Copy original
            final_img = img.copy()

            # Target background gray level: 60 (dark gray)
            background_gray_value = 60

            # Global replacement: find all near-black pixels (<20) and set them to 60
            # No longer distinguish between background black and object-internal black
            final_img[final_img < 20] = background_gray_value

            # Gaussian blur
            # This step is still necessary to soften hard edges from the color replacement
            output_img = cv2.GaussianBlur(final_img, (0, 0), sigmaX=5.0)

            print(f"Processed Group B (Object/Global Lift): {filename}")

        else:
            # ID out of range, copy as-is
            output_img = img
            print(f"Copied others: {filename}")

        # Save result
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, output_img)

    print("All processing complete.")

# =================================================
# Configuration / Entry point
# =================================================
if __name__ == "__main__":
    source_folder = "data\processed\camouflaged_object_image_qrcodemonster"
    target_folder = "data\processed\camouflaged_object_image_tile"
    
    process_tile_images(source_folder, target_folder)