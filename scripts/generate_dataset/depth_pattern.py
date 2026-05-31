import os
import cv2
import numpy as np

def generate_gradient_depth(binary_img, dilate_iter=5, gamma=0.8):
    # dilate iter
    # 1-12 25-27 29-56 -> 40 
    # 13-16 28 57 60 61 63 -> 20
    # 24 59 -> 10
    # 17-23 -> 5
    """
    Use distance transform to generate a “ridge-like” depth map
    binary_img: binary image with white text on black background
    dilate_iter: dilation iterations. Higher values make characters thicker and ridges wider (recommended 1-5)
    gamma: gamma value. Lower (e.g. 0.5) makes slopes rounder/fuller; 1.0 is linear; higher (e.g. 2.0) makes ridges sharper.
    """
    
    # --- 1. Preprocessing: Dilation [Key step] ---
    # This step physically thickens the characters for a more stable base
    if dilate_iter > 0:
        # 3x3 convolution kernel
        kernel = np.ones((3, 3), np.uint8) 
        # Higher iterations = thicker characters
        binary_img = cv2.dilate(binary_img, kernel, iterations=dilate_iter) 
    
    # --- 2. Distance transform ---
    dist_img = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    
    # --- 3. Normalization + Gamma correction ---
    if dist_img.max() > 0:
        # First normalize to 0-1 float for easier exponentiation
        dist_norm = dist_img / dist_img.max()
        
        # Gamma correction: color = color ^ gamma
        # gamma < 1 (e.g. 0.5) greatly boosts midtones, making the ridge appear wide
        dist_norm = np.power(dist_norm, gamma)
        
        # Scale back to 0-255
        dist_norm = (dist_norm * 255).astype(np.uint8)
    else:
        dist_norm = dist_img.astype(np.uint8)
    
    # --- 4. Blur ---
    # If you used heavy dilation, the blur here can be slightly smaller, e.g. (7,7) or (9,9)
    dist_final = cv2.GaussianBlur(dist_norm, (9, 9), 0)
    
    return dist_final

def main():
    # Configure directories
    SOURCE_DIR = './data/processed/camouflaged_object_image'
    OUTPUT_DIR = './data/processed/camouflaged_object_image_depth'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Get files
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for filename in files:
        prefix_str = filename[:3]
        if prefix_str.isdigit():
            prefix = int(prefix_str)
            
            if 17 <= prefix <= 23:
                print(f"Processing Pattern (Type A): {filename}")
                
                # Read the original image
                img_bgr = cv2.imread(os.path.join(SOURCE_DIR, filename))
                if img_bgr is None:
                    continue
                
                # Convert to grayscale
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                tile_img = cv2.bitwise_not(img_gray)
                
                # Ensure input is purely binary
                # Threshold 127, values above it set to 255
                _, binary_img = cv2.threshold(tile_img, 127, 255, cv2.THRESH_BINARY)
                
                # 2. Generate ridge-like depth map
                depth_img = generate_gradient_depth(binary_img)
                
                # Save the result
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), depth_img)

    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()