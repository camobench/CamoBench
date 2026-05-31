import os
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_FOLDER_DIR = BASE_DIR / "data" / "processed" / "camouflaged_object_image"

def process_images(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return

    print(f"Start processing folder: {folder_path}\n" + "-"*30)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Case-insensitive check for .png files only
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)

            try:
                # Open the image
                # Use with statement to ensure the image is automatically closed after reading dimensions, preventing file lock during overwrite
                with Image.open(file_path) as img:
                    width, height = img.size
                    # If modification is needed, a deep copy must be made or the image must be reopened below,
                    # because the img object will be closed after the with block
                
                # --- Logic branch ---
                
                # Case 1: If 1024x1024 -> skip
                if width == 1024 and height == 1024:
                    continue

                # Case 2: If 2048x2048 -> resize and overwrite
                elif width == 2048 and height == 2048:
                    print(f"[Resizing] {filename} (Original size: 2048x2048 -> Adjusting to 1024x1024)")

                    # Reopen and process
                    with Image.open(file_path) as img:
                        # Use LANCZOS filter to ensure scaling quality
                        resized_img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                        resized_img.save(file_path) # Overwrite original file

                # Case 3: Neither 1024 nor 2048 -> output filename
                else:
                    print(f"[Abnormal size] {filename} (Current size: {width}x{height})")

            except Exception as e:
                print(f"[Error] Unable to process file {filename}: {e}")

    print("-" * 30 + "\nProcessing complete.")


if __name__ == "__main__":
    target_dir = IMAGE_FOLDER_DIR 
    process_images(target_dir)