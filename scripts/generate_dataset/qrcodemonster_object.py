import os
from PIL import Image, ImageOps

def process_images():
    source_folder = './data/processed/camouflaged_object_image'
    target_folder = './data/processed/camouflaged_object_image_luminance'

    # Create target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created folder: {target_folder}")

    # Get the list of all files in the source folder
    files = os.listdir(source_folder)

    count = 0
    for filename in files:
        # Check file extension
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):

            # Extract the first 3 characters and check if between 065-128
            prefix = filename[:3]
            if prefix.isdigit() and 65 <= int(prefix) <= 128:

                source_path = os.path.join(source_folder, filename)
                target_path = os.path.join(target_folder, filename)

                try:
                    with Image.open(source_path) as img:
                        # ---------------- Modification start ----------------

                        # 1. Convert to grayscale (Mode 'L' = 8-bit pixels, black and white)
                        # convert('L') automatically handles color-to-grayscale conversion
                        gray_img = img.convert('L')

                        # 2. Invert colors (on the grayscale image: black becomes white, white becomes black)
                        inverted_image = ImageOps.invert(gray_img)

                        # ---------------- Modification end ----------------

                        # Save to target path
                        inverted_image.save(target_path)
                        count += 1
                        print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    print(f"\nDone! Converted to grayscale and inverted {count} images.")

if __name__ == "__main__":
    process_images()