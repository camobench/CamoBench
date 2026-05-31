import os
from PIL import Image, ImageOps

def process_images():
    source_folder = './data/processed/camouflaged_object_image'
    target_folder = './data/processed/camouflaged_object_image_luminance'

    # If target folder does not exist, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created folder: {target_folder}")

    # Get all file names in the source folder
    files = os.listdir(source_folder)

    count = 0
    for filename in files:
        # Check file extension to ensure it is an image (can add .png, .jpeg, etc. as needed)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):

            # Extract first three characters and check if they are between 001-064
            prefix = filename[:3]
            if prefix.isdigit() and 1 <= int(prefix) <= 64:

                source_path = os.path.join(source_folder, filename)
                target_path = os.path.join(target_folder, filename)

                try:
                    # Open and process image
                    with Image.open(source_path) as img:
                        # If RGBA mode, convert to RGB before inverting
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')

                        # Perform color inversion
                        inverted_image = ImageOps.invert(img)

                        # Save to target path
                        inverted_image.save(target_path)
                        count += 1
                        print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    print(f"\nTask complete! Inverted and saved {count} images.")

if __name__ == "__main__":
    process_images()