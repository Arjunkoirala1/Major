import os
import shutil

# Define file paths
caption_file = r'C:\Users\Nischal\Downloads\segmentation\output.txt'  # Path to the captioned text file
source_dir = r'devanagari_characters'  # Path to the source directory containing images
target_dir = r'C:\Users\Nischal\Downloads\segmentation\captioned_images'  # Path to the target directory

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Read the captioned file and filter images with non-blank captions
with open(caption_file, 'r') as file:
    lines = file.readlines()

# Extract image filenames with captions
captioned_images = {
    line.split('|')[0].strip() for line in lines if line.split('|')[1].strip()  # Check if caption is not blank
}

# Move only captioned images to the target directory
for image_file in captioned_images:
    src_path = os.path.join(source_dir, image_file)
    dst_path = os.path.join(target_dir, image_file)
    
    # Check if the file exists before moving
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved: {image_file}")
    else:
        print(f"File not found: {image_file}")

print("Captioned image transfer complete.")
