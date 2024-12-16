import os
import re

# Define the source directory (the directory where your images are located)
source_dir = r"C:\Users\Arjun Koirala\Documents\segmentation\devanagari_characters"  # Update this path
# Define the destination file where you want to save the names
destination_file = r"C:\Users\Arjun Koirala\Documents\segmentation\image_names.txt"  # Update this path

# Function to extract numeric part for sorting
def extract_number(filename):
    match = re.search(r'_(\d+)', filename)  # Find the number after the last underscore
    return int(match.group(1)) if match else float('inf')

# Check if the source directory exists
if not os.path.exists(source_dir):
    print(f"Error: The source directory does not exist: {source_dir}")
else:
    # Open the destination file for writing
    with open(destination_file, 'w') as f:
        # List all image files and sort them numerically based on the number in their names
        image_files = sorted([filename for filename in os.listdir(source_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))],
                             key=extract_number)

        # Iterate through the sorted list of image files
        for filename in image_files:
            # Keep the original image name without the extension
            original_name = os.path.splitext(filename)[0]  # Get the name without extension
            # Format the output with the original image name followed by a pipeline sign
            formatted_output = f"{original_name}|\n"
            # Write the formatted name to the file
            f.write(formatted_output)

    print(f"Image names have been saved to: {destination_file}")
