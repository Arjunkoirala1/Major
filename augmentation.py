import os
from PIL import Image
import numpy as np
import time

# Function to add grain to an image
def add_grain(image_array):
    noise = np.random.normal(loc=0, scale=25, size=image_array.shape).astype(np.uint8)
    noisy_image = np.clip(image_array + noise, 0, 255)
    return noisy_image

# Function to invert the colors of an image
def invert_colors(image):
    image_array = np.array(image)
    inverted_image_array = 255 - image_array  # Invert colors
    return Image.fromarray(inverted_image_array)

# Function to adjust the brightness of an image
def adjust_brightness(image, brightness_factor=1.5):
    return Image.fromarray(np.clip(np.array(image) * brightness_factor, 0, 255).astype(np.uint8))

# Function to save augmented images and captions
def save_images_with_captions(image_dir, captions_file, output_dir, sleep_time=2):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load captions
    with open(captions_file, "r", encoding="utf-8") as f:
        captions = f.readlines()

    # Get all files in the output directory
    processed_files = set(os.listdir(output_dir))

    # Identify fully processed images (check original files)
    augmented_prefixes = {"original_", "grainy_", "inverted_", "brightness_"}
    processed_base_files = set(
        file.split("_", 1)[-1] for file in processed_files if any(file.startswith(prefix) for prefix in augmented_prefixes)
    )

    new_captions = []
    processed_count = 0

    # Process each image
    for caption_line in captions:
        image_name, caption = caption_line.strip().split("|")
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Image {image_name} not found in {image_dir}, skipping.")
            continue

        # Skip augmentation if the image has already been processed
        if image_name in processed_base_files:
            print(f"Image {image_name} already augmented, skipping.")
            continue

        try:
            # Open original image
            original_image = Image.open(image_path).convert("RGB")
            image_array = np.array(original_image)

            # Save original image with prefix
            original_image_name = f"original_{image_name}"
            original_image.save(os.path.join(output_dir, original_image_name))
            new_captions.append(f"{original_image_name}|{caption}")

            # Save grainy image
            grainy_image = add_grain(image_array)
            grainy_image_name = f"grainy_{image_name}"
            Image.fromarray(grainy_image).save(os.path.join(output_dir, grainy_image_name))
            new_captions.append(f"{grainy_image_name}|{caption}")

            # Save inverted image
            inverted_image = invert_colors(original_image)
            inverted_image_name = f"inverted_{image_name}"
            inverted_image.save(os.path.join(output_dir, inverted_image_name))
            new_captions.append(f"{inverted_image_name}|{caption}")

            # Save brightness-adjusted image
            brightness_image = adjust_brightness(original_image)
            brightness_image_name = f"brightness_{image_name}"
            brightness_image.save(os.path.join(output_dir, brightness_image_name))
            new_captions.append(f"{brightness_image_name}|{caption}")

            processed_count += 1
            if processed_count % 100 == 0:  # Log progress every 100 images
                print(f"Processed {processed_count} images so far...")

            # Sleep after processing to avoid overheating
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue

    # Append new captions to the captions file
    captions_output_path = os.path.join(output_dir, "captions_augmented.txt")
    with open(captions_output_path, "a", encoding="utf-8") as f:
        f.write("\n".join(new_captions) + "\n")

    print(f"Processed {processed_count} new images. Augmented data saved to {output_dir}")

if __name__ == "__main__":
    # Specify the input and output directories
    input_image_dir = r"C:\Users\Nischal\Downloads\segmentation\captioned_images"  # Replace with your image folder path
    input_captions_file = r"C:\Users\Nischal\Downloads\segmentation\output.txt"  # Replace with your captions file
    output_directory = r"C:\Users\Nischal\Downloads\segmentation\augmentation"  # Replace with desired output path

    save_images_with_captions(input_image_dir, input_captions_file, output_directory, sleep_time=2)
