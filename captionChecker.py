import os
import shutil


# Function 1: Check if every line in the caption file follows "image|caption" rule and remove invalid lines
def clean_caption_file(caption_file):
    valid_lines = []
    with open(caption_file, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        # Check if the line follows the "image|caption" format and caption is non-empty
        parts = line.strip().split('|')
        if len(parts) == 2 and parts[1].strip():
            valid_lines.append(line.strip())
        else:
            print(f"Removing invalid line {idx + 1}: {line.strip()}")

    # Rewrite the caption file with valid lines only
    with open(caption_file, 'w') as f:
        f.write('\n'.join(valid_lines))


# Function 2: Check if all images in the directory have captions and move images without captions
def move_images_without_captions(image_directory, caption_file, no_caption_dir):
    # Ensure the no_caption_dir exists
    if not os.path.exists(no_caption_dir):
        os.makedirs(no_caption_dir)

    # Read captions from file into a set
    captioned_images = set()
    with open(caption_file, 'r') as f:
        for line in f:
            image_name, _ = line.strip().split('|', 1)
            captioned_images.add(image_name)

    # Check for images without captions
    for image_name in os.listdir(image_directory):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            if image_name not in captioned_images:
                print(f"Moving image without caption: {image_name}")
                shutil.move(
                    os.path.join(image_directory, image_name),
                    os.path.join(no_caption_dir, image_name)
                )


# Main function to run the process
def process_images_and_captions(image_directory, caption_file, no_caption_dir):
    print("Cleaning caption file...")
    clean_caption_file(caption_file)

    print("Checking and moving images without captions...")
    move_images_without_captions(image_directory, caption_file, no_caption_dir)

    print("Processing completed!")


if __name__ == '__main__':
    # Define directories and files
    image_directory = r"devanagari_characters"  # Image directory
    no_caption_directory = r"no_caption"  # Directory for images without captions
    caption_file = r"output.txt"  # Caption file

    # Run the process
    process_images_and_captions(image_directory, caption_file, no_caption_directory)
