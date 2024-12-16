import os

def check_augmentation_alignment(augmented_dir, augmented_caption_file):
    # Load captions from the file
    with open(augmented_caption_file, "r", encoding="utf-8") as f:
        caption_lines = [line.strip() for line in f.readlines()]

    # Build a dictionary of {image_name: caption} from the caption file
    caption_dict = {}
    for line in caption_lines:
        try:
            image_name, caption = line.split("|")
            caption_dict[image_name] = caption
        except ValueError:
            print(f"Invalid line format in captions file: {line}")
            continue

    # Get a list of augmented image files in the directory
    augmented_files = set(os.listdir(augmented_dir))

    # Save augmented files to a text file
    augmented_files_path = os.path.join(augmented_dir, "augmented_files_list.txt")
    with open(augmented_files_path, "w", encoding="utf-8") as file:
        for augmented_file in augmented_files:
            file.write(f"{augmented_file}\n")

    # Check alignment
    missing_files = []
    caption_only_files = []
    mismatched_captions = []

    # Check if any caption references a missing file
    for image_name in caption_dict:
        if image_name not in augmented_files:
            missing_files.append(image_name)

    # Check if any file in the directory is missing in the captions
    for file in augmented_files:
        # Skip the captions file itself
        if file == os.path.basename(augmented_caption_file):
            continue
        if file not in caption_dict:
            caption_only_files.append(file)

    # Report results
    if missing_files:
        print("\nThe following image names from the captions file are missing in the augmented directory:")
        for file in missing_files:
            print(f"- {file}")
    else:
        print("\nAll image names from the captions file are present in the augmented directory.")

    if caption_only_files:
        print("\nThe following files in the augmented directory are missing from the captions file:")
        for file in caption_only_files:
            print(f"- {file}")
    else:
        print("\nAll files in the augmented directory are referenced in the captions file.")

if __name__ == "__main__":
    # Specify paths to the augmented directory and captions file
    augmented_directory = r"C:\Users\Nischal\Downloads\segmentation\augmentation"  # Augmented images directory
    augmented_captions_file = os.path.join(augmented_directory, "captions_augmented.txt")  # Augmented captions file

    check_augmentation_alignment(augmented_directory, augmented_captions_file)
