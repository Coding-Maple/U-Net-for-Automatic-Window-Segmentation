import os
from PIL import Image
from tqdm import tqdm
import numpy as np


def create_patches(image_path, mask_path, output_img_dir, output_mask_dir, patch_size=1024, overlap=512):
    """
    Splits an image and mask into patches, but only saves them if the mask patch
    contains the 'window' label.
    """
    try:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Skipping this file pair.")
        return

    if image.size != mask.size:
        print(f"Error: Image {os.path.basename(image_path)} and mask have different sizes. Skipping.")
        return

    width, height = image.size
    stride = patch_size - overlap
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    patch_number = 0
    saved_patches = 0

    # window: 176, 32, 128 color map
    WINDOW_COLOR_RGB = np.array([176, 32, 128])
    # around 10% of the image should be a window
    MIN_PIXEL_THRESHOLD = 105000

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            left, top = x, y
            right, bottom = x + patch_size, y + patch_size

            if right > width or bottom > height:
                continue

            img_patch = image.crop((left, top, right, bottom))
            mask_patch = mask.crop((left, top, right, bottom))

            mask_patch_np = np.array(mask_patch)

            save_this_patch = False

            # Check if the mask is a 3-channel image (H, W, 3)
            if mask_patch_np.ndim == 3 and mask_patch_np.shape[2] == 3:

                # Find all pixels where all 3 channels match the WINDOW_COLOR_RGB
                matches = np.all(mask_patch_np == WINDOW_COLOR_RGB, axis=2)

                # Count the number of matching (True) pixels
                window_pixel_count = np.sum(matches)

                # Check if the count meets or exceeds the threshold
                if window_pixel_count >= MIN_PIXEL_THRESHOLD:
                    save_this_patch = True

            if save_this_patch:
                patch_filename = f"{base_filename}_patch_{patch_number}_{y}_{x}.png"
                img_patch_path = os.path.join(output_img_dir, patch_filename)
                mask_patch_path = os.path.join(output_mask_dir, patch_filename)

                img_patch.save(img_patch_path)
                mask_patch.save(mask_patch_path)
                saved_patches += 1

            patch_number += 1  # Always increment patch number

    #Print how many patches were saved out of the total generated
    print(f"For {base_filename}, saved {saved_patches} of {patch_number} generated patches (containing 'window').")


def process_all_images(raw_dir, ann_dir, output_img_dir, output_mask_dir, patch_size=1024, overlap=512):
    """
    Finds all image-mask pairs and processes them, saving to specified directories.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.png')]

    if not image_files:
        print(f"No .png images found in the directory: {raw_dir}")
        return

    print(f"Found {len(image_files)} images to process. Starting...")

    # Wrap the tqdm iterable in a list to get an accurate total
    for filename in tqdm(list(image_files), desc="Processing Images"):
        image_path = os.path.join(raw_dir, filename)
        mask_path = os.path.join(ann_dir, filename)

        if not os.path.exists(mask_path):
            print(f"\nWarning: Mask for {filename} not found in {ann_dir}. Skipping.")
            continue

        create_patches(image_path, mask_path, output_img_dir, output_mask_dir, patch_size, overlap)

    print("\nProcessing complete.")
    print(f"Patched images have been saved to: {output_img_dir}")
    print(f"Patched masks have been saved to: {output_mask_dir}")


if __name__ == "__main__":
    # Input directories
    RAW_IMAGE_DIR = "data/train_original"
    ANNOTATION_DIR = "data/train_original_masks"

    # Output directories
    OUTPUT_IMAGE_DIR = "data/patches/window_patches_train"
    OUTPUT_ANNOTATION_DIR = "data/patches/window_label_patches_train"

    # Patch generation parameters
    PATCH_SIZE = 1024
    OVERLAP = 512
    # --------------------

    # Validate that the input directories exist
    if not os.path.isdir(RAW_IMAGE_DIR):
        print(f"Error: Raw image directory not found at '{RAW_IMAGE_DIR}'")
        exit()
    if not os.path.isdir(ANNOTATION_DIR):
        print(f"Error: Annotation directory not found at '{ANNOTATION_DIR}'")
        exit()

    # Run the main processing function with the separate output directories
    process_all_images(
        raw_dir=RAW_IMAGE_DIR,
        ann_dir=ANNOTATION_DIR,
        output_img_dir=OUTPUT_IMAGE_DIR,
        output_mask_dir=OUTPUT_ANNOTATION_DIR,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP
    )