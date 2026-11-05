import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


SOURCE_DIR = Path("")
# Destination folder will be created if it doesn't exit
DESTINATION_DIR = Path("")

CLASS_MAP = {
    "background": 0,
    "window": 1,
    "car": 2,
    "vegetation": 3,
    "street-lamps": 4
}

#Format: (R, G, B) -> class_id
RGB_TO_ID_MAP = {
    (0, 0, 0): 0,  # background
    (176, 32, 128): 1,  # window
    (32, 128, 192): 0,  # car
    (144, 32, 192): 0,  # vegetation
    (4, 55, 231): 0  # street-lamps
}

def convert_rgb_mask_to_id_mask(rgb_mask):
    """
    Converts a 3-channel RGB segmentation mask (numpy array) to a 1-channel ID mask.
    Args: rgb_mask (np.array): The input RGB mask image as a NumPy array.
    Returns: np.array: A 2D NumPy array where each pixel value is a class ID.
    """
    # Ensure the input is in RGB format (OpenCV reads as BGR by default)
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)

    height, width, _ = rgb_mask.shape
    id_mask = np.zeros((height, width), dtype=np.uint8)

    # This is a more efficient, vectorized approach than iterating pixel by pixel
    for color, class_id in RGB_TO_ID_MAP.items():
        mask = np.all(rgb_mask == np.array(color).reshape(1, 1, 3), axis=2)
        id_mask[mask] = class_id

    return id_mask


def main():
    print("Starting mask conversion process...")

    if not SOURCE_DIR.exists():
        print(f"Error: Source directory not found at '{SOURCE_DIR}'")
        print("Please update the SOURCE_DIR variable in the script.")
        return

    # Create the destination directory if it doesn't exist
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Source directory:      '{SOURCE_DIR}'")
    print(f"Destination directory: '{DESTINATION_DIR}'")

    # Find all .png files in the source directory
    mask_files = list(SOURCE_DIR.glob("*.png"))

    if not mask_files:
        print(f"No .png files found in '{SOURCE_DIR}'. Please check the path.")
        return

    print(f"Found {len(mask_files)} masks to convert.")

    # Process each mask file with a progress bar
    for mask_path in tqdm(mask_files, desc="Converting masks"):
        try:
            # Read the original RGB mask
            rgb_mask_image = cv2.imread(str(mask_path))

            if rgb_mask_image is None:
                print(f"\nWarning: Could not read file {mask_path.name}. Skipping.")
                continue

            # Convert it to an ID mask
            id_mask_image = convert_rgb_mask_to_id_mask(rgb_mask_image)

            # Define the output path
            output_path = DESTINATION_DIR / mask_path.name

            # Save the new single-channel ID mask
            # cv2.imwrite will save a 2D array as a grayscale image
            cv2.imwrite(str(output_path), id_mask_image)

        except Exception as e:
            print(f"\nError processing {mask_path.name}: {e}")

    print("\nConversion complete!")
    print(f"All converted masks have been saved to '{DESTINATION_DIR}'")


if __name__ == "__main__":
    main()
