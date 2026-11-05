import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Update these paths to point to your ID masks and where you want the RGB masks to be saved
SOURCE_DIR = Path("")
DESTINATION_DIR = Path("")

# Format: class_id -> (R, G, B)
# in the 2 class data 0 is background, 255 is window
ID_TO_RGB_MAP = {
    0: (0, 0, 0),  # background
    64: (176, 32, 128), # window
    191: (144, 32, 192), # vegetation
    128: (32, 128, 192),  # car
    255: (4, 55, 231),  #  streetlamp
}


def convert_id_mask_to_rgb_mask(id_mask):
    height, width = id_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in ID_TO_RGB_MAP.items():
        mask_indices = (id_mask == class_id)
        rgb_mask[mask_indices] = color

    return rgb_mask


def main():
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory not found at '{SOURCE_DIR}'")
        return

    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    mask_files = list(SOURCE_DIR.glob("*.png"))

    print(f"Found {len(mask_files)} masks. Starting analysis and conversion...")

    for mask_path in tqdm(mask_files, desc="Processing"):
        try:
            #load as grayscale
            id_mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if id_mask_image is None:
                print(f"\n[SKIP] Could not read {mask_path.name}")
                continue
            # see what pixel values there are
            unique_values = np.unique(id_mask_image)

            # Only print if it looks suspicious (values outside 0-4)
            # or you can remove this 'if' to print for every single file
            if np.any(unique_values > 4):
                print(f"\n[WARN] {mask_path.name}: Found unexpected values: {unique_values}")
            # else:
            #     print(f"File: {mask_path.name} | IDs: {unique_values}")

            # Convert to RGB
            rgb_mask_image = convert_id_mask_to_rgb_mask(id_mask_image)

            # Save (Convert RGB -> BGR for OpenCV)
            bgr_output_image = cv2.cvtColor(rgb_mask_image, cv2.COLOR_RGB2BGR)
            output_path = DESTINATION_DIR / mask_path.name
            cv2.imwrite(str(output_path), bgr_output_image)

        except Exception as e:
            print(f"\n[ERROR] {mask_path.name}: {e}")

    print("\nProcess complete.")


if __name__ == "__main__":
    main()