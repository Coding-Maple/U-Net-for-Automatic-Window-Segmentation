import cv2
import numpy as np
import os
import concurrent.futures
import time


INPUT_ROOT_DIR = 'source'
OUTPUT_ROOT_DIR = 'output'

MAX_WORKERS = os.cpu_count() # Use number of CPU cores for optimal threading

# --- Grayscale Conversion Function (ITU-R BT.709) ---
def convert_to_grayscale_bt709(input_image_path, output_image_path):
    """
    Converts an RGB image to grayscale using the ITU-R BT.709 standard.

    Args:
        input_image_path (str): Full path to the input RGB image.
        output_image_path (str): Full path where the grayscale image will be saved.
    """
    try:
        # Read the image using OpenCV in BGR format
        rgb_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

        if rgb_image is None:
            print(f"Warning: Could not read image {input_image_path}. Skipping.")
            return

        # Convert BGR to RGB for correct application of BT.709 formula
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Convert to float for precise calculation and normalize to [0, 1]
        # Because doing the multiplication on int values might lead to precision loss
        rgb_float = rgb_image.astype(np.float32) / 255.0

        # Apply the ITU-R BT.709 formula
        # Y' = 0.2126 * R + 0.7152 * G + 0.0722 * B
        grayscale_image = (
            0.2126 * rgb_float[:, :, 0] +  # Red channel
            0.7152 * rgb_float[:, :, 1] +  # Green channel
            0.0722 * rgb_float[:, :, 2]    # Blue channel
        )

        # Scale back to 0-255 and convert to uint8
        grayscale_image = np.clip(grayscale_image * 255, 0, 255).astype(np.uint8)

        # Save the grayscale image
        cv2.imwrite(output_image_path, grayscale_image)

    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")

if __name__ == "__main__":
    print(f"--- Starting Image Processing ---")
    print(f"Input Root Directory: {INPUT_ROOT_DIR}")
    print(f"Output Root Directory: {OUTPUT_ROOT_DIR}")
    print(f"Max Workers (Threads): {MAX_WORKERS}")

    # Check if the input root directory exists
    if not os.path.isdir(INPUT_ROOT_DIR):
        print(f"Error: The specified INPUT_ROOT_DIR '{INPUT_ROOT_DIR}' does not exist.")
        exit()
    # Check if the output root directory exists
    if not os.path.isdir(OUTPUT_ROOT_DIR):
        print(f"Error: The specified OUTPUT_ROOT_DIR '{OUTPUT_ROOT_DIR}' does not exist.")
        exit()

    # Get a list of subdirectories within the input root directory
    subfolders = [d for d in os.listdir(INPUT_ROOT_DIR) if os.path.isdir(os.path.join(INPUT_ROOT_DIR, d))]

    if not subfolders:
        print(f"No subfolders found in '{INPUT_ROOT_DIR}'. Please ensure your 'test', 'train', 'valid' folders are there.")
    else:
        print(f"\nFound subfolders to process: {', '.join(subfolders)}")
        total_images_processed = 0
        start_time = time.time()

        # Iterate through each subfolder
        for subfolder_name in subfolders:
            current_input_dir = os.path.join(INPUT_ROOT_DIR, subfolder_name)
            current_output_dir = os.path.join(OUTPUT_ROOT_DIR, subfolder_name)

            print(f"\nProcessing images in '{current_input_dir}'...")

            # Get list of all image files to process in the current subfolder
            image_files = [f for f in os.listdir(current_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            print(f"  Found {len(image_files)} images in '{subfolder_name}'.")
            total_images_processed += len(image_files)

            # Process images using ThreadPoolExecutor for the current subfolder
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_image = {
                    executor.submit(
                        convert_to_grayscale_bt709,
                        os.path.join(current_input_dir, filename),
                        os.path.join(current_output_dir, f"{filename}")
                    ): filename
                    for filename in image_files
                }

                for future in concurrent.futures.as_completed(future_to_image):
                    filename = future_to_image[future]
                    try:
                        future.result() # This will re-raise any exception from the worker thread
                    except Exception as exc:
                        print(f'  Error processing {filename} in {subfolder_name}: {exc}')

        end_time = time.time()
        print(f"\n--- Image processing complete ---")
        print(f"Total images processed across all subfolders: {total_images_processed}")
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        if total_images_processed > 0:
            print(f"Average time per image: {(end_time - start_time) / total_images_processed:.4f} seconds")
