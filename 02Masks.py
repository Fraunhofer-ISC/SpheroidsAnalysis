from PIL import Image
import os
import numpy as np

# Get the working directory
working_dir = os.getcwd()

# Define the folder paths based on the working directory
raw_folder = os.path.join(working_dir, "output")
mask_folder = os.path.join(working_dir, "seg_output", "inferenceSeg", "mask_crops", "1")
output_folder = os.path.join(mask_folder, "RawMasks")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get lists of raw and mask images using first 3 characters of filename as key
raw_images = {img[:3]: os.path.join(raw_folder, img)
              for img in os.listdir(raw_folder)
              if img.lower().endswith(('.png', '.jpg', '.jpeg'))}

mask_images = {img[:3]: os.path.join(mask_folder, img)
               for img in os.listdir(mask_folder)
               if img.lower().endswith(('.png', '.jpg', '.jpeg'))}

# Process each matching raw and mask pair
for key in raw_images.keys() & mask_images.keys():
    raw_path = raw_images[key]
    mask_path = mask_images[key]

    # Open the binary mask and raw image
    bin_image = Image.open(mask_path).convert('L')  # Convert mask to grayscale
    raw_image = Image.open(raw_path)  # Open raw image

    # Normalize the binary mask: threshold to get a binary image (0 or 255)
    bin_image = bin_image.point(lambda p: 255 if p > 127 else 0)

    # Normalize the raw image
    raw_array = np.array(raw_image, dtype=np.float32)
    raw_array -= raw_array.min()  # Shift minimum to 0
    if raw_array.max() != 0:
        raw_array /= raw_array.max()  # Normalize to [0, 1]
    raw_array *= 255  # Scale to [0, 255]
    raw_array = raw_array.astype(np.uint8)

    # If the raw image is grayscale (2D array), convert using mode 'L'; 
    # otherwise, use the array as is.
    if raw_array.ndim == 2:
        raw_image = Image.fromarray(raw_array, mode='L').convert('RGB')
    else:
        raw_image = Image.fromarray(raw_array).convert('RGB')

    # Ensure sizes match
    if bin_image.size != raw_image.size:
        raise ValueError(f"Size mismatch: {os.path.basename(mask_path)} and {os.path.basename(raw_path)}")

    # Prepare pixel access objects
    bin_pixels = bin_image.load()
    raw_pixels = raw_image.load()

    # Create a new output image
    output_image = Image.new('RGB', raw_image.size)
    output_pixels = output_image.load()

    # Define green color for background
    green = (0, 255, 0)

    # Process each pixel: if the mask pixel is white, use the raw pixel; else, use green.
    for y in range(raw_image.height):
        for x in range(raw_image.width):
            if bin_pixels[x, y] == 255:  # Foreground pixel from mask
                output_pixels[x, y] = raw_pixels[x, y]
            else:  # Background pixel replaced with green
                output_pixels[x, y] = green

    # Save the output image
    output_path = os.path.join(output_folder, f"{key}_output.png")
    output_image.save(output_path)
    print(f"Output saved for {key} at {output_path}")

print("Processing complete.")
