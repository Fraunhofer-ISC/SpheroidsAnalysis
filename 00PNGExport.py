from PIL import Image, ImageSequence
import numpy as np
import os

def inspect_tiff(file_path):
    """
    Opens the TIFF file and prints its format, mode, size, number of frames,
    and the pixel data type and range for the first frame.
    """
    with Image.open(file_path) as img:
        print("Format:", img.format)
        print("Mode:", img.mode)
        print("Size:", img.size)
        n_frames = getattr(img, "n_frames", 1)
        print("Number of frames:", n_frames)
        
        # For inspection, convert the first frame to I;16 (16-bit integer mode)
        frame = img.convert("I;16")
        arr = np.array(frame)
        print("Pixel data type:", arr.dtype)
        print("Min pixel value:", arr.min(), "Max pixel value:", arr.max())

def process_tiff(file_path, output_folder):
    """
    Processes the TIFF file:
      - If the image is in 16-bit mode ("I;16"), scales its pixel values to 8-bit.
      - Converts the image to grayscale ("L") and then to RGB.
      - Saves the processed image as a PNG in the specified output folder.
    """
    with Image.open(file_path) as img:
        # Check for multi-frame images if needed; here we just process the first frame.
        if img.mode == "I;16":
            print("Processing a 16-bit image; scaling down to 8-bit.")
            # Scale pixel values from 16-bit (0-65535) to 8-bit (0-255) by dividing by 256.
            img_8bit = img.point(lambda i: i / 256)
            # Convert to grayscale
            img_8bit = img_8bit.convert("L")
        else:
            print("Processing a non-16-bit image.")
            # If not 16-bit, simply convert to grayscale.
            img_8bit = img.convert("L")
        
        # Convert grayscale to RGB for PNG compatibility
        final_img = img_8bit.convert("RGB")
        
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Save the processed image as a PNG file in the output folder.
        output_path = os.path.join(output_folder, 
                                   os.path.splitext(os.path.basename(file_path))[0] + "_processed.png")
        final_img.save(output_path, "PNG")
        print("Processed image saved to", output_path)

# Full execution: inspect and then process the TIFF file in the Acquisition folder
working_dir = os.getcwd()
acquisition_folder = os.path.join(working_dir, "Acquisition")
output_folder = os.path.join(working_dir, "output")
file_path = os.path.join(acquisition_folder, "Spheroid.tiff")

print("Inspecting TIFF file:")
inspect_tiff(file_path)
print("\nProcessing TIFF file:")
process_tiff(file_path, output_folder)
