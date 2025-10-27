import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate the darkness score and save intermediate steps
def calculate_darkness_and_save(image_path, output_folder, file_name):
    image = cv2.imread(image_path)
    
    # Convert the image to HSV color space to isolate the green background
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the green mask (threshold for green areas)
    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    
    # Invert the mask to focus on non-green pixels
    object_mask = cv2.bitwise_not(green_mask)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract pixel intensities of non-green areas
    object_pixels = gray[object_mask > 0]
    
    # Calculate the average intensity of the non-green pixels
    average_intensity = np.mean(object_pixels)
    
    # Save intermediate processing images
    cv2.imwrite(os.path.join(output_folder, f"{file_name}_green_mask.png"), green_mask)
    cv2.imwrite(os.path.join(output_folder, f"{file_name}_inverted_mask.png"), object_mask)
    cv2.imwrite(os.path.join(output_folder, f"{file_name}_gray_object.png"), object_pixels)
    
    return average_intensity, object_pixels

# Function to save the grayscale histogram for non-green pixels
def save_gray_histogram(object_pixels, output_folder, file_name):
    plt.figure()
    plt.title(f"Grayscale Histogram: {file_name}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.hist(object_pixels, bins=256, range=(0, 256), color='gray', alpha=0.7)
    histogram_path = os.path.join(output_folder, f"{file_name}_gray_histogram.png")
    plt.savefig(histogram_path)
    plt.close()

# Main function to process all images in a folder
def process_images_and_save(folder_path):
    output_folder = os.path.join(folder_path, "output")
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    output_data = []

    # Process each image in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):  # Process only PNG files
            image_path = os.path.join(folder_path, file_name)
            
            # Get the first three characters of the filename
            short_name = file_name[:3]
            
            # Calculate the average intensity and save intermediate images
            average_intensity, object_pixels = calculate_darkness_and_save(image_path, output_folder, short_name)
            
            # Save the grayscale histogram for the non-green pixels
            save_gray_histogram(object_pixels, output_folder, short_name)
            
            # Append the results to the output data
            output_data.append({"Image": short_name, "Average Intensity": round(average_intensity)})
    # Save results to a CSV file
    output_csv_path = os.path.join(output_folder, "average_intensities.csv")
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_path, index=False)

    print(f"Analysis complete. Results and histograms saved in {output_folder}")

# Example usage
folder_path = "C:\Dalia\Dalia\Spheroids\PANC1\Drug Screening\Experiments\Day10\RBTDS1B\PNG\inferenceSeg\RawMasks"  # Update this path to your working directory
process_images_and_save(folder_path)
