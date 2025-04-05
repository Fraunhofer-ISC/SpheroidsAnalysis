# Spheroid Image Processing Pipeline

This repository contains a series of Python scripts that work after one another to process images of spheroidal tissue models, perform segmentation using a YOLO model, post-process the segmentation masks, and analyze image darkness scores. The scripts are numbered in the order in which they should be executed.

## Table of Contents

- [Overview](#overview)
- [File Descriptions](#file-descriptions)
  - [00PNGExport.py](#00pngexportpy)
  - [01Segmentation_Inference.py](#01segmentation_inferencepy)
  - [02Masks.py](#02maskspy)
  - [03DarknessScore.py](#03darknessscorepy)
- [Setup and Installation](#setup-and-installation)
- [Usage Instructions](#usage-instructions)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

This pipeline is designed to process high-bit TIFF images of spheroids, convert them to PNG format, perform YOLO-based segmentation inference, generate composite images using segmentation masks, and finally analyze the darkness score (i.e., average pixel intensity) of non-masked areas.

The workflow is as follows:

1. **PNG Export (00PNGExport.py):**  
   Inspects and processes a TIFF image from the `Acquisition` folder, converting its final layer to an 8-bit grayscale PNG. The result is saved in the `output` folder.

2. **Segmentation Inference (01Segmentation_Inference.py):**  
   Loads a YOLO segmentation model (`Segment.pt`) from the working directory and runs inference on PNG images from the `output` folder. The segmentation outputs (binary masks and object statistics CSV) are saved in the `seg_output` folder.

3. **Mask Post-Processing (02Masks.py):**  
   Combines the raw PNG image (from the `output` folder) with the corresponding binary segmentation mask (from `seg_output/inferenceSeg/mask_crops/1`). The composite images—where background areas are replaced with green—are saved in a new folder `RawMasks` within the mask folder.

4. **Darkness Score Analysis (03DarknessScore.py):**  
   Processes the composite images from `seg_output/inferenceSeg/RawMasks` to calculate darkness scores (average pixel intensity) for non-green areas. It also generates intermediate images (green mask, inverted mask, grayscale object image, histogram) and saves the analysis results as a CSV file.

## File Descriptions

### 00PNGExport.py

- **Purpose:**  
  - Inspects a TIFF image and prints its properties (format, mode, size, number of frames, pixel range).
  - Processes the TIFF by converting its final layer from 16-bit to an 8-bit grayscale image.
  - Exports the processed image as a PNG to the `output` folder.
- **Input:**  
  - A TIFF image (e.g., `Spheroid.tiff`) located in the `Acquisition` folder.
- **Output:**  
  - Processed PNG image saved in the `output` folder.

### 01Segmentation_Inference.py

- **Purpose:**  
  - Loads the YOLO segmentation model (`Segment.pt`) from the current working directory.
  - Runs segmentation inference on images located in the `output` folder.
  - Saves segmentation outputs including binary masks and a CSV (`seg_circularity.csv`) with area, perimeter, and circularity data.
- **Input:**  
  - PNG images from the `output` folder.
  - Model file `Segment.pt` (located in the working directory).
- **Output:**  
  - Segmentation outputs saved in the `seg_output` folder, including binary mask crops in `seg_output/inferenceSeg/mask_crops`.

### 02Masks.py

- **Purpose:**  
  - Combines the raw PNG image from the `output` folder with its corresponding binary segmentation mask from `seg_output/inferenceSeg/mask_crops/1`.
  - Creates composite images where pixels corresponding to the mask remain as in the raw image, and the rest are replaced with a green background.
  - Saves the composite images in a new folder named `RawMasks` within `seg_output/inferenceSeg/mask_crops/1`.
- **Input:**  
  - Raw PNG images from the `output` folder.
  - Binary mask images from `seg_output/inferenceSeg/mask_crops/1`.
- **Output:**  
  - Composite images saved in `seg_output/inferenceSeg/mask_crops/1/RawMasks`.

### 03DarknessScore.py

- **Purpose:**  
  - Processes the composite images (from `seg_output/inferenceSeg/RawMasks`) to compute a darkness score (average grayscale intensity) for the non-green areas.
  - Generates intermediate output images: the green mask, inverted mask, a grayscale image of the object, and a histogram of pixel intensities.
  - Saves the analysis results to a CSV file (`average_intensities.csv`).
- **Input:**  
  - Composite images (PNG) from `seg_output/inferenceSeg/RawMasks`.
- **Output:**  
  - Intermediate processing images and a CSV file with darkness scores, all saved in a new subfolder (`output`) under the processed folder.

> **Note:** You may need to adjust the folder path in `03DarknessScore.py` to ensure it points to the correct location relative to your working directory.

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_folder>
