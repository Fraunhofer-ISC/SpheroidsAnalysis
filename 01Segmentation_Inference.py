import os
import csv
import math
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

def run_yolo_seg_inference(
    model_path,
    input_folder,
    output_folder,
    confidence_threshold=0.25,
    iou_threshold=0.45
):
    """
    Runs YOLO segmentation inference and saves:
        - Per-instance full-size binary masks in <output_folder>/inferenceSeg/mask_crops/
          sized exactly as the *original* input image.
        - A CSV with area, perimeter, circularity (seg_circularity.csv).
    """
    model = YOLO(model_path)

    # Run segmentation inference on the images in the input folder
    results = model.predict(
        source=input_folder,
        conf=confidence_threshold,
        iou=iou_threshold,
        task="segment",
        save=True,
        save_txt=False,
        project=output_folder,
        name="inferenceSeg",
        exist_ok=True
    )

    # Create subfolder for binary masks
    masks_dir = Path(output_folder) / "inferenceSeg" / "mask_crops"
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_data_dict = {}

    POSSIBLE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    for result in results:
        inference_path = Path(result.path)
        stem = inference_path.stem

        # Try to find the original image in input_folder by matching the same stem
        original_img_path = None
        for ext in POSSIBLE_EXTS:
            candidate = Path(input_folder) / f"{stem}{ext}"
            if candidate.exists():
                original_img_path = candidate
                break

        if not original_img_path:
            original_img_path = inference_path

        with Image.open(original_img_path) as tmp_img:
            orig_w, orig_h = tmp_img.size

        image_name = original_img_path.name

        if not result.masks or len(result.masks.data) == 0:
            mask_data_dict[image_name] = []
            continue

        per_image_data = []

        for mask_i, mask_array in enumerate(result.masks.data):
            mask_2d = mask_array.cpu().numpy()
            cls_id = int(result.boxes.cls[mask_i])

            mask_uint8 = (mask_2d * 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area = 0.0
            total_perimeter = 0.0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                total_area += area
                total_perimeter += perimeter

            circularity = 0.0 if total_perimeter == 0 else (4.0 * math.pi * total_area) / (total_perimeter ** 2)

            per_image_data.append({
                "class_id": cls_id,
                "area": total_area,
                "perimeter": total_perimeter,
                "circularity": circularity
            })

            # Resize the mask to match the original image size and save as a full-size binary mask.
            mask_resized = cv2.resize(mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            full_mask_img = Image.fromarray(mask_resized).convert("L")

            out_mask_name = f"{stem}_mask{mask_i}_class{cls_id}_binary.png"
            full_mask_img.save(masks_dir / out_mask_name)

        mask_data_dict[image_name] = per_image_data

    # Save CSV with separate columns for class 0 and class 1 area, etc.
    csv_path = Path(output_folder) / "inferenceSeg" / "seg_circularity.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "area_class_0", "area_class_1", "perimeter", "circularity"])

        for image_name, obj_list in mask_data_dict.items():
            area_class_0 = 0.0
            area_class_1 = 0.0
            total_perimeter = 0.0
            total_circularity = 0.0

            for obj_data in obj_list:
                if obj_data["class_id"] == 0:
                    area_class_0 += obj_data["area"]
                elif obj_data["class_id"] == 1:
                    area_class_1 += obj_data["area"]

                total_perimeter += obj_data["perimeter"]
                total_circularity += obj_data["circularity"]

            writer.writerow([
                image_name,
                area_class_0,
                area_class_1,
                total_perimeter,
                total_circularity
            ])

    return mask_data_dict

def main():
    # Get the current working directory
    working_dir = os.getcwd()

    # MODEL_PATH is "Segment.pt" in the current working directory
    MODEL_PATH = os.path.join(working_dir, "Segment.pt")
    
    # The inference image(s) are in the "output" folder in the current working directory.
    INPUT_FOLDER = os.path.join(working_dir, "output")
    
    # Set the output folder for segmentation results.
    OUTPUT_FOLDER = os.path.join(working_dir, "seg_output")

    seg_data = run_yolo_seg_inference(
        model_path=MODEL_PATH,
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        confidence_threshold=0.25,
        iou_threshold=0.45
    )

    for image_name, objs in seg_data.items():
        print(f"\nIMAGE: {image_name}")
        for i, obj in enumerate(objs, start=1):
            print(
                f"  Mask {i}: class={obj['class_id']}, "
                f"area={obj['area']:.2f}, perimeter={obj['perimeter']:.2f}, "
                f"circularity={obj['circularity']:.3f}"
            )

if __name__ == "__main__":
    main()
