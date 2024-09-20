import cv2
import numpy as np
import torch
from PIL import Image
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import easyocr

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = r"sam_vit_h_4b8939.pth"

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def initialize_model():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    return SamAutomaticMaskGenerator(sam)

def load_image(image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb

def generate_masks(mask_generator, image_rgb):
    return mask_generator.generate(image_rgb)

def process_masks(masks, image_bgr):
    results = []
    for mask_data in masks:
        mask = mask_data['segmentation']  # Extract the mask from the dictionary
        segmented_image_uint8 = mask.astype(np.uint8) * 255
        segmented_image_resized = cv2.resize(segmented_image_uint8, (image_bgr.shape[1], image_bgr.shape[0]))

        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
            segmented_image_resized = cv2.cvtColor(segmented_image_resized, cv2.COLOR_GRAY2BGR)

        segmented_image_result = cv2.bitwise_and(segmented_image_resized, image_bgr)
        results.append(segmented_image_result)

    return results

def save_images_to_folder(results, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_image_paths = []
    for i, image in enumerate(results):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        non_black_pixels = np.where(gray_image != 0)

        if non_black_pixels[0].size > 0 and non_black_pixels[1].size > 0:
            top_left_y = np.min(non_black_pixels[0])
            bottom_right_y = np.max(non_black_pixels[0])
            top_left_x = np.min(non_black_pixels[1])
            bottom_right_x = np.max(non_black_pixels[1])

            segmented_image_result = image[top_left_y:bottom_right_y + 1, top_left_x:bottom_right_x + 1]
            mask = np.all(segmented_image_result == [0, 0, 0], axis=-1)
            segmented_image_result[mask] = [255, 255, 255]

            image_rgb = cv2.cvtColor(segmented_image_result, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            image_filename = os.path.join(output_folder, f'segmented_image_{i + 1}.png')
            pil_image.save(image_filename)

            saved_image_paths.append(image_filename)

    return saved_image_paths

def do_segmentation(image_path):
    print("Starting segmentation process...")
    mask_generator = initialize_model()
    image_bgr, image_rgb = load_image(image_path)
    masks = generate_masks(mask_generator, image_rgb)

    segmented_images = process_masks(masks, image_bgr)

    # Dynamically create output folder based on the image name (without extension)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = f'segmented_{image_name}'

    # Save images to the folder
    saved_images = save_images_to_folder(segmented_images, output_folder=output_folder)
    print(f"Segmentation completed. Images saved in {output_folder}")
    return saved_images

def detect_text_and_delete(image_path):
    # Read text from the image
    result = reader.readtext(image_path)

    # Check if any text was detected
    if any(detection[1].strip() for detection in result):
        print(f"Deleting text image: {image_path}")
        os.remove(image_path)
        return True
    return False

def delete_text_images(root_folder):
    # Traverse through all subfolders and files
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            image_path = os.path.join(subdir, file)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Image file formats
                try:
                    detect_text_and_delete(image_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

def fetch_segments(image_path):
    # Perform segmentation for the given image
    do_segmentation(image_path)

    # Check and delete images that contain mostly text
    delete_text_images(f'segmented_{os.path.splitext(os.path.basename(image_path))[0]}')
