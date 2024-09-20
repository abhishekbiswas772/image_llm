import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

img_height = 1024
img_width = 1024
row_count = 3  # Set to 3 to create a 3x3 grid

def create_panoramic_view(query_image_path: str, retrieved_images: list, output_image_path: str) -> dict:
    """
    Creates a 3x3 panoramic view image from a list of images, saves the combined image, and returns a dictionary mapping
    indices to retrieved image file names.
    """

    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count

    # Create a blank panoramic image (3x3 grid of images)
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)

    # Create and resize the query image with a blue border
    query_image_null = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)
    query_image = Image.open(query_image_path).convert("RGB")
    query_array = np.array(query_image)[:, :, ::-1]  # Convert from RGB to BGR (OpenCV format)
    resized_image = cv2.resize(query_array, (img_width, img_height))

    # Add a blue border around the query image
    border_size = 10
    blue = (255, 0, 0)  # Blue color in BGR
    bordered_query_image = cv2.copyMakeBorder(
        resized_image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=blue,
    )

    # Place the query image at the center of the grid
    query_image_null[img_height * 2: img_height * 3, 0:img_width] = cv2.resize(bordered_query_image,
                                                                               (img_width, img_height))

    # Add the label "query" below the query image
    text = "query"
    font_scale = 1
    font_thickness = 2
    text_org = (10, img_height * 3 + 30)
    cv2.putText(
        query_image_null,
        text,
        text_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        blue,
        font_thickness,
        cv2.LINE_AA,
    )

    # Dictionary to map indices to image file names
    image_index_dict = {}

    # Combine the retrieved images into the panoramic view
    retrieved_imgs = [np.array(Image.open(img).convert("RGB"))[:, :, ::-1] for img in retrieved_images]

    for i, image in enumerate(retrieved_imgs):
        image = cv2.resize(image, (img_width - 4, img_height - 4))
        row = i // row_count
        col = i % row_count
        start_row = row * img_height
        start_col = col * img_width

        # Ensure that the start_row and start_col are within bounds
        if start_row + img_height > panoramic_image.shape[0] or start_col + img_width > panoramic_image.shape[1]:
            print(f"Skipping image at index {i} due to size mismatch.")
            continue

        # Add a black border around each retrieved image
        border_size = 2
        bordered_image = cv2.copyMakeBorder(
            image,
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # Place the image in the panoramic view
        panoramic_image[start_row: start_row + img_height, start_col: start_col + img_width] = bordered_image

        # Add red index numbers to each image
        text = str(i)
        org = (start_col + 50, start_row + 30)
        cv2.putText(
            panoramic_image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Add to the dictionary (index -> image filename)
        image_index_dict[i] = os.path.basename(retrieved_images[i])

    # Combine the query image with the panoramic view
    final_image = np.hstack([query_image_null, panoramic_image])

    # Save the final panoramic image
    cv2.imwrite(output_image_path, final_image)

    # Return the dictionary with indices and image file names
    return image_index_dict

def text_to_image(text, output_image_path, font_size=60, image_size=(300, 300), background_color=(255, 255, 255),
                  text_color=(0, 0, 0)):
    # Create a blank image with the given size and background color
    image = Image.new('RGB', image_size, color=background_color)

    # Initialize drawing context
    draw = ImageDraw.Draw(image)

    # Load a font (optional). If you don't have a font file, you can use a default one.
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Adjust to any available font path
    except IOError:
        font = ImageFont.load_default()

    # Calculate text bounding box and size using font.getbbox
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Center the text on the image
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    # Add text to the image
    draw.text(position, text, fill=text_color, font=font)

    # Save the image
    image.save(output_image_path)
    print(f"Image saved at {output_image_path}")
    return output_image_path



import numpy as np
from PIL import Image
import cv2
import os

img_height = 1024
img_width = 1024

def create_panoramic_view_t2i(retrieved_images: list, output_image_path: str) -> dict:
    """
    Creates a panoramic view image from a list of images, saves the combined image, and returns a dictionary mapping
    indices to retrieved image file names.
    """

    # Calculate row and column count based on the number of images
    total_images = len(retrieved_images)
    row_count = int(np.ceil(np.sqrt(total_images)))  # Dynamically determine the row/col count based on image count
    col_count = row_count  # Make it a square grid

    # Create the blank panoramic image
    panoramic_width = img_width * col_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)

    # Dictionary to map indices to image file names
    image_index_dict = {}

    # Combine the retrieved images into the panoramic view
    retrieved_imgs = [np.array(Image.open(img).convert("RGB"))[:, :, ::-1] for img in retrieved_images]

    for i, image in enumerate(retrieved_imgs):
        image = cv2.resize(image, (img_width - 4, img_height - 4))
        row = i // col_count
        col = i % col_count
        start_row = row * img_height
        start_col = col * img_width

        # Ensure that the start_row and start_col are within bounds
        if start_row + img_height > panoramic_image.shape[0] or start_col + img_width > panoramic_image.shape[1]:
            print(f"Skipping image at index {i} due to size mismatch.")
            continue

        # Add a black border around each retrieved image
        border_size = 2
        bordered_image = cv2.copyMakeBorder(
            image,
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # Place the image in the panoramic view
        panoramic_image[start_row: start_row + img_height, start_col: start_col + img_width] = bordered_image

        # Add red index numbers to each image
        text = str(i)
        org = (start_col + 50, start_row + 30)
        cv2.putText(
            panoramic_image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Add to the dictionary (index -> image filename)
        image_index_dict[i] = os.path.basename(retrieved_images[i])

    # Save the final panoramic image
    cv2.imwrite(output_image_path, panoramic_image)

    # Return the dictionary with indices and image file names
    return image_index_dict
