import os
import numpy as np
from PIL import Image
import cv2

# Function 1: Extract Mean Intensity (Grayscale Analysis)
def process_single_image(row_tuple):
    # Unpack tuple inputs
    index, row, image_folder = row_tuple
    img_name = row['image'] + '.jpeg'
    img_path = os.path.join(image_folder, img_name)
    
    if os.path.exists(img_path):
        try:
            # Open image and convert to grayscale ("L" mode)
            with Image.open(img_path).convert("L") as img:
                # Convert image to NumPy array
                img_array = np.array(img)
                # Filter out very dark pixels
                non_black = img_array[img_array > 10]
                # Compute mean intensity of meaningful pixels
                mean_intensity = non_black.mean() if len(non_black) > 0 else 0
                return {
                    "image": row['image'],
                    "level": row['level'],
                    "mean_intensity": mean_intensity
                }
        except Exception:
            return None
    return None

# Function 2: Extract Image Dimensions
def get_image_dims(row_tuple):
    # Unpack inputs
    index, row, image_folder = row_tuple
    img_name = str(row['image']) + '.jpeg'
    img_path = os.path.join(image_folder, img_name)
    
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                 # Return width, height, and aspect ratio
                return {'width': w, 'height': h, 'aspect_ratio': w/h}
        except Exception:
            return None
    return None

# Function 3: Full Image Preprocessing Pipeline
def image_preprocess(row_tuple):
    # Unpack inputs
    index, row, input_folder, output_folder, scale = row_tuple
    # Image identifiers
    img_id = str(row['image']).strip()
    img_name = str(row['image']) + '.jpeg'
    
    input_path = os.path.join(input_folder, img_name)
    output_path = os.path.join(output_folder, img_name)

    if not os.path.exists(input_path):
        return False

    if os.path.exists(output_path):
        return True
    try:
        img = cv2.imread(input_path)
        if img is None: return False

        # Auto-Cropping
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 10
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        # Crop only if valid region exists
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            img = img[rmin:rmax+1, cmin:cmax+1]

        # Scale Normalization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        middle_row = gray[gray.shape[0] // 2, :]
        eye_pixels = (middle_row > middle_row.mean() / 10).sum()
        r = eye_pixels / 2
        # Resize image based on radius
        if r > 0:
            s = scale / r
            img = cv2.resize(img, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)

        # Square Padding
        h, w = img.shape[:2]
        side = max(h, w)
        # Create padded square image with gray background
        padded_img = np.full((side, side, 3), 128, dtype=np.uint8)
        # Center original image
        top = (side - h) // 2
        left = (side - w) // 2
        padded_img[top:top+h, left:left+w] = img
        img = padded_img

        # Ben Graham Enhancement
        # Subtract local average to remove lighting bias
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Apply Gaussian blur (local average)
        blurred = cv2.GaussianBlur(img, (0, 0), scale / 30)
        # Enhance contrast by subtracting local average
        img = cv2.addWeighted(img, 4, blurred, -4, 128)

        # Circular Mask
        mask = np.zeros(img.shape, dtype=np.uint8)
        h, w = img.shape[:2]
        # Create circular mask centered in image
        cv2.circle(mask, (w // 2, h // 2), int(scale * 0.9), (1, 1, 1), -1)
        # Apply mask and fill outside with gray
        img = img * mask + 128 * (1 - mask)

        # Save Processed Image
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True
    except Exception:
        return False