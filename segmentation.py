import os
import cv2
import numpy as np
from utils.constants import LP_IMAGE_DIR, ORIENTED_IMAGE_DIR

input_directory = LP_IMAGE_DIR
output_directory = ORIENTED_IMAGE_DIR
not_found = 0
target_size = (300, 100)

def is_mostly_black(image, threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mean_brightness = np.mean(gray)
    
    return mean_brightness < threshold

def is_right_size(image, aspect_ratio_threshold=2.0):
    global not_found
    
    height, width, _ = image.shape
    aspect_ratio = width / height

    return aspect_ratio > aspect_ratio_threshold

def process_license_plate(input_path, output_directory):
    global not_found
    
    image = cv2.imread(input_path)

    if is_mostly_black(image) or not is_right_size(image):
        not_found += 1
        print(f'#{not_found} Skipping image:', input_path)
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to binarize the image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (likely not a license plate)
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    if valid_contours:
        max_contour = max(valid_contours, key=cv2.contourArea)
        
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width, height = rect[1]
        orientation = rect[2]

        if width < height:
            if orientation < -45:
                orientation += 90
            else :
                orientation -= 90

        # Rotate the image to correct the orientation
        corrected = cv2.warpAffine(image, cv2.getRotationMatrix2D(rect[0], orientation, 1), image.shape[1::-1])

        # Crop the rotated license plate region
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped = corrected[y:y + h, x:x + w]

        resized_plate = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

        if is_mostly_black(resized_plate) :
            not_found += 1
            print(f'#{not_found} Skipping image:', input_path)
            return

        os.makedirs(output_directory, exist_ok=True)

        output_path = os.path.join(output_directory, os.path.basename(input_path))
        cv2.imwrite(output_path, resized_plate)
    else:
        not_found += 1
        print(f'#{not_found} Skipping image:', input_path)


for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_directory, filename)
        process_license_plate(input_path, output_directory)
print(f'License plates have been processed and saved in the output directory. {not_found} skiped images')
