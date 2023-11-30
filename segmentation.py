import os
import cv2
import numpy as np
from utils.constants import LP_IMAGE_DIR, ORIENTED_IMAGE_DIR_PHASE_1, ORIENTED_IMAGE_DIR_PHASE_2, ORIENTED_IMAGE_DIR_PHASE_3, ORIENTED_IMAGE_DIR

input_directory = LP_IMAGE_DIR
output_directory1 = ORIENTED_IMAGE_DIR_PHASE_1
output_directory2 = ORIENTED_IMAGE_DIR_PHASE_2
output_directory3 = ORIENTED_IMAGE_DIR_PHASE_3
output_directory4 = ORIENTED_IMAGE_DIR
output_directory_skipped_wrong_color = 'data/oriented/skipped/wrong_color'
output_directory_skipped_bright = 'data/oriented/skipped/bright'
output_directory_skipped_size = 'data/oriented/skipped/size'
output_directory_skipped_countur = 'data/oriented/skipped/contour'
not_found = 0
target_size = (300, 100)

def has_enough_black_pixels(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #  adaptive thresholding to binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the percentage of black pixels in the binary image
    total_pixels = np.prod(image.shape[:2])
    black_pixels = np.sum(binary == 0)
    black_ratio = black_pixels / total_pixels

    return black_ratio <= threshold 

def has_too_much_blue(image, blue_threshold=1.15, red_threshold=0.7, green_threshold=0.7):
    blue, green, red = cv2.split(image)

    mean_blue = np.mean(blue)
    mean_green = np.mean(green)
    mean_red = np.mean(red)

    # Check if the blue channel intensity is significantly higher than red and green
    if mean_blue > blue_threshold * mean_red and mean_blue > blue_threshold * mean_green:
        if mean_red < red_threshold * mean_blue and mean_green < green_threshold * mean_blue:
            return True
    
    return False

def has_too_much_red(image, red_threshold=1.1, blue_threshold=0.7, green_threshold=0.4):
    blue, green, red = cv2.split(image)

    mean_blue = np.mean(blue)
    mean_green = np.mean(green)
    mean_red = np.mean(red)

    # Check if the blue channel intensity is significantly higher than red and green
    if mean_red > red_threshold * mean_blue and mean_red > red_threshold * mean_green:
        if mean_blue < blue_threshold * mean_red and mean_green < green_threshold * mean_red:
            return True
    
    return False

def is_mostly_black(image, black_threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    avg_color_per_row = np.average(binary, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color <= black_threshold 

def colorFilter(image, black_threshold=30):
    return is_mostly_black(image, black_threshold) or has_too_much_blue(image) or has_too_much_red(image)

def is_right_size(image, aspect_ratio_threshold=1.0):
    global not_found
    
    height, width, _ = image.shape
    aspect_ratio = width / height

    return aspect_ratio > aspect_ratio_threshold

def process_license_plate(image, output_directory):
    global not_found

    if colorFilter(image):
        not_found += 1
        print(f'#{not_found} Skipping image:', input_path)
        os.makedirs(output_directory_skipped_wrong_color, exist_ok=True)
        output_path = os.path.join(output_directory_skipped_wrong_color, os.path.basename(input_path))
        cv2.imwrite(output_path, image)
        return

    if not is_right_size(image) and colorFilter(image, 155):
        not_found += 1
        print(f'#{not_found} Skipping image:', input_path)
        os.makedirs(output_directory_skipped_size, exist_ok=True)
        output_path = os.path.join(output_directory_skipped_size, os.path.basename(input_path))
        cv2.imwrite(output_path, image)
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (likely not a license plate)
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 700]

    if valid_contours:
        max_contour = max(valid_contours, key=cv2.contourArea)
        
        rect = cv2.minAreaRect(max_contour)

        width, height = rect[1]
        orientation = rect[2]

        if width < height:
            if orientation < -45:
                orientation += 90
            else :
                orientation -= 90

        # Rotate the image
        corrected = cv2.warpAffine(image, cv2.getRotationMatrix2D(rect[0], orientation, 1), image.shape[1::-1])

        x, y, w, h = cv2.boundingRect(max_contour)
        cropped = corrected[y:y + h, x:x + w]

        resized_plate = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

        if colorFilter(resized_plate) :
            not_found += 1
            print(f'#{not_found} Skipping image:', input_path)
            os.makedirs(output_directory_skipped_wrong_color, exist_ok=True)
            output_path = os.path.join(output_directory_skipped_wrong_color, os.path.basename(input_path))
            cv2.imwrite(output_path, resized_plate)
            return

        if has_enough_black_pixels(resized_plate, 0.1):
            not_found += 1
            print(f'#{not_found} Skipping image:', input_path)
            os.makedirs(output_directory_skipped_bright, exist_ok=True)
            output_path = os.path.join(output_directory_skipped_bright, os.path.basename(input_path))
            cv2.imwrite(output_path, resized_plate)
            return

        os.makedirs(output_directory, exist_ok=True)

        output_path = os.path.join(output_directory, os.path.basename(input_path))
        cv2.imwrite(output_path, resized_plate)
    else:
        not_found += 1
        print(f'#{not_found} Skipping image:', input_path)
        os.makedirs(output_directory_skipped_countur, exist_ok=True)
        output_path = os.path.join(output_directory_skipped_countur, os.path.basename(input_path))
        cv2.imwrite(output_path, image)


processed_images = 0
for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_directory, filename)
        process_license_plate(cv2.imread(input_path), output_directory1)
        processed_images+=1
print(f'{processed_images - not_found} license plates have been processed and saved in the output directory. {not_found} skiped images')
not_found = 0
processed_images = 0
for filename in os.listdir(output_directory1):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(output_directory1, filename)
        process_license_plate(cv2.imread(input_path), output_directory2)
        processed_images+=1
print(f'{processed_images - not_found} license plates have been processed and saved in the output directory. {not_found} skiped images')
not_found = 0
processed_images = 0
for filename in os.listdir(output_directory2):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(output_directory2, filename)
        process_license_plate(cv2.imread(input_path), output_directory3)
        processed_images+=1
print(f'{processed_images - not_found} license plates have been processed and saved in the output directory. {not_found} skiped images')
not_found = 0
processed_images = 0
for filename in os.listdir(output_directory3):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(output_directory3, filename)
        process_license_plate(cv2.imread(input_path), output_directory4)
        processed_images+=1
print(f'{processed_images - not_found} license plates have been processed and saved in the output directory. {not_found} skiped images')