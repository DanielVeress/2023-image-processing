import os
import statistics
import cv2
from matplotlib import pyplot as plt
from utils.constants import ORIENTED_IMAGE_DIR, SEGMENTED_LP_DIR, MAX_SEGMENTED_CHAR_WIDTH


def pad_image(image, target_size):
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate padding
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    # Calculate padding amounts for top, bottom, left, and right
    top = int(pad_h // 2)
    bottom = int(pad_h - top)
    left = int(pad_w // 2)
    right = int(pad_w - left)

    # Add padding
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return padded_image


def segment_characters(license_plate_image):
    '''
    Segments characters from a license plate image.

    Parameters:
    - license_plate_image: The input license plate image in BGR format.

    Returns:
    - List or None: A list of segmented character images if successful, 
      or None if no characters were found.
    '''

    gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours
    min_contour_area = 100
    min_aspect_ratio = 0.0
    max_aspect_ratio = 2.0
    filtered_contours = [cnt for cnt in contours if
                         min_contour_area < cv2.contourArea(cnt) < 10000 and
                         min_aspect_ratio < aspect_ratio(cnt) < max_aspect_ratio]
    if len(filtered_contours) == 0: return None

    # sort contours from left to right
    sorted_contours = sorted(filtered_contours, key=lambda x: cv2.boundingRect(x)[0])

    segments = []
    segment_heights = []
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        segment = license_plate_image[y:y + h, x:x + w]
        segments.append(segment)
        segment_heights.append(segment.shape[0])

    # final filtering
    most_common_height = statistics.mode(segment_heights)
    most_common_height = round(most_common_height, -1)
    acceptable_error = 0.2
    characters = []
    for segment in segments:
        lower_bound = most_common_height - acceptable_error * most_common_height
        upper_bound = most_common_height + acceptable_error * most_common_height
        if (lower_bound < segment.shape[0] and segment.shape[0] < upper_bound):
            gray_char = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            bg = cv2.morphologyEx(gray_char, cv2.MORPH_DILATE, se)
            out_gray = cv2.divide(gray_char, bg, scale=255)
            out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
            out_binary = pad_image(out_binary, (MAX_SEGMENTED_CHAR_WIDTH, MAX_SEGMENTED_CHAR_WIDTH))
            characters.append(out_binary)
    if len(characters) == 0: return None

    return characters


def aspect_ratio(contour):
    _, _, w, h = cv2.boundingRect(contour)
    return float(w) / h


if __name__ == '__main__':
    input_dir = ORIENTED_IMAGE_DIR
    output_dir = SEGMENTED_LP_DIR

    os.makedirs(output_dir, exist_ok=True)

    not_found = 0
    for idx, filename in zip(range(len(os.listdir(input_dir))), os.listdir(input_dir)):
        if idx % 200 == 0: print(f'{idx} processed')

        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(input_dir, filename)
            license_plate_image = cv2.imread(file_path)

            segmented_characters = segment_characters(license_plate_image)

            # save segments
            if segmented_characters is not None:
                file_id = filename.split('.')[0]
                file_segment_path = os.path.join(output_dir, file_id)
                os.makedirs(file_segment_path, exist_ok=True)

                for idx, character in zip(range(len(segmented_characters)), segmented_characters):
                    cv2.imwrite(file_segment_path + f'/{file_id}_{idx}.jpg', character)
            else:
                not_found += 1

    print(f'For {not_found} images, no segments were found.')
