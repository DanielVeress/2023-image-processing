import statistics
import cv2
import numpy as np
from matplotlib import pyplot as plt


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
        segment = license_plate_image[y:y+h, x:x+w]
        segments.append(segment)
        segment_heights.append(segment.shape[0])
        
    # final filtering
    most_common_height = statistics.mode(segment_heights)
    most_common_height = round(most_common_height, -1)
    acceptable_error = 0.2
    characters = []
    for segment in segments:
        lower_bound = most_common_height - acceptable_error*most_common_height
        upper_bound = most_common_height + acceptable_error*most_common_height
        if (lower_bound < segment.shape[0] and segment.shape[0] < upper_bound):
            characters.append(segment)
    if len(characters) == 0: return None
        
    return characters


def aspect_ratio(contour):
    _, _, w, h = cv2.boundingRect(contour)
    return float(w) / h


if __name__ == '__main__':
    license_plate_image = cv2.imread('data/oriented/18152279.jpg')#17548853, 17558046, 17560430, 18068110, 18142603

    segmented_characters = segment_characters(license_plate_image)

    if segmented_characters is not None:
        plt.figure(figsize=(10, 5))
        for i, char in enumerate(segmented_characters):
            plt.subplot(1, len(segmented_characters), i + 1)
            plt.imshow(cv2.cvtColor(char, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.show()

    if segmented_characters is None:
        print('License plate was not found!')
    elif len(segmented_characters) < 6:
        print('Not enough characters were found!')