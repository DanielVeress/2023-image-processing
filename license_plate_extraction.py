import os
import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
from utils.constants import PROCESSED_DATABASE_FILE, IMAGE_DIR, LP_IMAGE_DIR

not_found = 0
def has_enough_black_pixels(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #  adaptive thresholding to binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the percentage of black pixels in the binary image
    total_pixels = np.prod(image.shape[:2])
    black_pixels = np.sum(binary == 0)
    black_ratio = black_pixels / total_pixels

    return black_ratio >= threshold 


def get_license_plate(original, edged): 
    global not_found
    
    kps = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(kps)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        
        if len(approx) == 4 and cv2.contourArea(approx) > 65:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 1.1 < aspect_ratio < 12.0:  # Adjust aspect ratio thresholds
                cropped = original[y:y+h, x:x+w]
                avg_color = np.mean(cropped, axis=(0, 1))  # Get average color
                is_bright_enough = np.all(avg_color > 60)
                is_not_too_bright = np.all(avg_color < 230)
                
                if is_bright_enough and is_not_too_bright and has_enough_black_pixels(cropped, 0.08):
                    location = approx
                    break
            
    if location is None:
        not_found += 1
        return None, None
        
    mask = np.zeros(original.shape[:2], np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(original, original, mask=mask)
    
    return new_image, mask


def crop_license_plate(original, mask):
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped = original[x1:x2+1, y1:y2+1]

    return cropped


if __name__ == '__main__':
    plot = False
    
    df = pd.read_csv(PROCESSED_DATABASE_FILE)
    for idx, row in df.iterrows():
        if idx % 200 == 0: print(f'{idx} processed')
        file_name = f"{IMAGE_DIR}/{row['Image']}"
        
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        masked_image, mask = get_license_plate(img, edged) 
        
        if masked_image is not None:
            license_plate = crop_license_plate(img, mask)
        
            if plot:
                fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,10))
                axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[0, 1].imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
                axes[1, 0].imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
                axes[1, 1].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
                axes[2, 0].imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))

                fig.delaxes(axes[2,1])

            file_name = row['Image']
            if file_name not in os.listdir(LP_IMAGE_DIR):
                path = f'{LP_IMAGE_DIR}/{file_name}'
                cv2.imwrite(path, license_plate)
        
    print(f'We have found {len(os.listdir(LP_IMAGE_DIR))} license plates!')
    print(f'We could not find {not_found} license plates!')