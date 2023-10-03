import os
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import cv2 

from utils.constants import IMAGE_DIR, LP_IMAGE_DIR, DATABASE_FILE, PROCESSED_DATABASE_FILE


def read_image_from_url(url, verbose=0):
    '''Returns an image from a given URL'''

    req = Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0'},
    )
    arr = np.asarray(bytearray(urlopen(req).read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)

    if verbose: print(f'Got: {url}')
    return img


if __name__ == '__main__':
    if not os.path.exists(IMAGE_DIR): os.mkdir(IMAGE_DIR)
    if not os.path.exists(LP_IMAGE_DIR): os.mkdir(LP_IMAGE_DIR)

    df = pd.read_csv(DATABASE_FILE)

    for idx, row in df.iterrows():
        if idx % 200 == 0: print(f'{idx} downloaded')

        url = row['Image']
        file_name = url.split('/')[-1]

        if file_name not in os.listdir(IMAGE_DIR):
            if url.find('https') == -1:
                url = url.replace('http', 'https')

            img = read_image_from_url(url)

            path = f'{IMAGE_DIR}/{file_name}'
            cv2.imwrite(path, img)

        df.loc[idx, 'Image'] = file_name

    if df.shape[0] != len(os.listdir(IMAGE_DIR)):
        #print(f'Should have got {df.shape[0]} files, yet there are only {len(os.listdir(IMAGE_DIR))}!')
        print(df.nunique())
        print('Nevermind, they just put duplicated images into the database.')

    df.to_csv(PROCESSED_DATABASE_FILE, index=False)