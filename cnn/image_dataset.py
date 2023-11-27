import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision.transforms import PILToTensor
import torchvision.transforms as tf


class ImageDataset(Dataset):
    # Mapping from char to int. We could also use their ASCII values as well
    char_to_index = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                     'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18,
                     'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
                     'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}

    def __init__(self, csv_file, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(csv_file, header=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[[idx]]

        url = str(row['Image'].iloc[0])
        base_file_name = url.split('/')[-1]  # Name of the image extracted
        directory_name, extension = base_file_name.split('.')
        label = str(row['Plate number'].iloc[0]).replace("-", "").replace(" ", "")

        # Check the directory, if it contains less than 5 images, return dummy values
        current_dir = os.path.join(self.data_path, directory_name)
        if not os.path.exists(current_dir):
            return None, None

        file_list = os.listdir(current_dir)
        if len(file_list) != len(label):
            return None, None

        transform = tf.Compose([
            tf.ToTensor()
        ])

        # Load the image
        image_list = []
        label_list = []
        for idx, file in enumerate(file_list):
            img = Image.open(f"{current_dir}/{directory_name}_{idx}.{extension}").convert("L")
            image_list.append(transform(img))
            label_tensor = torch.zeros((1, 36), dtype=torch.float32)
            label_tensor[0][self.char_to_index[label[idx]]] = 1.0
            label_list.append(label_tensor)

        try:
            return torch.stack(image_list, dim=0), torch.stack(label_list, dim=0)
        except RuntimeError:
            return None, None

    def sanitize_data(self):
        idx_to_delete = []
        for idx in range(self.__len__()):
            images, labels = self.__getitem__(idx)
            if images is None or labels is None:
                idx_to_delete.append(idx)

        print(f"Out of {len(self.df)} plate images {len(self.df) - len(idx_to_delete)} are usable. "
              f"Dropping the others from the dataset...")
        self.df = self.df.drop(self.df.index[idx_to_delete])

