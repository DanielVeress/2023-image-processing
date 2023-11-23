import torch
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
                     'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, '-': 36}

    def __init__(self, csv_file, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(csv_file, header=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[[idx]]

        url = str(row['Image'].iloc[0])
        file_name = url.split('/')[-1]  # Name of the image extracted

        label = str(row['Plate number'].iloc[0])

        # Load the image
        img = Image.open(f"{self.data_path}/{file_name}").convert('RGB')
        transform = tf.Compose([
            tf.ToTensor()
        ])

        # Convert label to tensor (assuming label is a string)
        label_tensor = torch.tensor([self.char_to_index[c] for c in label], dtype=torch.long)

        return transform(img), label_tensor
