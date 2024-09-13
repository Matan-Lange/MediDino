import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, base_path, csv_path, label_map, transform=None):
        self.data = pd.read_csv(csv_path)
        self.base_path = base_path
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(os.path.join(self.base_path, row[0]))
        y = torch.tensor(self.label_map[row[1]])

        if self.transform:
            image = self.transform(image)

        return image, float(y)
