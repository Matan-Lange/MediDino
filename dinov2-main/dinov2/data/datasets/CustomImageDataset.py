# this file was added completely new
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from .decoders import ImageDataDecoder
import random


class CustomImageDataset(Dataset):
    def __init__(
            self,
            split,
            root: str,
            transform=None,
            target_transform=None):

        self.img_dir = root
        self.imgs = os.listdir(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.imgs[idx])
            with open(img_path, mode="rb") as f:
                image_pil = f.read()
            image_pil = ImageDataDecoder(image_pil).decode()
        except Exception as e:
            # in case of error when reading image, just take a random different one
            random_index = random.randint(0, len(self) - 1)
            image = self.__getitem__(random_index)
            return image, None
        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, None

    def get_test_item(self, idx):
        img_path = os.path.join(self.img_dir, self.test_data.iloc[idx, 0])
        image = read_image(img_path)
        image_pil = transforms.ToPILImage()(image)
        if self.transform:
            image_pil = self.transform(image_pil)
        return image_pil