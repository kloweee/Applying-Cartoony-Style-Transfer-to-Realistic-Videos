# GOOD

from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

# CartoonRealDataset
# cartoon == zebra, real == horse
class CartoonRealDataset(Dataset):
    def __init__(self, root_cartoon, root_real, transform=None):
        self.root_cartoon = root_cartoon
        self.root_real = root_real
        self.transform = transform

        self.cartoon_images = os.listdir(root_cartoon)
        self.real_images = os.listdir(root_real)
        self.length_dataset = max(len(self.cartoon_images), len(self.real_images))
        self.cartoon_len = len(self.cartoon_images)
        self.real_len = len(self.real_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        # make sure image index is in correct range
        cartoon_img = self.cartoon_images[index % self.cartoon_len]
        real_img = self.real_images[index % self.real_len]

        cartoon_path = os.path.join(self.root_cartoon, cartoon_img)
        real_path = os.path.join(self.root_real, real_img)

        cartoon_img = np.array(Image.open(cartoon_path).convert("RGB"))
        real_img = np.array(Image.open(real_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=cartoon_img, image0 = real_img)
            cartoon_img = augmentations["image"]
            real_img = augmentations["image0"]

        return cartoon_img, real_img