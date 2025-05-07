# GOOD

from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

# CartoonRealDataset
class CartoonRealDataset(Dataset):
    def __init__(self, root_cartoon, root_real, root_blurry, transform=None):
        self.root_cartoon = root_cartoon
        self.root_real = root_real
        self.root_blurry = root_blurry
        self.transform = transform

        self.cartoon_images = os.listdir(root_cartoon)
        self.real_images = sorted(os.listdir(root_real))
        self.blurry_images = os.listdir(root_blurry)
        self.length_dataset = min(len(self.cartoon_images), len(self.real_images))
        self.cartoon_len = len(self.cartoon_images)
        self.real_len = len(self.real_images)
        self.blurry_len = len(self.blurry_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        # make sure image index is in correct range
        cartoon_ind = index % self.cartoon_len
        cartoon_img = self.cartoon_images[index % self.cartoon_len]
#         real_img = self.real_images[index % self.real_len]
        real_img = self.real_images[index]
        blurry_img = self.blurry_images[index % self.blurry_len]

        cartoon_path = os.path.join(self.root_cartoon, cartoon_img)
        real_path = os.path.join(self.root_real, real_img)
        blurry_path = os.path.join(self.root_blurry, blurry_img)

        cartoon_img = np.array(Image.open(cartoon_path).convert("RGB"))
        real_img = np.array(Image.open(real_path).convert("RGB"))
        blurry_img = np.array(Image.open(blurry_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=cartoon_img, image0=real_img, image1=blurry_img)
            cartoon_img = augmentations["image"]
            real_img = augmentations["image0"]
            blurry_img = augmentations["image1"]

        return cartoon_img, real_img, blurry_img
