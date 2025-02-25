import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
import random


print(torch.cuda.is_available())
# g_scaler = torch.cuda.amp.GradScaler()
# print(g_scaler)

# how to unzip files in jupyterlab
# import zipfile
# with zipfile.ZipFile("disney_people.zip","r") as zip_ref:
#     zip_ref.extractall("data")