import torch 
import sys
from utils import load_checkpoint 
import torch.optim as optim
import config
from torchvision.utils import save_image
from generator import Generator
import config as mc
import importlib
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os 

importlib.reload(mc)

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    is_check_shapes=False,
    additional_targets={"image0":"image"},
)


def test_cartoon(gen_C, image, name):
    fake_cartoon = gen_C(image);
    save_image(fake_cartoon*0.5+0.5, f"test_results/cartoon_{name}.png")
    save_image(image*0.5+0.5,f"test_results/input_{name}.png")
         

def main():
    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_R =  Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_C.parameters()) + list(gen_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    
    # load the models up so I can use them
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_C, gen_C, opt_gen, config.LEARNING_RATE,)
        
    img_name = input("Please enter the name of your image: ")
    path_name = config.TEST_DIR + img_name
    save_name = img_name[0:(len(img_name)-4)]

    if not os.path.exists(path_name):
        raise FileNotFoundError(f"The file '{path_name}' does not exist.") 
        
    try:
        with open(path_name, 'r') as file:
            img = np.array(Image.open(path_name).convert("RGB"))
            augmentations = transforms(image=img, image0=img)
            img = augmentations["image"]
            img = img.cuda()
            test_cartoon(gen_C, img, save_name)
            print("Done!")
            
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()