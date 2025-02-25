import torch 
from dataset import CartoonRealDataset
import sys
from utils import save_checkpoint, load_checkpoint 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
import config as mc
import importlib
from glob import glob

importlib.reload(mc)

def test_cartoon(gen_C, image, name):
    fake_cartoon = gen_C(image);
    save_image(fake_cartoon*0.5+0.5, f"test_results/cartoon_{name}.png")
         

# use glob to read in all the test images 
def main():
     gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
#     load the models up so I can use them
     if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_gen_C, gen_C, opt_gen, config.LEARNING_RATE,)
        
    globname = "data/test/*.png"
    img_file = glob(globname)
    
    for i in range(0, len(img_file)):
        print(f"Testing image {0}...")
        img = cv2.imread(img_file[i])
        img_arr = np.array(img)
        test_cartoon(gen_C, img, img_file[0])
    
    
    
if __name__ == "__main__":
    main()