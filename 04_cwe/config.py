import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "../02_CYCLE_GAN/data"
TEST_DIR = "../02_CYCLE_GAN/data"
VAL_DIR = "../02_CYCLE_GAN/data"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_R = "genr.pth.tar"
CHECKPOINT_GEN_C = "genc.pth.tar"
CHECKPOINT_CRITIC_R = "criticr.pth.tar"
CHECKPOINT_CRITIC_C = "criticc.pth.tar"
INITIALIZE = False

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    is_check_shapes=False,
    additional_targets={"image0":"image", "image1":"image"},
)