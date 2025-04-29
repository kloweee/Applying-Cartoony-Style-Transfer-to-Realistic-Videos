import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# use own directories
TRAIN_DIR = "../02_CYCLE_GAN/data"
TEST_DIR = "../02_CYCLE_GAN/data/test/own_test/"
VAL_DIR = "../02_CYCLE_GAN/data"

BATCH_SIZE = 1
LEARNING_RATE = 1e-5
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True

# create files to save training process
CHECKPOINT_GEN_R = "genr.pth.tar"
CHECKPOINT_GEN_C = "genc.pth.tar"
CHECKPOINT_CRITIC_R = "criticr.pth.tar"
CHECKPOINT_CRITIC_C = "criticc.pth.tar"

# CHECKPOINT_GEN_R = "genr_restart.pth.tar"
# CHECKPOINT_GEN_C = "genc_restart.pth.tar"
# CHECKPOINT_CRITIC_R = "criticr_restart.pth.tar"
# CHECKPOINT_CRITIC_C = "criticc_restart.pth.tar"

CONTENT_WEIGHT = 1.13
ADVERSARIAL_WEIGHT = 1
IDENTITY_WEIGHT = 0.3
CYCLE_WEIGHT = 10

INITIALIZE = False
SAVE_AT = 500

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