import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# change train_dir to the folder with video frames
REAL_TRAIN_DIR = "fourth_vid_frames"

# get data from other directory
CARTOON_TRAIN_DIR = "../02_CYCLE_GAN/data"
TEST_DIR = "../02_CYCLE_GAN/data"
BATCH_SIZE = 1
LEARNING_RATE = 1e-7
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT_GEN_R = "../04_CARTOON_GAN_EDGES/genr.pth.tar"
CHECKPOINT_GEN_C =  "../04_CARTOON_GAN_EDGES/genc.pth.tar"
CHECKPOINT_CRITIC_R = "../04_CARTOON_GAN_EDGES/criticr.pth.tar"
CHECKPOINT_CRITIC_C = "../04_CARTOON_GAN_EDGES/criticc.pth.tar"

CHECKPOINT_SAVE_GEN_R = "genr.pth.tar"
CHECKPOINT_SAVE_GEN_C = "genc.pth.tar"
CHECKPOINT_SAVE_CRITIC_R = "criticr.pth.tar"
CHECKPOINT_SAVE_CRITIC_C = "criticc.pth.tar"

TEMPORAL_WEIGHT = 15
ADVERSARIAL_WEIGHT = 1
CONTENT_WEIGHT = 1.1
IDENTITY_WEIGHT = 0.1
CYCLE_WEIGHT = 10

INITIALIZE = False

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    is_check_shapes=False,
    additional_targets={"image0":"image", "image1":"image"},
)