print("Importing modules...")

import cv2
import numpy as np
import os 
from glob import glob
import time
import torch 
from dataset import CartoonRealDataset
import sys
from utils import save_checkpoint, load_checkpoint 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import config
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision.transforms as T
from discriminator import Discriminator
from generator import Generator
import config as mc
import importlib

print("Done!")

importlib.reload(mc)
frame_dir_name = ""
stylized_dir_name = ""
video_name = ""
fps = 0

def train_fxn(disc_R, disc_C, gen_C, gen_R, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    # visualize training process with tqdm
    loop = tqdm(loader, leave=True)
    real_next = 0 
    
#     TO DO TEMPORAL LOSS -> just create outside variable to save prev picture and train it with the next picture 
    
    for idx, (cartoon, real, blurry) in enumerate(loop):
        cartoon = cartoon.to(config.DEVICE)
        
        # gen both current and next real frames in video?
        real = real.to(config.DEVICE)
        if (idx == 0):
            real_next = real.to(config.DEVICE)
        
        blurry = blurry.to(config.DEVICE)
        
        # train discriminators H and Z
        with torch.amp.autocast("cuda"):
            # generate fake real image
            fake_real = gen_R(cartoon)
            
            # run disc on real and fake reals
            D_R_real = disc_R(real)
            D_R_fake = disc_R(fake_real.detach()) # detach to use fake real original
            D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
            D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_R_loss = D_R_real_loss + D_R_fake_loss

            # generate fake cartoon images for both tn and tn+1
            fake_cartoon = gen_C(real)
            fake_cartoon_next = gen_C(real_next)

            D_C_real = disc_C(cartoon)
            D_C_fake = disc_C(fake_cartoon.detach()) 
            D_C_fake_next = disc_C(fake_cartoon_next.detach())

            D_C_blurry = disc_C(blurry)
            D_C_blurry_loss = mse(D_C_blurry, torch.zeros_like(D_C_blurry)) * 100 #COMPARE TO REAL IMAGE
            D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))
            D_C_fake_loss = mse(D_C_fake, torch.zeros_like(D_C_fake))
            D_C_fake_loss_next = mse(D_C_fake_next, torch.zeros_like(D_C_fake_next))
            
            D_C_loss = D_C_real_loss + D_C_fake_loss + D_C_blurry_loss + D_C_fake_loss_next

            # put it together
            D_loss = (D_R_loss + D_C_loss)/2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

        # train generators h and z
        with torch.amp.autocast("cuda"): 
            D_R_fake = disc_R(fake_real)
            D_C_fake = disc_C(fake_cartoon)
            D_C_fake_next = disc_C(fake_cartoon_next)

            # want to trick the discriminator into thinking fake data is real with this part, so switch the two parts 
            loss_G_R = mse(D_R_fake, torch.ones_like(D_R_fake))
            loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))
            loss_G_C_next = mse(D_C_fake_next, torch.ones_like(D_C_fake_next))
            
            temporal_loss = l1(fake_cartoon, fake_cartoon_next)

            # content loss
            content_cartoon_loss = l1(real, fake_cartoon)
            content_real_loss = l1(cartoon, fake_real)
            content_cartoon_next_loss = l1(real_next, fake_cartoon_next)
            
            # cycle loss
            # difference between input image and generated INPUT image (so loop basically)
            cycle_cartoon = gen_C(fake_real)
            cycle_real = gen_R(fake_cartoon)
            cycle_real_next = gen_R(fake_cartoon_next)
            cycle_cartoon_loss = l1(cartoon, cycle_cartoon)
            cycle_real_loss = l1(real, cycle_real)
            cycle_real_next_loss = l1(real_next, cycle_real_next)

            # identity loss -> see if corresponding generators will generate the right thing? 
            identity_cartoon = gen_C(cartoon)
            identity_real = gen_R(real)
            identity_real_next = gen_R(real_next)
            identity_cartoon_loss = l1(cartoon, identity_cartoon)
            identity_real_loss = l1(real, identity_real)
            identity_real_next_loss = l1(real_next, identity_real_next)

            # add all generator losses together
            G_loss = (
                loss_G_C * config.ADVERSARIAL_WEIGHT +
                loss_G_R * config.ADVERSARIAL_WEIGHT +
                loss_G_C_next * config.ADVERSARIAL_WEIGHT + 
                temporal_loss * config.TEMPORAL_WEIGHT + 
                cycle_cartoon_loss * config.LAMBDA_CYCLE + 
                cycle_real_loss * config.LAMBDA_CYCLE + 
                identity_real_loss * config.LAMBDA_IDENTITY +
                identity_cartoon_loss * config.LAMBDA_IDENTITY + 
                cycle_real_next_loss * config.LAMBDA_CYCLE +
                identity_real_next_loss * config.LAMBDA_IDENTITY +
                content_cartoon_loss * config.CONTENT_WEIGHT + 
                content_real_loss * config.CONTENT_WEIGHT +
                content_cartoon_next_loss * config.CONTENT_WEIGHT
            )
            
#             removed cycle_real_next_loss * config.LAMBDA_CYCLE +  identity_real_next_loss * config.LAMBDA_IDENTITY + loss_G_C_NEXT + 
            
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
        
        # save images throughout training project
        fname = str(idx)
        while len(fname) < 3:
            fname = "0" + fname
        
        
        resize = T.Resize((720, 1280))
        resized_img = resize(fake_cartoon)
        save_image(resized_img*0.5+0.5, f"{stylized_dir_name}/{fname}.png")
        real_next = real.to(config.DEVICE)
       
    
def make_video():
    globname = f"{stylized_dir_name}/*.png"
    video_name = "vid_to_cartoon.avi"

    images = glob(globname)
    images.sort()

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), int(fps), (width, height))

    # loop through all the images in the folder
    for image in images:
        video.write(cv2.imread(image))

    # release the video 
    video.release()
    print("Created video!")

     
def get_frames():
    vidObj = cv2.VideoCapture(video_name)
    global fps
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    count = 0
    success = 1

    start = time.time()
    
    while success:
        success, image = vidObj.read()
        if not success:
            break
        str_num = str(count)
        while len(str_num) < 3:
            str_num = "0" + str_num

        fname = f"{frame_dir_name}/" + str_num + ".png"

        cv2.imwrite(fname, image)
        count += 1
    
    vidObj.release()
    end = time.time()
    print(f"It took {end-start} seconds to split up all the frames in this video")
    
    delete_dir = frame_dir_name + "/.ipynb_checkpoints"
    if os.path.exists(delete_dir):
        os.rmdir(delete_dir)
        
        
def main():
    global frame_dir_name
    global video_name
    global stylized_dir_name
    video_name = input("Enter video file path: ")
    frame_dir_name = input("Enter path for directory to store video frames: ")
    stylized_dir_name = "stylized_" + frame_dir_name

    if not os.path.exists(frame_dir_name):
        os.makedirs(frame_dir_name)
        
    if not os.path.exists(stylized_dir_name):
        os.makedirs(stylized_dir_name)
    
    start_new = input("Are you using a new video? (Y/N): ")
    if (start_new == "Y"):
        print("Splitting video into frames now...")
        get_frames()
    
    # initialize the discriminators and generators
    # real and cartoon discriminator
    disc_R = Discriminator(in_channels=3).to(config.DEVICE)
    disc_C = Discriminator(in_channels=3).to(config.DEVICE) 
    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_R =  Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
 
    # optimization functions are used to quantify the error between models predictions and actual target values

    # Adam (adaptive movement estimation): combines both momentum optimization and RMS (root mean square) propagation 
    # lr = learning rate, use the one from config, which has a lot of the main variables (global variables)
    opt_disc = optim.Adam(
        list(disc_R.parameters()) + list(disc_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # make optimizer for discriminator and generator
    opt_gen = optim.Adam(
        list(gen_C.parameters()) + list(gen_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # define losses     
    # l1 loss for cycle consistency and identity loss
    # mse for adversarial loss
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_R, gen_R, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_GEN_C, gen_C, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_CRITIC_R, disc_R, opt_disc, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_CRITIC_C, disc_C, opt_disc, config.LEARNING_RATE,)

    dataset = CartoonRealDataset(
        root_cartoon = config.CARTOON_TRAIN_DIR+"/cartoon/disney_people_processed", root_real = frame_dir_name, root_blurry = config.CARTOON_TRAIN_DIR+"/blurry/disney_blurry", transform=config.transforms,
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    g_scaler = torch.amp.GradScaler("cuda")
    d_scaler = torch.amp.GradScaler("cuda")
    
    while True:
        should_train = input("Begin/continue training? (Y/N): ")

        if should_train == "Y":
            train_fxn(disc_R, disc_C, gen_C, gen_R, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

            if config.SAVE_MODEL:
                save_checkpoint(gen_R, opt_gen, filename=config.CHECKPOINT_SAVE_GEN_R)
                save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_SAVE_GEN_C)
                save_checkpoint(disc_R, opt_disc, filename=config.CHECKPOINT_SAVE_CRITIC_R)
                save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_SAVE_CRITIC_C)

        elif should_train == "N":
            make_video()
            break
            
        else:
            print("Invalid response!")
            break

    
# run the main file
if __name__ == "__main__":
    main()