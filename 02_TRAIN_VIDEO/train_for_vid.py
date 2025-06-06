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

importlib.reload(mc)

def train_fxn(disc_R, disc_C, gen_C, gen_R, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    # visualize training process with tqdm
    loop = tqdm(loader, leave=True)
    real_next = 0 
    
    
    for idx, (cartoon, real, blurry) in enumerate(loop):
        cartoon = cartoon.to(config.DEVICE)
        if idx == 0:
            save_image(real*0.5+0.5, "normal.png")
        
        # gen both current and next real frames in video
        real = real.to(config.DEVICE)
        if (idx == 0):
            real_next = real.to(config.DEVICE)
        
        blurry = blurry.to(config.DEVICE)
        
        # train cartoon and real discriminators
        with torch.amp.autocast("cuda"):
            # generate fake real image
            fake_real = gen_R(cartoon)
            
            # run disc on real and fake reals
            D_R_real = disc_R(real)
            D_R_fake = disc_R(fake_real.detach()) # detach to use fake real original
            D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
            D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_R_loss = D_R_real_loss + D_R_fake_loss

            # generate fake cartoon images for both n and n+1 frames
            fake_cartoon = gen_C(real)
            fake_cartoon_next = gen_C(real_next)

            D_C_real = disc_C(cartoon)
            D_C_fake = disc_C(fake_cartoon.detach()) 
            D_C_fake_next = disc_C(fake_cartoon_next.detach())

            D_C_blurry = disc_C(blurry)
            D_C_blurry_loss = mse(D_C_blurry, torch.zeros_like(D_C_blurry))
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

        # train cartoon and real generators
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
                cycle_cartoon_loss * config.CYCLE_WEIGHT + 
                cycle_real_loss * config.CYCLE_WEIGHT + 
                identity_real_loss * config.IDENTITY_WEIGHT +
                identity_cartoon_loss * config.IDENTITY_WEIGHT + 
                cycle_real_next_loss * config.CYCLE_WEIGHT +
                identity_real_next_loss * config.IDENTITY_WEIGHT +
                content_cartoon_loss * config.CONTENT_WEIGHT + 
                content_real_loss * config.CONTENT_WEIGHT +
                content_cartoon_next_loss * config.CONTENT_WEIGHT
            )
                        
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
        
        # save ALL images since this is for video style transfer
        fname = str(idx)
        while len(fname) < 3:
            fname = "0" + fname
        
        save_image(fake_cartoon*0.5+0.5, f"fourth_vid_frames_no_temporal/{fname}.png")
        real_next = real.to(config.DEVICE)
        

def main():
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
        root_cartoon = config.CARTOON_TRAIN_DIR+"/cartoon/disney_people_processed", root_real = config.REAL_TRAIN_DIR, root_blurry = config.CARTOON_TRAIN_DIR+"/blurry/disney_blurry", transform=config.transforms,
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
    
    for epoch in range(config.NUM_EPOCHS):
        if (config.INITIALIZE == True):
            initialization(disc_R, disc_C, gen_C, gen_R, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        
        else:
            train_fxn(disc_R, disc_C, gen_C, gen_R, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_R, opt_gen, filename=config.CHECKPOINT_SAVE_GEN_R)
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_SAVE_GEN_C)
            save_checkpoint(disc_R, opt_disc, filename=config.CHECKPOINT_SAVE_CRITIC_R)
            save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_SAVE_CRITIC_C)

# run the main file
if __name__ == "__main__":
    main()