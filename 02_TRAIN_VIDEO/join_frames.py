import cv2
import numpy as np
import os 
from glob import glob

# have to handle images of different sizes for final product 
# make into gif instead of movie? 

def make_video():
    globname = "fourth_vid_frames_no_temporal/*.png"
    video_name = "videos/gen_vid_one.avi"

    images = glob(globname)
    images.sort()

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 24, (width, height))
    print("worked!")

    # loop through all the images in the folder
    for image in images:
        video.write(cv2.imread(image))

    print("worked!")    
    # release the video 
    video.release()
    print("done!")

make_video()