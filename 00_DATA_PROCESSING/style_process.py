import numpy as np
from glob import glob
import cv2

globname = "../disney_people/*.jpg"
img_file = glob(globname)

cascade = cv2.CascadeClassifier("anime-eyes-cascade.xml")

# [[ 949  328   39   39] [1002  350   30   30]]
# file size: 1920 x 1080

count = 0

for i in range(0, len(img_file)):
    img = cv2.imread(img_file[i])
    array = np.array(img)

    eyes = cascade.detectMultiScale(
        array,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(20, 20),
        flags = cv2.CASCADE_SCALE_IMAGE)
    
    if (len(eyes) == 0):
        cur_img = img.copy()
        crop_img = cur_img[660:1260, 240:840]
        resize_img = cv2.resize(crop_img, (256,256))
        newfname = f"../disney_people_processed/{count}.jpg"
        cv2.imwrite(newfname, resize_img)
        count += 1

    for ind in range(len(eyes)):
        cur_img = img.copy()
        x = eyes[ind][0]
        y = eyes[ind][1]
        
        n_back = 0
        while (x>0 and y>0 and n_back < 100):
            x-=1
            y-=1
            n_back+=1

        # extend width and height by same length to get square? 
        # cropped = img[start_row:end_row, start_col:end_col]
        length = 0
        while x+length < 1920 and y+length < 1080 and length < 250:
            length += 1

        crop_img = cur_img[y:y+length, x:x+length]

        # resize by -> resized_image = cv2.resize(image, (width, height))
        resize_img = cv2.resize(crop_img, (256,256))

        newfname = f"../disney_people_processed/{count}.jpg"
        cv2.imwrite(newfname, resize_img)
        count += 1