import numpy as np
from glob import glob
import pickle
import cv2

# load train/test indices from preexisting file
def load_data():
    dbfile = open("train_test_inds", "rb")
    db = pickle.load(dbfile)
    dbfile.close()
    return (db["train"], db["test"])

# load up each img 
# find where face is
# crop img
# save to either train/test folder
def get_img():
    globname = f'../celeb_faces_unprocessed/*.jpg' 
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    
    # create list of image file names
    img_file = glob(globname)
    cutoff_len = len("../celeb_faces_unprocessed\\")

    # load preexisting indices
    db = load_data()
    test_inds = db[1]

    for i in range(0, 15000):
        # read in image one by one
        cur_img = cv2.imread(img_file[i])
        array = np.array(cur_img)

        # find where face is in img
        face = cascade.detectMultiScale(
            array,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        if len(face) > 0:
            # x,y is top left corner of bounding box
            x = face[0][0]
            y = face[0][1]
            
            # x,y corner is close to forehead, go back a few pixels to account for that
            n_back = 0
            while (x>0 and y>0 and n_back < 30):
                x-=1
                y-=1
                n_back+=1

            # extend width and height by same length to get square? 
            # cropped = img[start_row:end_row, start_col:end_col]
            length = 0
            while x+length < 178 and y+length < 218:
                length += 1

            crop_img = cur_img[y:y+length, x:x+length]

            # resize by -> resized_image = cv2.resize(image, (width, height))
            resize_img = cv2.resize(crop_img, (256,256))

            if i in test_inds:
                newfname = f"../faces_test/{i}.jpg"
                cv2.imwrite(newfname, resize_img)
            else:
                newfname = f"../faces_train/{i}.jpg"
                cv2.imwrite(newfname, resize_img)

        else:
            print("no face")

    print("done!")

get_img()