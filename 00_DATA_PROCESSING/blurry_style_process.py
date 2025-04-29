import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob

# get image folder 
globname = "../disney_people_processed/*.jpg"
cutoff = len("../disney_people_processed/")
img_file = glob(globname)

for i in range(0, len(img_file)):
    img = cv2.imread(img_file[i])

    #` apply canny edge detection to image
    edges = cv2.Canny(img, 100, 200) #img, threshold 1, threshold 2

    # dilate the edges to make them thicker
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel)

    # take mask and apply blurring to regions where the mask == 255
    blur = cv2.blur(img,(7,7),0)
    out = img.copy()
    out[edges > 100] = blur[edges > 100]

    fname = "../disney_blurry/" + img_file[i][cutoff:len(img_file[i])]
    cv2.imwrite(fname, out)
