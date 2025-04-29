import cv2
import time

def get_frames():
    path = "videos/zoomed_in.mp4"

    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1

    start = time.time()
    
    while success:
        success, image = vidObj.read()
        str_num = str(count)
        while len(str_num) < 3:
            str_num = "0" + str_num

        fname = "fourth_vid_frames/" + str_num + ".png"
        #       <class 'numpy.ndarray'>

        cv2.imwrite(fname, image)
        count += 1
        
    end = time.time()
    print(f"This took {end-start} seconds")

get_frames()