import random
import numpy as np
from glob import glob
import matplotlib.pylab as plt
import pickle

# save train/test indices on all the celeba face data
# determine which celeba faces are going to be training data and which ones are going to be testing
def store_data(train, test):
    db = {}
    db["train"] = train
    db["test"] = test

    dbfile = open("train_test_inds", "ab")
    pickle.dump(db, dbfile)
    dbfile.close()

# load up each img 
def get_inds():
    globname = f'../celeb_faces_unprocessed/*.jpg' 
    
    # create list of image file names
    img_file = glob(globname)
    inds = [i for i in range(15000)]

    # calc number of images for test/train sets
    lentest = int(len(inds) * 0.2)
    lentrain = len(inds) - lentest

    # randomly select image indices for test set
    test_inds = random.sample(inds, k=lentest)
    train_inds = [i for i in range(len(inds)) if i not in test_inds]

    store_data(train_inds, test_inds)

    print("saved!")

get_inds()