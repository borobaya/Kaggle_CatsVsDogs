# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:55:10 2015

@author: mdmiah
"""

# Useful URLs:
# https://gilscvblog.wordpress.com/2013/08/23/bag-of-words-models-for-visual-categorization/
# https://github.com/Itseez/opencv/blob/master/samples/python2/find_obj.py
# http://answers.opencv.org/question/17460/how-to-use-bag-of-words-example-with-brief-descriptors/

# Parameters
trainSize = 12499 # Index of last image
testSize = 1
m = 100 # Maximum number of images per animal to process
t = 5 # Number of seconds to show each set of images for
n_features= 400
voca_size = 50 # Size of Bag of Words vocabulary

# Import packages
import numpy as np
import pandas as pd
# OpenCV
import cv2
# Other
import gc; gc.enable()
import random
import time

start = time.time()

# Clear anything from previous runs of the script
cv2.destroyAllWindows()

# Paths
path_train = "Data/train/"
path_test = "Data/test1/"

# ---------------------------------- Methods ----------------------------------

# Load classes used
detector = cv2.SIFT(n_features)
#detector = cv2.ORB(n_features)
detector = cv2.BRISK()

def getImage(animal, i):
    path = path_train + animal + "." + str(i) + ".jpg"
    img = cv2.imread(path)
    if img.size==0:
        return None
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def addImageDescriptors(img):
    # Features
    kp, des = detector.detectAndCompute(img, None)
    if des is None:
        return None
    img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
    return img2

def showImage(img):
    cv2.imshow("image", img)
    cv2.waitKey(100)

def processImage(animal, i):
    img = getImage(animal, i)
    if img is not None:
        img = addImageDescriptors(img)
    if img is not None:
        showImage(img)

# ----------------------------------        ----------------------------------

# Load cats and display them
print "Processing cats..."
dispTime = time.time()
for i in random.sample(range(0,trainSize), m):
    processImage("cat", i)
    if time.time()-dispTime>t:
        break;

# Load dogs and display them
print "Processing dogs..."
dispTime = time.time()
for i in random.sample(range(0,trainSize), m):
    processImage("dog", i)
    if time.time()-dispTime>t:
        break;

cv2.destroyAllWindows()







