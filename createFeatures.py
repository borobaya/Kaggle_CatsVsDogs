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
trainTotal = 12499 # Index of last image
trainSize = int(trainTotal*0.75)
testSize = trainTotal-trainSize
voca_size = 30 # Size of Bag of Words vocabulary
n_features = 400 # for SIFT and ORB
mode = "BRISK" # Either "SIFT", "ORB" or "BRISK"

# Import packages
import numpy as np
# OpenCV
import cv2
# Other
import gc; gc.enable()
import random
import time

start = time.time()

# Paths
path_train = "Data/train/"
path_test = "Data/test1/"

# Image indexes to use
catTrainIndexes = random.sample(range(0,trainTotal), trainSize)
dogTrainIndexes = random.sample(range(0,trainTotal), trainSize)
catTestIndexes = random.sample(set(range(0,trainTotal))-set(catTrainIndexes), testSize)
dogTestIndexes = random.sample(set(range(0,trainTotal))-set(dogTrainIndexes), testSize)

# ---------------------------------- Methods ----------------------------------

bow = cv2.BOWKMeansTrainer(voca_size)

detector = None
if mode=="SIFT":
    detector = cv2.SIFT(n_features)
elif mode=="ORB":
    detector = cv2.ORB(n_features)
elif mode=="BRISK":
    detector = cv2.BRISK()
else:
    print "Please set mode to either SIFT, ORB or BRISK"

def getImage(animal, i):
    path = path_train + animal + "." + str(i) + ".jpg"
    img = cv2.imread(path)
    if img.size==0:
        return None
    #img = cv2.resize(img, (500, 500))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# ----------------------------------        ----------------------------------

# Process cats to create Bag of Words model
print "Extracting cat features for BoW model..."
for i in catTrainIndexes:
    img = getImage("cat", i)
    if img is None:
        continue
    kp, des = detector.detectAndCompute(img, None) # Features
    # Bag of Words model
    if des is None:
        continue
    bow.add(np.float32(des)) # store as float32 or else it will give an error

# Process dogs to create Bag of Words model
print "Extracting dog features for BoW model..."
for i in dogTrainIndexes:
    img = getImage("dog", i)
    if img is None:
        continue
    kp, des = detector.detectAndCompute(img, None) # Features
    # Bag of Words model
    if des is None:
        continue
    bow.add(np.float32(des)) # store as float32 or else it will give an error

# ----------------------------------        ----------------------------------

vocabulary = bow.cluster() # Create the vocabulary with K-means

matcher = None
if mode=="SIFT":
    matcher = cv2.BFMatcher(normType=cv2.NORM_L1)
elif mode=="ORB" or mode=="BRISK":
    vocabulary = np.uint8(vocabulary)
    matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

extractor = cv2.DescriptorExtractor_create(mode)
dextract = cv2.BOWImgDescriptorExtractor(extractor, matcher)
dextract.setVocabulary(vocabulary)

X = []
y = []
XTest = []
yTest = []

# Add training cat set
print "Adding cat training features..."
for i in catTrainIndexes:
    img = getImage("cat", i)
    kp, des = detector.detectAndCompute(img, None)
    if des is None:
        continue
    bowDes = dextract.compute(img, kp, des)
    X.append(bowDes.flatten())
    y.append(0)

# Add training dog set
print "Adding dog training features..."
for i in dogTrainIndexes:
    img = getImage("dog", i)
    kp, des = detector.detectAndCompute(img, None)
    if des is None:
        continue
    bowDes = dextract.compute(img, kp, des)
    X.append(bowDes.flatten())
    y.append(1)

# Add test cat set
print "Adding cat testing features..."
for i in catTestIndexes:
    img = getImage("cat", i)
    kp, des = detector.detectAndCompute(img, None)
    if des is None:
        continue
    bowDes = dextract.compute(img, kp, des)
    XTest.append(bowDes.flatten())
    yTest.append(0)

# Add test dog set
print "Adding dog testing features..."
for i in dogTestIndexes:
    img = getImage("dog", i)
    kp, des = detector.detectAndCompute(img, None)
    if des is None:
        continue
    bowDes = dextract.compute(img, kp, des)
    XTest.append(bowDes.flatten())
    yTest.append(1)

X = np.asarray(X)
y = np.asarray(y)
XTest = np.asarray(XTest)
yTest = np.asarray(yTest)

# ----------------------------------        ----------------------------------

# Save training and test features to file
np.save('Cache/X.npy', X)
np.save('Cache/y.npy', y)
np.save('Cache/XTest.npy', XTest)
np.save('Cache/yTest.npy', yTest)


# ----------------------------------   End   ----------------------------------

m, s = divmod((time.time() - start), 60)
print "Time taken to run:", int(m), "minutes", round(s,3), "seconds"
