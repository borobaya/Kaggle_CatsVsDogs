# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:37:50 2015

@author: mdmiah
"""

# Parameters
trainLastIndex = 12499
testLastIndex = 12500
voca_size = 30 # Size of Bag of Words vocabulary
n_features = 400 # for SIFT and ORB
mode = "ORB" # Either "SIFT", "ORB" or "BRISK"

# Import packages
import numpy as np
import pandas as pd
# OpenCV
import cv2
# Machine Learning algorithm
from sklearn import neighbors
# Other
import gc; gc.enable()
import time

start = time.time()

# Paths
path_train = "Data/train/"
path_test = "Data/test1/"

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

def getImage(path):
    img = cv2.imread(path)
    if img.size==0:
        return None
    #img = cv2.resize(img, (500, 500))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def getTrainImage(animal, i):
    path = path_train + animal + "." + str(i) + ".jpg"
    return getImage(path)

def getTestImage(i):
    path = path_test + str(i) + ".jpg"
    return getImage(path)

# ----------------------------------        ----------------------------------

# Process cats to create Bag of Words model
print "Extracting cat features for BoW model..."
for i in range(trainLastIndex):
    img = getTrainImage("cat", i)
    if img is None:
        continue
    kp, des = detector.detectAndCompute(img, None) # Features
    # Bag of Words model
    if des is None:
        continue
    bow.add(np.float32(des)) # store as float32 or else it will give an error

# Process dogs to create Bag of Words model
print "Extracting dog features for BoW model..."
for i in range(trainLastIndex):
    img = getTrainImage("dog", i)
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

# Add training cat set
print "Adding cat training features..."
for i in range(trainLastIndex):
    img = getTrainImage("cat", i)
    kp, des = detector.detectAndCompute(img, None)
    if des is None:
        continue
    bowDes = dextract.compute(img, kp, des)
    X.append(bowDes.flatten())
    y.append(0)

# Add training dog set
print "Adding dog training features..."
for i in range(trainLastIndex):
    img = getTrainImage("dog", i)
    kp, des = detector.detectAndCompute(img, None)
    if des is None:
        continue
    bowDes = dextract.compute(img, kp, des)
    X.append(bowDes.flatten())
    y.append(1)

# Add test image set
print "Adding test features..."
for i in range(1,testLastIndex+1):
    img = getTestImage(i)
    kp, des = detector.detectAndCompute(img, None)
    if des is None:
        XTest.append(np.zeros(voca_size))
        continue
    bowDes = dextract.compute(img, kp, des)
    XTest.append(bowDes.flatten())

X = np.asarray(X)
y = np.asarray(y)
XTest = np.asarray(XTest)

# ----------------------------------        ----------------------------------

def metrics(y, Z):
    ct = pd.crosstab(y, Z, rownames=['actual'], colnames=['preds'])
    print ct
    
    # Precision and recall
    tp = np.double(ct[1][1])
    tn = np.double(ct[0][0])
    fp = np.double(ct[1][0])
    fn = np.double(ct[0][1])
    precision = round(100* tp / (tp+fp) ,1)
    recall    = round(100* tp / (tp+fn) ,1)
    accuracy  = round(100* (tp+tn)/ct.sum().sum() ,1)
    print precision, recall, accuracy

print "Training model..."
clf = neighbors.KNeighborsClassifier(3)
clf.fit(X,y)

print "Predicting on training data..."
Z = clf.predict(X)
metrics(y, Z)

print "Predicting on test data..."
ZTest = clf.predict(XTest)

# ----------------------------------        ----------------------------------

submission = pd.DataFrame(ZTest, columns=["label"])
submission.index += 1
submission.index.name = "id"
submission.to_csv("Cache/submission.csv")

# ----------------------------------   End   ----------------------------------

m, s = divmod((time.time() - start), 60)
print "Time taken to run:", int(m), "minutes", round(s,3), "seconds"
