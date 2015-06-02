# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:39:00 2015

@author: mdmiah
"""

# Import packages
import numpy as np
import pandas as pd
# Other
import gc; gc.enable()
import time
# Machine Learning algorithm
from sklearn import neighbors

start = time.time()

# Load training and test features from file
X = np.load('Cache/X.npy')
y = np.load('Cache/y.npy')
XTest = np.load('Cache/XTest.npy')
yTest = np.load('Cache/yTest.npy')

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

clf = neighbors.KNeighborsClassifier(3)
clf.fit(X,y)

Z = clf.predict(X)
metrics(y, Z)

ZTest = clf.predict(XTest)
metrics(yTest, ZTest)

# ----------------------------------   End   ----------------------------------

m, s = divmod((time.time() - start), 60)
print "Time taken to run:", int(m), "minutes", round(s,3), "seconds"
