# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 21:47:06 2016

@author: root
"""

import csv
import fparser
import scipy
import random as rn
import numpy as np
import matplotlib.pyplot as plt
#import importlib.util
from sklearn import datasets
from sklearn.gaussian_process.gpc import GaussianProcessClassifier as gp
from sklearn.gaussian_process.kernels import RBF



#spec = importlib.util.spec_from_file_location("GaussianProcessClassifier", "/usr/local/lib/python2.7/site-packages/sklearn/gaussian_process/gpc.py")
#gp = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(gp)

dname = './data/uci/adult/adult.data'
dina, dinb = fparser.parse_file(0,dname, 1, .01)

#testV = []
#for i in range(len(din[0])):
#    testV.append(din[rn.randrange(len(din))][i])
#da.append(testV)

X = []
Y = []
X_t = []
Y_t = []

for i in dina:
    if i != []:
        X.append(i[0:len(i)-1])
        Y.append(i[len(i)-1])
X = np.matrix(X).astype(np.float)
Y = np.array(Y).astype(np.float)

for i in dinb:
    if i != []:
        X_t.append(i[0:len(i)-1])
        Y_t.append(i[len(i)-1])
X_t = np.matrix(X_t).astype(np.float)
Y_t = np.array(Y_t).astype(np.float)



kernel = 1.0 * RBF([1.0])
pc_rbf_isotropic = gp(kernel=kernel).fit(X_t, Y_t)
#kernel = 1.0 * RBF([1.0, 1.0])
#gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(X, Y)

count = 200000
tot = len(dinb)
X_f = []
for j in range(count):
    x = []
    for i in range(len(dina[0])-1):
        ran = rn.randint(0, tot-1)
        x.append(dina[ran][i])
def popdatabase():
    dbh.craftSystem('Welcome07')
    
    x=np.matrix(x).astype(np.float)
    x.reshape(1, -1)
    p = pc_rbf_isotropic.predict(x)
    b = x.tolist()
    bl = p.tolist()
    b[0].append(bl[0])
    X_f.append(b[0])
X_w = dina + dinb + X_f
nfile = dname + '.falsedata'
w1 = open(nfile,'wb')
wr1 = csv.writer(w1, delimiter = ' ')
wr1.writerows(X_w)