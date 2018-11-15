# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 04:59:58 2016

@author: o-4

File used for the collection of meta data from candidate data sets 
"""

import numpy as np
import metaCollector as mc

def extractFeatures(dset, nc):
    dwmean = mc.weightedMean(dset,nc)
    ds_dev = mc.standard_deviation(dset,nc)
    dpskew = mc.p_skew(dset,nc)
    dkurt = mc.p_kurtosis(dset,nc) 
    return dwmean,ds_dev,dpskew,dkurt

# Probagbility Density Moments 
def mean(dset,nc):
    mean = np.zeros((1,nc))
    mean = mean[0]
    for i in dset:
        num = 0
        for j in i:
            mean[num] = mean[num] + j
            num = num + 1
    mean = mean/len(dset)
    return mean

def weightedMean(dset,nc):
    wmean = np.zeros((1,nc))
    wmean = wmean[0]
    max = np.zeros((1,nc))
    max = max[0]
    for i in dset:
        num = 0
        for j in i:
            if j > max[num]:
                max[num] = j
            wmean[num] = wmean[num] + j
            num = num + 1
    wmean = wmean/len(dset)
    wmean = wmean/max
    return wmean
        
def variance(dset, nc):
    mean = mc.mean(dset,nc)
    var = np.zeros((1,nc))
    var = var[0]
    for i in dset:
        num = 0
        for j in i:
            tmp = (j - mean[num])**2
            var[num] = var[num] + tmp
            num = num + 1
    var = var/len(dset)
    return var

def standard_deviation(dset, nc):
    var = mc.variance(dset,nc)
    sd = var**(1/2)
    return sd

def n_moment(dset,nc,N):
    tm = np.zeros((1,nc))
    tm = tm[0]
    mean = mc.mean(dset,nc)
    for i in dset:
        num = 0
        for j in i:
            tmp = (j-mean[num])**N
            tm[num] = tm[num] + tmp
            num = num + 1
    tm = tm/len(dset)
    return tm
    
def p_skew(dset, nc):
    sd = mc.standard_deviation(dset,nc)
    tm = mc.n_moment(dset,nc, 3)
    sk = tm/(sd**3)
    return sk 
    
    
def p_kurtosis(dset, nc): 
    sd = mc.standard_deviation(dset,nc)
    fm = mc.n_moment(dset,nc,4)
    ku = fm/sd**4
    return ku
    
def classEntropy(dset, nc):
    return 0
    
#Simple Meta Features     
def classNumber(dset, nc): 
    return 0    
    
def examplesToFeatures(dset, nc):
    return 0
