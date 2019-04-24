# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 04:59:58 2016

@author: o-4

File used for the collection of meta data from candidate data sets 
"""

import numpy as np
import metaCollector as mc
from scipy.stats import kurtosis, skew, entropy
import time
import math

def extractFeatures(dset):
    now = time.time()
    
    sawm = mc.setAveragedWeightedMean(dset)
    sacv = mc.setAveragedCoefficientOfVariation(dset)
    saskew = mc.setAveragedSkew(dset)
    sakurt = mc.setAveragedKurtosis(dset)
    saent = mc.setAveragedEntropy(dset)

    future = time.time()

    duration = future - now 
    
    return sawm, sacv, saskew, sakurt, saent, duration

# Probagbility Density Moments 
def setAveragedWeightedMean(dset):
    """Finds weighted mean for each column in given dataset 
    then returns the average of these weighted means"""
    cwm = 0 #combined weighted mean

     
    for col in range(len(dset[0])):
        column = []

        try: 
            for row in dset:
                column.append(row[col])
        except IndexError as ex:
            print "Encountered index error in set, filling in with a zero"
            column.append(0)            
                
        col_mean = np.mean(column)

        if max(column) != 0:
            w_mean = col_mean/max(column)
        else:
            w_mean = 0

        cwm += w_mean
                            
    return cwm/len(dset[0])

def setAveragedCoefficientOfVariation(dset):
    """Finds the coefficient of variation for each column in given dataset
    then returns the average of these values"""
    ccv = 0 #combined coefficient of variation 

    for col in range(len(dset[0])):
        column = []
        
        try: 
            for row in dset:
                column.append(row[col])
        except IndexError as ex:
            print "Encountered index error in set, filling in with a zero"
            column.append(0)            
                
        col_std = np.std(column)
        col_mean = abs(np.mean(column))
        if col_mean != 0:
            col_cv = col_std/col_mean
        else:
            col_cv = 0
        ccv += col_cv
        
    return ccv/len(dset[0])

def setAveragedSkew(dset):
    """Finds the skew of each column in given dataset then 
    returns the average of these values"""
    csk = 0 #combined skew 

    for col in range(len(dset[0])):
        column = []
        
        try: 
            for row in dset:
                column.append(row[col])
        except IndexError as ex:
            print "Encountered index error in set, filling in with a zero"
            column.append(0)            
                            
        col_skew = skew(column)       
        csk += col_skew
        
    return csk/len(dset[0])

def setAveragedKurtosis(dset):
    """Finds the kurtosis of each column in given dataset then 
    returns the average of these values"""
    cku = 0 #combined kurtosis 

    for col in range(len(dset[0])):
        column = []
        
        try: 
            for row in dset:
                column.append(row[col])
        except IndexError as ex:
            print "Encountered index error in set, filling in with a zero"
            column.append(0)            
                            
        col_kurt = kurtosis(column)       
        cku += col_kurt
        
    return cku/len(dset[0])

def setAveragedEntropy(dset):
    """Finds the shannon entropy of each column in a given dataset then 
    returns the average of these values"""
    cse = 0 #combined shannon entropy 

    for col in range(len(dset[0])):
        column = []

        try: 
            for row in dset:
                column.append(row[col])
        except IndexError as ex:
            print "Encountered index error in set, filling in with a zero"
            column.append(0)            
                           
        col_cse = entropy(column)
        
        if math.isnan(col_cse) or math.isinf(col_cse):
            cse += 0    
        else:
            cse += col_cse
                    
    return cse/len(dset[0])
    
    
# def n_moment(dset,nc,N):
#     tm = np.zeros((1,nc))
#     tm = tm[0]
#     mean = mc.mean(dset,nc)
#     for i in dset:
#         num = 0
#         for j in i:
#             tmp = (j-mean[num])**N
#             tm[num] = tm[num] + tmp
#             num = num + 1
#     tm = tm/len(dset)
#     return tm
    
# def p_skew(dset, nc):
#     sd = mc.standard_deviation(dset,nc)
#     tm = mc.n_moment(dset,nc, 3)
#     sk = tm/(sd**3)
#     return sk 
    
    
# def p_kurtosis(dset, nc): 
#     sd = mc.standard_deviation(dset,nc)
#     fm = mc.n_moment(dset,nc,4)
#     ku = fm/sd**4
#     return ku
