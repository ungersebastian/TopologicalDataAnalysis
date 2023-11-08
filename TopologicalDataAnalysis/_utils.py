# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:33:16 2023

@author: basti
"""

import numpy as np

def norm(X, pNorm = None, axis = 1, centered = False):
    if pNorm == None:
        return None
    if len(X)==0:
        return None
    else:
        X = np.array(X)
    if pNorm == 'var':
        norm = (np.var(X, axis = axis))
    if pNorm == 'std':
        norm = (np.std(X, axis = axis))
    elif pNorm == np.infty:
        norm = np.amax(np.abs(X), axis = axis)
    elif pNorm > 0:
        norm = (np.sum(np.abs(X)**pNorm, axis = axis))**(1/pNorm)
    
    else:
        print('Warning: norm not implemented')
        return None
    
    return norm

def center(X):
    return X - np.mean(X, axis = 0)

def apply_norm(X, norm):
    return np.array([d/n if n > 0 else np.zeros(len(d)) for d, n in zip(X, norm)])
    