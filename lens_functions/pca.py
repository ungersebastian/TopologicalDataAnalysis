# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:14:41 2023

@author: basti
"""
from sklearn.decomposition import PCA

import numpy as np

from .._utils import norm, center, apply_norm

def pca(X):
    pca = PCA(n_components=1)
    
    my_X = apply_norm(X, norm(X, 2))
    my_X = center(my_X)
    
    pca.fit(my_X)
    
    return pca.transform(my_X), pca.transform