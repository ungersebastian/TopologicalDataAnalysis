# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:14:41 2023

@author: basti
"""
from sklearn.decomposition import PCA

from .._utils import norm, center, apply_norm

class pca(object):
    def __init__(self, parent):
        super(pca, self).__init__()
        
        self.parent = parent
        my_X = self.parent.parent.X
        
        pca_fun = PCA(n_components=1)
        my_X = apply_norm(my_X, norm(my_X, 2))
        my_X = center(my_X)
        
        pca_fun.fit(my_X)
        
        self.values = pca_fun.transform(my_X)
        self.predict = pca_fun.transform
    
    
    
    
    
    
    