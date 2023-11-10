# -*- coding: utf-8 -*-
"""
@author: basti
"""
from sklearn.decomposition import PCA

from .._utils import norm, center, apply_norm

class pca(object):
    def __init__(self, parent):
        super(pca, self).__init__()
        
        self.parent = parent
        self.parameter = self.parent.parameter
        
        my_X = self.parent.parent.X
        
        self.n_components = self.parameter['lens_axis']+1
        self.norm = self.parameter['lens_norm']
        self.center = self.parameter['lens_center']
        self.component = self.parameter['lens_axis']
        
        self.pca_fun = PCA(n_components=self.n_components)
        
        if not isinstance(self.norm, type(None)):
            my_X = apply_norm(my_X, norm(my_X, self.norm))
            
        self.mean = None
        if self.center:
            my_X, self.mean = center(my_X)
        
        self.pca_fun.fit(my_X)
        
        if self.component == 0:
             self.values = self.pca_fun.transform(my_X)
        else:
             self.values = self.pca_fun.transform(my_X)[:, self.component]
             
    
    def predict(self, X):
        my_X = X.__copy__()
        if not isinstance(self.norm, type(None)):
            my_X = apply_norm(my_X, norm(my_X, self.norm))
        
        if self.center:
            my_X, _ = center(my_X, self.mean)
        
        if self.component == 0:
             return self.pca_fun.transform(my_X)
        else:
             return self.pca_fun.transform(my_X)[:, self.component]
        
        
        
    
    
    
    
    
    