# -*- coding: utf-8 -*-
"""
@author: basti
"""

from .._tda_child_class import tda_child
from .._utils import norm, center, apply_norm


from sklearn.decomposition import PCA

class pca(tda_child):
    def __init__(self, parent):
        super(tda_child, self).__init__()
        self.parameter = parent.parameter
        self.parent = parent
        
        
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
        
        
        
    
    
    
    
    
    