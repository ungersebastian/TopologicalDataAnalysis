# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:48:41 2023

@author: basti
"""

from .lens_functions.pca import pca

class _lens(object):
    def __init__(self, parent):
        super(_lens, self).__init__()
        self.parent = parent
        
        self.lens_function = parent.lens_function
        
        if self.lens_function == 'PCA':
            self.values, self.predict = pca(self.parent.X)
        
