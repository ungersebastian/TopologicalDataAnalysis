# -*- coding: utf-8 -*-
"""
@author: basti
"""

from .lens_functions.pca import pca

class _lens(object):
    def __init__(self, parent):
        super(_lens, self).__init__()
        self.parent = parent
        
        self.parameter = self.parent.parameter
        
        self.lens_function = parent.lens_function
        
        if self.parameter['lens_function'] == 'PCA':
            self._lens = pca(self)
            self.values = self._lens.values
            self.predict = self._lens.predict
        
