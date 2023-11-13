# -*- coding: utf-8 -*-
"""
@author: basti
"""

from ._tda_child_class import tda_child

from .lens_functions.pca import pca

class _lens(tda_child):
    def __init__(self, parent):
        super(tda_child, self).__init__()
        self.parameter = parent.parameter
        self.parent = parent
        
        self.lens_function = parent.lens_function
        
        if self.parameter['lens_function'] == 'PCA':
            self._lens = pca(self)
            self.values = self._lens.values
            self.predict = self._lens.predict
        
