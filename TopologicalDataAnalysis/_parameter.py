# -*- coding: utf-8 -*-
"""
@author: basti
"""

class _parameter(dict):
    def __init__(self, parent, *arg,**kwargs):
        super(_parameter, self).__init__(*arg, **kwargs)
        
        self.parent = parent
        
        self.add('lens_function', 'PCA')
        self.add('lens_norm', 2)
        self.add('lens_axis', 0)
        self.add('lens_center', True)
        
        # check kwargs
        self.update(**kwargs)
    
    def update(self, **kwargs):
        keys = list(self.keys())
        for kw in kwargs:
            if kw in keys:
                self[kw] = kwargs[kw]
        
    
    def add(self, key, val):
        self[key] = val
        
    def load(self):
        pass
    def save(self):
        pass