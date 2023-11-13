# -*- coding: utf-8 -*-
"""
@author: basti
"""

class _parameter(dict):
    def __init__(self, parent, *arg,**kwargs):
        super(_parameter, self).__init__(*arg, **kwargs)
        
        self.parent = parent
        
        # lens parameter
        self.add('lens_function', 'PCA')
        self.add('lens_norm', 2)
        self.add('lens_axis', 0)
        self.add('lens_center', True)
        
        # lens sectioning
        self.add('resolution', 30)
        self.add('gain', 4)
        
        # subset clustering
        self.add('cluster_function', 'linkage')
        self.add('cluster_metric', 'cosine') # euclidean, seuclidean, correlation, cosine)
        if self['cluster_function'] == 'linkage':
            self.add('cluster_method', 'single') # Linkage: single, complete, ward)
        else:
            self.add('cluster_method', 'single') # Linkage: single, complete, ward)
        self.add('cluster_tmax_rel', 0.3)
        self.add('cluster_n_min', 5) # minimum cluster size 
        
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
    
    