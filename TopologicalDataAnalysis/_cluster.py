# -*- coding: utf-8 -*-
"""
@author: basti
"""
from ._tda_child_class import tda_child
from .cluster_functions.linkage import tda_linkage

class _cluster(tda_child):
    def __init__(self, parent):
        super(tda_child, self).__init__()
        
        self.parameter = parent.parameter
        self.parent = parent
        
        self.cluster_function = parent.cluster_function
        
        if self.parameter['cluster_function'] == 'linkage':
            self.cluster = tda_linkage(self)
        else:
            self.cluster = None
    