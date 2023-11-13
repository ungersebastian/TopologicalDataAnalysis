# -*- coding: utf-8 -*-
"""
@author: basti
"""

from .._tda_child_class import tda_child

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist



class tda_linkage(tda_child):
    def __init__(self, parent):
        super(tda_child, self).__init__()
        self.parameter = parent.parameter
        self.parent = parent
        
    def cluster(self, my_id, my_data, my_lens):
        
        idd = pdist(my_data, self.parent.cluster_metric)
        idd[np.isnan(idd)] = 0
        Z = linkage(idd, self.method)
        
        #cluster analyse
        # calculate the median/mean distance in that subset        
        lens_dist = np.median(np.sort(my_lens)[1:]-np.sort(my_lens)[:-1])
        # clustering of the distance matrix
        c = fcluster(Z, t = self.cluster_threshold*lens_dist, criterion='distance')
        # amount of clusters found
        n0=len(np.unique(c))
        
        print(n0)
        
        return None
        
        
        
    
    
    
    
    
    