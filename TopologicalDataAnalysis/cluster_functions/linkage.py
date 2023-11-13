# -*- coding: utf-8 -*-
"""
@author: basti
"""

from .._tda_child_class import tda_child

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, maxinconsts, inconsistent, maxRstat
from scipy.spatial.distance import pdist

class tda_linkage(tda_child):
    def __init__(self, parent):
        super(tda_child, self).__init__()
        self.parameter = parent.parameter
        self.parent = parent
        
    def cluster(self, my_id, my_data, my_lens):
        
        idd = pdist(my_data, self.parent.cluster_metric)
        
        idd[np.isnan(idd)] = 0
        Z = linkage(idd, self.cluster_method)
        
        #cluster analyse
        # calculate the median/mean distance in that subset        
        #lens_dist = np.mean(np.sort(my_lens)[1:]-np.sort(my_lens)[:-1])
        # clustering of the distance matrix
        R = inconsistent(Z, d=2)
        MI = maxinconsts(Z, R)
        t = np.ceil(self.cluster_tmax_rel*len(my_data)).astype(int)
        c = fcluster(Z, t=t, criterion='maxclust_monocrit', monocrit=MI)
        # amount of clusters found
        n0=len(np.unique(c))
        if n0 == 1:
            c = fcluster(Z, t=t//4, criterion='maxclust')
        n0=len(np.unique(c))
        
        """
        # take only clusters, which are big enough
        c_uni = np.unique(c)
        c_o = [np.sum(c==ic)<limit for ic in c_uni]
        c_n = [np.sum(c==ic)>=limit for ic in c_uni]
        c_out = c_uni[c_o]
        c_uni = c_uni[c_n]
        """
        return c
        
        
        
    
    
    
    
    
    