# -*- coding: utf-8 -*-
"""
@author: basti
"""

from .._tda_child_class import tda_child

from sklearn.cluster import KMeans
import numpy as np

class tda_kmeans(tda_child):
    
    def __init__(self, parent):
        super(tda_child, self).__init__()
        self.parameter = parent.parameter
        self.parent = parent
        
    def cluster(self, data):
        """
        Perform hierarchical clustering on lens subset.
        

        Parameters
        ----------
        data : ndarray
            NxM array of observations.
            
        TDA-Parameters
        --------------
            
        cluster_t: int
            how many max classes relative to datasize
            e.g. 5

        Returns
        -------
        c : ndarray
            Nx1 array of classes.

        """
        n_clusters = np.amin([int(self.cluster_t), len(data)])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data)
        
        c = kmeans.labels_

        return c
    
    def help(self):
        pass
        
        
        
    
    
    
    
    
    