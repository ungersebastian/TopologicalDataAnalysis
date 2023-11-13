# -*- coding: utf-8 -*-
"""TDA clustering"""
# Authors: Sebastian Unger <basti.unger@googlemail.com>
# License: pending

###############################################################################
# Initialization
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils import check_random_state
import networkx as nx
import numpy as np
from ._tda_child_class import tda_child
from ._lens import _lens
from ._cluster import _cluster
from ._parameter import _parameter

class tda(TransformerMixin, ClusterMixin, BaseEstimator, tda_child):
    @_deprecate_positional_args
    def __init__(self, *, 
                 copy_x=True, random_state=None,
                 **kwargs
                 ):
        self.parameter = _parameter(self, **kwargs)
        
        # for convention of sklearn
        self.n_clusters = None
                        
        self.copy_x = copy_x
        self.random_state = random_state
        
        # epsilon to account for machine precission
        self.eps = 1E-3
    
    
    #######################################################################
    # tda code starts here                                                #
    #                                                                     #
    #######################################################################    
    
    def _check_params(self, X):
        # future: check constraints on X
        pass
    
    
    def fit(self, X):
        """Compute tda clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

            .. versionadded:: 0.0

        Returns
        -------
        self
            Fitted estimator.
        """
        
        self.X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)
        
        self._check_params(self.X)
        random_state = check_random_state(self.random_state)
        
        ################################################################
        ## Step 1: Create Lens
        ################################################################
        
        self.lens = _lens(self)
        
        ################################################################
        ## Step 2: Choose Resolotion, Gain: Create Subsets
        ################################################################
        
        self.topograph = nx.Graph()
        data_id = np.arange(len(X))   
        
        overlap = (self.gain-1)/self.gain
        values = self.lens.values
        minimum = np.amin(values)
        minmax = np.amax(values) - np.amin(values)
        eps = (minmax/self.resolution)*self.eps
        
        # width of each window:
        c_width = (minmax+2*eps)/(1+(self.resolution-1)*(1-overlap))
        # distance between windows
        c_dist  = c_width*(1-overlap)
        
        # create the borders of each lens subset
        self.lens_subsets = [[minimum - eps + r*c_dist, minimum - eps + r*c_dist + c_width] for r in range(self.resolution)]
        
        ################################################################
        ## Step 3: Choose Metric, Clustering Method: Cluster Analysis on subsets
        ################################################################
        
        self.f_cluster = _cluster(self)
        
        i_node = 0
        for subset in self.lens_subsets:
            lens_select = ((values>=subset[0])*(values<=subset[1])).flatten()
            
            # if only 1 datapoint in subset --> this is a node
            if np.sum(lens_select) == 1:
                my_id = data_id[lens_select]
                self.topograph.add_node(i_node, ids = my_id, height = len(my_id))
                i_node = i_node+1
            elif np.sum(lens_select) > 1:
                my_id = data_id[lens_select]
                my_data = X[lens_select]
                my_lens = values[lens_select]
                
                ## perform clustering
                new_nodes = self.f_cluster.cluster.cluster(my_id, my_data, my_lens)
                    
       
        ################################################################
        ## Step 4: Create (weighted) network
        ################################################################
        
        ################################################################
        ## Step 5: Cluster Analysis on the network
        ################################################################
        
        ################################################################
        ## Step 6: Create soft/hard clustering
        ################################################################
    
        return self
   