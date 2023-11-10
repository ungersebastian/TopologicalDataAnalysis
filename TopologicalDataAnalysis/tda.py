# -*- coding: utf-8 -*-
"""TDA clustering"""
# Authors: Sebastian Unger <basti.unger@googlemail.com>
# License: pending

###############################################################################
# Initialization
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils import check_random_state
import numpy as np
from ._lens import _lens
from ._parameter import _parameter

class tda(TransformerMixin, ClusterMixin, BaseEstimator):
    @_deprecate_positional_args
    def __init__(self, *, 
                 copy_x=True, random_state=None,
                 **kwargs
                 ):
        self.parameter = _parameter(self, **kwargs)
        
        self.n_clusters = None
                        
        self.copy_x = copy_x
        self.random_state = random_state
    
    #######################################################################
    # the following will allow the usage of the _parameter class to store #
    # all the parameters for computation                                  #
    #######################################################################
    
    def __getattr__(self, key):
        if key in list(object.__getattribute__(self, 'parameter').keys()):
            return object.__getattribute__(self, 'parameter')[key]
        else:
            return object.__getattribute__(self, key)
        
    def __setattr__(self, key, value):
        if key == 'parameter':
            object.__setattr__(self, key, value)
        elif key in list(object.__getattribute__(self, 'parameter').keys()):
            self.parameter[key] = value
        else:
            object.__setattr__(self, key, value)
    
    def __hasattr__(self, key):
        if key in list(object.__getattribute__(self, 'parameter').keys()):
            return True
        else:
            return object.__hasattr__(self, key)
    
    def __delattr__(self, key):
        if key in list(object.__getattribute__(self, 'parameter').keys()):
            print('Parameter can´t be removed!')
            return None
        else:
            return object.__delattr__(self, key)
        
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
        
        # Step 1: Create Lens
        
        self.lens = _lens(self)
        
        # Step 2: Choose Resolotion, Gain: Create Subsets
        # Step 3: Choose Metric, Clustering Method: Cluster Analysis on subsets
        # Step 4: Create (weighted) network
        # Step 5: Cluster Analysis on the network
        # Step 6: Create soft/hard clustering
    
        return self
   