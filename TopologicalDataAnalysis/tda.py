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
import matplotlib.pyplot as plt
import community as community_louvain #pip install python-louvain
from ._tda_child_class import tda_child
from ._lens import _lens
from ._cluster import _cluster
from ._parameter import _parameter

class tda(TransformerMixin, ClusterMixin, BaseEstimator, tda_child):
    @_deprecate_positional_args # Decorator for methods that issues warnings for positional arguments.
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
        
        self.fit_1()
        
        ################################################################
        ## Step 2: Choose Resolotion, Gain: Create Subsets
        ################################################################
        
        self.fit_2()
        
        ################################################################
        ## Step 3: Choose Metric, Clustering Method: Cluster Analysis on subsets
        ################################################################
        
        self.fit_3()
       
        ################################################################
        ## Step 4: Create (weighted) network
        ################################################################
        
        self.fit_4()
        
        ################################################################
        ## Step 5: Cluster Analysis on the network
        ################################################################
        
        self.fit_5()
        
        ################################################################
        ## Step 6: Create soft/hard clustering
        ################################################################
    
        return self
    
    def fit_1(self):
        self.lens = _lens(self)
        
    def fit_2(self):
        self.topograph = nx.Graph()
        self.data_id = np.arange(len(self.X))   
        
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
        
    def fit_3(self):
        self.f_cluster = _cluster(self)
        values = self.lens.values
        i_node = 0
        for subset in self.lens_subsets:
            lens_select = ((values>=subset[0])*(values<=subset[1])).flatten()
            
            # if only 1 datapoint in subset --> this is a node
            if np.sum(lens_select) == 1:
                my_id = self.data_id[lens_select]
                self.topograph.add_node(i_node, ids = my_id, height = len(my_id))
                i_node = i_node+1
            elif np.sum(lens_select) > 1:
                my_id = self.data_id[lens_select]
                my_data = self.X[lens_select]
                my_lens = values[lens_select]
                
                ## perform clustering
                new_nodes = self.f_cluster.cluster.cluster(my_id, my_data, my_lens)
                
                for i_c in np.unique(new_nodes):
                    id_list = my_id[new_nodes==i_c]
                    self.topograph.add_node(i_node, ids = id_list, height = len(id_list))
                    i_node = i_node+1
                
        self.n_nodes = len(self.topograph)   
        
    def fit_4(self):
        rm = []
        for node_a in range(self.n_nodes):
            delete = 1
            l1 = self.topograph.nodes[node_a]['ids']
            for node_b in range(node_a+1, self.n_nodes):
                l2 = self.topograph.nodes[node_b]['ids']
                d = set(l1).intersection(l2)
                if (len(d)>0):
                    self.topograph.add_edge(node_a, node_b, weight=len(d)) 
                    #topo_graph.add_edge(node_a, node_b) 
                    delete = 0
            if delete == 1:
                rm = rm+[node_a]
                
        n_rm = len(rm)
        rm = np.flip(rm)
        for r in rm:
            self.topograph.remove_node(r)
        
        print('n_nodes: ', len(self.topograph), " = ", self.n_nodes , ' - ', n_rm, ' (', self.n_nodes-n_rm, ') ')
        
        self.n_nodes = self.n_nodes-n_rm
        
    def fit_4(self):
        self.partition = community_louvain.best_partition(self.topograph, weight = 'height')
        
    def draw_network(self):
        self.pos_topo = nx.spring_layout(self.topograph, weight = None)
        fig = plt.figure()
        title = 'TOPO - unweighted'
        fig.canvas.set_window_title(title)
        nx.draw_networkx_nodes(self.topograph, pos_topo, node_size=1)
        nx.draw_networkx_edges(self.topograph, pos_topo, alpha=0.4)
        
        plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(self.topograph, 'height').values()))
        plt.title(title)
        plt.show()
            
   