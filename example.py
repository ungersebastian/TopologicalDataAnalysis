# -*- coding: utf-8 -*-
"""
A Python implementation of the topological data analysis clustering method.
@author: ungersebastian
"""

#%%
quit()


#%% Examaple on Raman hyperspectral map

### imports of standard libs
from os.path import join
import numpy as np
import matplotlib.pyplot as plt #### NEED version 2.2.3
import os

### import of the tda package
from TopologicalDataAnalysis.tda import tda

### Import of training data
path_dir = os.path.dirname(os.path.abspath(__file__))
spc_in = np.load(join(path_dir, 'TopologicalDataAnalysis', 'resources', 'raman.npy'))

shape_im = spc_in.shape[:2]
n_wl = spc_in.shape[-1]

spc_train = np.reshape(spc_in, (np.prod(shape_im), n_wl))

#%%
### initialize tda
tda = tda(
    lens_function = 'PCA', lens_axis = 0, lens_norm = 2,
    resolution = 70, gain = 4,
    cluster_function = 'kmeans', cluster_metric = 'cosine', cluster_method = 'average', cluster_t = 4
    )
c = tda.fit(spc_train)

plt.figure()
plt.imshow(np.reshape(tda.lens.values, shape_im))
#%%
tda.draw_network()
tda.draw_cluster_network()

#%%
import networkx as nx
import itertools
from skimage import exposure

cl = set(tda.partition.values())
part = np.asarray(list(tda.partition.values()))
ids = nx.get_node_attributes(tda.topograph, 'ids')
id_topo = np.asarray(list(ids.keys()))

clmap = []
for ic in cl:
    nodes_c = id_topo[part==ic]
    my_nodes = []
    for node in nodes_c: my_nodes.append(list(ids[node]))
    my_nodes = list(set(itertools.chain(*my_nodes)))
    if len(my_nodes)>1:
        
        as_map = np.zeros(np.prod(shape_im))
        as_map[my_nodes] = 1
        
        as_map = as_map * np.sum(spc_train, axis = -1)
        as_map = (as_map-np.amin(as_map))/(np.amax(as_map)-np.amin(as_map))
        as_map = exposure.equalize_hist(as_map)
        im = np.reshape(as_map, shape_im)
        clmap.append(im)
        
        if np.sum(my_nodes)>2:
            plt.figure()
            plt.imshow(im)

#%%
tda.draw_network()
tda.draw_cluster_network()

import networkx as nx
import itertools
from skimage import exposure

cl = set(tda.partition.values())
part = np.asarray(list(tda.partition.values()))
ids = nx.get_node_attributes(tda.topograph, 'ids')
id_topo = np.asarray(list(ids.keys()))

clmap = []
for ic in cl:
    nodes_c = id_topo[part==ic]
    my_nodes = []
    for node in nodes_c: my_nodes.append(list(ids[node]))
    my_nodes = list(set(itertools.chain(*my_nodes)))
    if len(my_nodes)>0:
        
        as_map = np.zeros(np.prod(shape_im))
        as_map[my_nodes] = 1
        
        clmap.append(as_map)

reddim = np.array(clmap).T
from sklearn.decomposition import PCA
pca_fun = PCA(n_components=20)
pca_fun.fit(reddim)
values = pca_fun.transform(reddim).T

evr = pca_fun.explained_variance_ratio_
values = values[evr > 0.01]

i =0
for v in values:
    print(i)
    i=i+1
    #as_map = v * np.sum(spc_train, axis = -1)
    as_map=v
    #as_map = (as_map-np.amin(as_map))/(np.amax(as_map)-np.amin(as_map))
    #as_map = exposure.equalize_hist(as_map)
    im = np.reshape(as_map, shape_im)
        
    plt.figure()
    plt.imshow(im)
