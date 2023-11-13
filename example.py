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
import matplotlib.pyplot as plt
import os

### import of the tda package
from TopologicalDataAnalysis.tda import tda

### Import of training data
path_dir = os.path.dirname(os.path.abspath(__file__))
spc_in = np.load(join(path_dir, 'TopologicalDataAnalysis', 'resources', 'raman.npy'))

shape_im = spc_in.shape[:2]
n_wl = spc_in.shape[-1]

spc_train = np.reshape(spc_in, (np.prod(shape_im), n_wl))

### initialize tda
tda = tda(
    lens_function = 'PCA', lens_axis = 1, lens_norm = 2,
    resolution = 70, gain = 4,
    cluster_function = 'linkage', cluster_metric = 'cosine', cluster_method = 'average', cluster_tmax_rel = 0.2
    )
c = tda.fit(spc_train)

plt.figure()
plt.imshow(np.reshape(tda.lens.values, shape_im))
#%%

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, maxdists
from scipy.spatial.distance import pdist

idd = pdist(my_data, tda.cluster_metric)
idd[np.isnan(idd)] = 0
Z = linkage(idd, tda.cluster_method)

fig = plt.figure(figsize=(25, 10))

dn = dendrogram(Z)
md = maxdists(Z)