# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:51:56 2023

@author: basti
"""


### imports of standard libs
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #### NEED version 2.2.3
import os

### import of the tda package
from TopologicalDataAnalysis.tda import tda




### Import of training data
path_dir = os.path.dirname(os.path.abspath(__file__))

filename = join(path_dir, 'TopologicalDataAnalysis', 'resources', 'spc_export.csv')

def read_csv(filename, chunksize = 10**5):
    
    chunks = pd.read_csv(
        filename,
        chunksize = chunksize,
        header = None,
        sep=',',
        escapechar='\\'
        )
    
    df = pd.concat(chunks)
    
    return df.values

spc_in = read_csv(filename)

print(spc_in.shape)

#%%

max_chunk = 500
resolution = spc_in.shape[0]//max_chunk

my_tda = tda(
    lens_function = 'PCA', lens_axis = 1, lens_norm = 2,
    resolution = resolution, gain = 4,
    cluster_function = 'linkage', cluster_metric = 'cosine', cluster_method = 'average', cluster_tmax_rel = 0.2
    )
my_tda.fit(spc_in)

my_tda.draw_network()
my_tda.draw_cluster_network()