# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:18:18 2021

@author: basti
"""

from sklearn.decomposition import PCA
import numpy as np

def norm(data, pNorm = None, axis = 1, centered = False):
    if norm == None:
        return data
    if len(data)==0:
        print('Warning: data must be given and an array')
        return None
    else:
        data = np.array(data)
    if pNorm == 'var':
        data_norm = (np.var(data, axis = axis))
    if pNorm == 'std':
        data_norm = (np.std(data, axis = axis))
    elif pNorm == np.infty:
        data_norm = np.amax(np.abs(data), axis = axis)
    elif pNorm > 0:
        data_norm = (np.sum(np.abs(data)**pNorm, axis = axis))**(1/pNorm)
    
    else:
        print('Warning: norm not implemented')
        return data
    
    data = np.array([d/n if n > 0 else np.zeros(len(d)) for d, n in zip(data, data_norm)])    
    
    if centered == True:
        data = np.array([d-np.mean(d) for d in data])
    return data
            
class pca(object):
    def __init__(self, data = [], nComp = None, center = True, pNorm = None, normAxis = 1, *args, **kwargs):   
        self.warn = 0
        if len(data)==0:
            print('Warning: data must be given and an array')
            self.warn = 1
        else:
            data = np.array(data)
        if nComp == None:
            print('Warning: nComp must be given')
            self.warn = 1
        if self.warn == 0:
            if pNorm != None:
                self.pNorm = pNorm
                data = norm(data = data, pNorm = pNorm, axis = normAxis)
            else:
                self.pNorm = None
            if center == True:
                self.center = np.mean(data, axis = normAxis-1)
                data = data-self.center
            else:
                self.center=[]
                
            self.normAxis = normAxis
            self.pca = PCA(nComp, *args, **kwargs)
            self.scores = self.pca.fit(data).transform(data).T
        
        
    def transform(self, newData):
        if self.pNorm != None:
            newData = norm(data = newData, pNorm = self.pNorm, axis = self.normAxis)
        if len(self.center) > 0:
            newData = newData-self.center
        return self.pca.transform(newData)
    
    def __getattribute__(self, name):
        if name=='components' or name=='components_':
            return self.pca.components_
        elif name=='explained_variance' or name=='explained_variance_':
            return self.pca.explained_variance_
        elif name=='explained_variance_ratio' or name=='explained_variance_ratio_':
            return self.pca.explained_variance_ratio_
        elif name=='singular_values' or name=='singular_values_':
            return self.pca.singular_values_
        elif name=='mean' or name=='mean_':
            return self.pca.mean_
        elif name=='n_components' or name=='n_components_':
            return self.pca.n_components_
        elif name=='n_features' or name=='n_features_':
            return self.pca.n_features_
        elif name=='n_samples' or name=='n_samples_':
            return self.pca.n_samples_
        elif name=='noise_variance' or name=='noise_variance_':
            return self.pca.noise_variance_
        else:
            return object.__getattribute__(self, name)

        
        
        
        
        
        
        
        
        
        