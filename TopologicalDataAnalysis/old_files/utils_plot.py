# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:24:54 2021

@author: basti
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def stack_plot(data, wl=[]):
    nComp = len(data)
    
    dist = np.amax(data[:-1]-data[1:])
    zeros = np.arange(nComp)*dist
    data = np.array([l+z for l,z in zip(data, zeros)])
    
    if len(wl)==0:
        wl = np.arange(data.shape[1])
    
    xMin = np.amin(wl)
    xMax = np.amax(wl)
    
    zeros = np.array([[z,z] for z in zeros])
    
    colors = list(mcolors.BASE_COLORS.keys())
    nCol = len(colors)
    plt.figure()
    for (i, d), z in zip(enumerate(data), zeros):
        plt.plot(wl, d, ''.join([colors[i%nCol],'-']), [xMin,xMax], z, ''.join([colors[i%nCol],'--']))
    plt.show()
        
    
def pairs(data, *args, **kwargs):
    if data.ndim == 2:
        nComp = len(data)
    elif data.ndim == 1:
        nComp = 1
    else:
        print('Warning: data.ndim not valid')
        return None
    if 'stretch_factor' in kwargs:
        stretch_factor = kwargs['stretch_factor']
    else:
        stretch_factor = 0.1
    
    if nComp > 1:
        pRange = np.array([[np.quantile(d,0.01), np.quantile(d,0.99)] for d in data])
        d = (pRange[:,1]-pRange[:,0])*stretch_factor
        pRange[:,0] -= d
        pRange[:,1] += d
        fig, axs = plt.subplots(nComp, nComp)
        
        for i1 in np.arange(0,nComp):
            axs[i1,i1].hist(data[i1],bins = 20, range=(pRange[i1,0],pRange[i1,1]))
            axs[i1,i1].set_xlim(pRange[i1,0],pRange[i1,1])
            axs[i1,i1].set_xlabel(''.join(['PC - ', str(i1+1)]))
            for i2 in np.arange(i1+1, nComp):
                axs[i1,i2].scatter(data[i2], data[i1], alpha=0.7, s = 1)
                axs[i1,i2].set_xlim(pRange[i2,0],pRange[i2,1])
                axs[i1,i2].set_ylim(pRange[i1,0],pRange[i1,1])
                axs[i1,i2].set_xlabel(''.join(['PC - ', str(i2+1)]))
                axs[i1,i2].set_ylabel(''.join(['PC - ', str(i1+1)]))
                axs[i2,i1].scatter(data[i1], data[i2], alpha=0.7, s = 1)
                axs[i2,i1].set_xlim(pRange[i1,0],pRange[i1,1])
                axs[i2,i1].set_ylim(pRange[i2,0],pRange[i2,1])
                axs[i2,i1].set_xlabel(''.join(['PC - ', str(i1+1)]))
                axs[i2,i1].set_ylabel(''.join(['PC - ', str(i2+1)]))
        
        
        plt.show()
    
    
    
    
    
