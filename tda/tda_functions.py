# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:05:41 2022

@author: basti
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as savgol 
import scipy.optimize as optimize
from numpy.linalg import solve
from scipy.spatial.distance import squareform, pdist

def catshow(imlist, namelist = ['',], normalize = False, axis = 1, cmap = 'gray'):
    name = []
    for k in namelist:
        name.append(k)
        name.append(' - ')
    name = ''.join(name)
    
    if normalize:
        for ia, a in enumerate(imlist):
            amin = np.amin(a)
            amax = np.amax(a)
            a = (a-amin)/(amax-amin)
            imlist[ia] = a#/np.std(a)
    fig = np.concatenate(imlist, axis = axis)
    
    plt.figure(name)
    plt.imshow(fig, cmap = cmap)
    plt.show()

def ortho_poly(x, deg=3):
    x = np.asarray(x)
    n = deg + 1
    xm = np.mean(x)
    x = x - xm
    X = np.fliplr(np.vander(x, n))
    q,r = np.linalg.qr(X)
    z = np.diag(np.diag(r))
    raw = np.dot(q, z)
    norm2 = np.sum(raw**2, axis=0)
    Z = raw / np.sqrt(norm2)
    return Z.T

def base_poly(spc, spc_wl = None, deg = 4, lam = 5, poly_sav = 4, window_sav = 5, return_filtered = False):
    
    if type(spc_wl)==type(None):
        spc_wl = np.arange(spc.shape[-1])
        
    deg = deg+1
        # calculating the orthonormal polygons
        
    Z = ortho_poly(spc_wl, deg=deg)
    
    n_spc = len(spc)
    
    # calculating the savgol filter
    S0 = savgol(spc,
            window_length = window_sav,
            polyorder = poly_sav,
            deriv = 0,
            delta = 1,
            axis = 1)
    
    S1 = savgol(spc,
            window_length = window_sav,
            polyorder = poly_sav,
            deriv = 1,
            delta = 1,
            axis = 1)
    S2 = savgol(spc,
            window_length = window_sav,
            polyorder = poly_sav,
            deriv = 2,
            delta = 1,
            axis = 1)
    
    # calculating the positions of local minima
    
    Pos = np.array(np.where(np.diff(np.sign(S1), axis = 1 ) > 0))
    Pos = [Pos[1][Pos[0]==i] for i in range(n_spc)]
    
    baseline = []
    
    for s0, s1, s2, pos in zip (S0, S1, S2, Pos):
        iMax = pos+ s1[pos] / ( s1[pos] - s1[pos+1] ) 
        
        # calculating the values of the local minima
        P0_0, P0_1, P1_0, P1_1, P2_0, P2_1 = s0[pos], s0[pos+1], s1[pos], s1[pos+1], s2[pos], s2[pos+1]
        X1_0, X1_1 = pos, (pos+1)
        X2_0, X2_1 = X1_0**2, X1_1**2
        X3_0, X3_1 = X1_0*X2_0, X1_1*X2_1
        X4_0, X4_1 = X2_0**2, X2_1**2
        X5_0, X5_1 = X4_0*X1_0, X4_1*X1_1
        
        X0 = np.zeros(len(X1_0))
        
        bigmat = np.array([
         [X0+1, X1_0,   X2_0,   X3_0,    X4_0,    X5_0],
         [X0+1, X1_1,   X2_1,   X3_1,    X4_1,    X5_1],
         [X0  , X0+1, 2*X1_0, 3*X2_0, 4* X3_0, 5* X4_0],
         [X0  , X0+1, 2*X1_1, 3*X2_1, 4* X3_1, 5* X4_1],
         [X0  , X0  ,   X0+2, 6*X1_0, 12*X2_0, 20*X3_0],
         [X0  , X0  ,   X0+2, 6*X1_1, 12*X2_1, 20*X3_1]
         ])
        
        X = np.array([bigmat[:,:,i] for i in range(bigmat.shape[-1])])
        
        bigmat = np.array([
         P0_0, P0_1, P1_0, P1_1, P2_0, P2_1
        ])
        
        Y = np.array([bigmat[:,i] for i in range(bigmat.shape[-1])])
        res = solve(X, Y)
        yMax = np.sum(res*np.array([iMax**i for i in range(6)]).T, axis = 1)
        
        iF = np.floor(iMax).astype(int)
        iC = iF+1
        Z_loc = Z[:,iF] + (Z[:,iC]-Z[:,iF])*(iMax-iF)
        
        # curve fitting with constraints
        
        def residuals_expit(p):
            val = yMax - np.sum(np.asarray([a*z for a,z in zip(p, Z_loc)]), axis = 0)
            val_neg = val.__copy__()
            val_neg[val_neg>0]=0
            return np.abs(val) - lam * val_neg
        
        out = optimize.leastsq(
            residuals_expit, x0 = np.ones(deg))[0]
        
        line = np.sum(np.asarray([o*z for o,z in zip(out, Z)]), axis = 0)
        
        baseline.append(line)
    
    if return_filtered:
        return S0-baseline
    else:
        return spc-baseline
    
def poolNd(im, windowwidth = 3, fun = np.median):
    if not isinstance(windowwidth, (list, tuple, np.ndarray)):
        ww = windowwidth*im.ndim
    elif len(windowwidth) == im.ndim:
        ww = windowwidth
    else:
        'windowwidth not valid'
        return False

    shape = im.shape
    npx = [(s//w)*w for s, w in zip(shape, ww)]
    dnpx = [(s-p)//2 for s, p in zip(shape, npx)]
    nwin = [p//w for p, w in zip(npx, ww)]
    data_red = im.__copy__()[tuple([slice(d,d+p) for d, p in zip(dnpx, npx)])]
    
    reshape = nwin.copy()
    for index, item in enumerate(ww):
            insert_index = index*2 + 1
            reshape.insert(insert_index, item)

    data_red = fun(data_red.reshape(*reshape), axis = tuple(np.arange(1,1+len(ww)*2,2).astype(int)))
    
    return data_red


def fun_dist_lens(data, metric_name='cosine', nLensSteps=10):
    data_rs = np.reshape(data, (np.prod(data.shape[:-1]),data.shape[-1]))
    data_id = list(np.arange(len(data_rs)))
    
    dist = squareform(pdist(data_rs, metric_name))
    
    lensList = []
    
    mymax = np.amin([nLensSteps, dist.shape[0]])
    sList = [np.sum(dist[i][np.argsort(dist[i])[:mymax]]) for i in data_id]
    iStart = data_id[np.argmin(sList)]
    lensList.append(data_id[iStart])
    distList = [0]
    distOld = 0
    dist_lens = dist.__copy__()
    id_lens = list(np.array(data_id).__copy__())
    data_id = np.array(data_id)
    
    while(dist_lens.shape[0]>1):
        iOld = iStart
        dList = list(dist_lens[iOld])
        dList = dList[:iOld]+dList[iOld+1:]
        iList = np.argsort(dList)[:mymax]
        
        id_lens.remove(id_lens[iOld])
        dist_lens = np.delete(np.delete(dist_lens, iOld, axis = 0), iOld, axis = 1)
        
        sList = [np.sum(dist_lens[i][np.argsort(dist_lens[i])[:mymax]]) for i in iList]
        
        iStart = iList[np.argmin(sList)]
        lensList.append(id_lens[iStart])
        distOld = distOld + dist[iOld, id_lens[iStart]]
        distList.append(distOld)
    
    lensIm = np.zeros(len(lensList))
    #lensIm[lensList] = np.array(distList)[lensList]
    lensIm[lensList] = distList
    return distList, lensIm, lensList

def get_coords(imshape, index):
    
    n_dim = len(imshape)
    t = index
    xl = []
    
    for it in range(n_dim):
        p = np.prod(imshape[it+1:])
        x = int(t // p)
        t = t - x * p
        xl.append(x)
    
    return tuple(xl)

def extract(im, coords, widths):
    shape = im.shape
    tup = (slice(max((0,cc-ww)), min((cc+ww+1, ss)), 1) for cc, ww, ss in zip(coords, widths, shape))
    return list(np.asarray(im[tuple(tup)]).flatten())

def flatten_unique(t):
    return list(np.sort(np.unique([item for sublist in t for item in sublist])))
