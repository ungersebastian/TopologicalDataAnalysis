"""
---------------------------------------------------------------------------------------------------

	@author Sebastian Unger
	@email basti.unger@gmail.com
	@create date 2022-04-01 12:07:12
	@desc [description]

---------------------------------------------------------------------------------------------------
"""

#%% imports
# public
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.signal import medfilt
import networkx as nx
import functools
import csv
from scipy.ndimage import median_filter
from scipy.ndimage.filters import convolve
import community as community_louvain #pip install python-louvain
import matplotlib.cm as cm
import NanoImagingPack as nip
import itertools
from sklearn.decomposition import non_negative_factorization
# private
from IRAFM import IRAFM as ir
from tda_functions import catshow, base_poly, poolNd, fun_dist_lens, get_coords, extract, flatten_unique

"""#####################################################"""

# loading the data
path_project = path.dirname(path.realpath(__file__))



#%% Import of spectra, clipping to the same spectral region, calibration using CaF2

""" general working parameters """
use_calibration = True
use_baseline = False
""""""""""""""

eps = 1E-20     # artificial offset for zero-value spectra - has no impact on the result, just makes things easier

""" select directory and files to be used """
path_resources = path.join(path_project, r'resources\BacillusSubtilis')
files = {
    #'2104_DalVan'   :   ['DalVan',  '2104_DalVan_1351-1659cm-1',    'Dalvan0006.txt',     'avCaF2_DalVan0006.txt'       ], 
    '2105_BacVan30' :   ['BacVan',  '2105_BacVan30_1351-1659cm-1',  'BacVan30_0014.txt',  'avCaF2_forBacVan30_0014.txt' ],
    #'2107_BacVan30' :   ['BacVan',  '2107_BacVan30_1351-1659cm-1',  'BacVan30_0012.txt',  'avCaF2_BacVan30_0012.txt'    ],
    #'2107_Control30':   ['Control', '2107_Control30_1351-1659cm-1', 'Control30_0016.txt', 'avCaF2_forControl30_0016.txt'],
    #'2108_BacVan15' :   ['BacVan',  '2108_BacVan15_1400-1659cm-1',  'BacVan15_0007.txt',  'avCaF2_BacVan15_0007.txt'    ],
    #'2108_BacVan30' :   ['BacVan',  '2108_BacVan30_1400-1659cm-1',  'BacVan30_0011.txt',  'avCaF2_forBacVan30.txt'      ],
    #'2108_BacVan60' :   ['BacVan',  '2108_BacVan60_1400-1659cm-1',  'BacVan60_0013.txt',  'avCaF2_BacVan60_0013.txt'    ],
    #'2108_Control30':   ['Control', '2108_Control30_1400-1659cm-1', 'Control30_0016.txt', 'avCaF2_forControl30.txt'     ],
    #'2108_Control60':   ['Control', '2108_Control60_1400-1659cm-1', 'Control60_0011.txt', 'avCaF2_Control60_0011.txt'   ]
    }
""""""""""""""

keys = list(files.keys())
k = keys[0]
p = path.join(path_resources, files[k][1], files[k][3])
with open(p, 'r') as f:
    csv_reader = csv.reader(f,delimiter='\t')
    lines = []
    for row in csv_reader:
        lines.append(row[1])
    lines = lines[1:]

# analyse the wl-axes, crop to same region

data = {}
lengths = []
for k in keys:
    
    my_data = ir( 
        path.join(path_resources, files[k][1]),
        files[k][2])
   
    wl = my_data['wavelength']
    data[k] = [k[0], my_data, wl]
    
    lengths.append(len(wl))
    
# all spc share the same wl-datapoints --> only clipping/no interpolation
# i've written a fully working calibration script if this is ever needed --> ask me 
# this method here works only if every referennce wl-datapoint is present in the current one!

amin = np.argmin(lengths)
wavelength = data[keys[amin]][2]

for k in keys:
    
    my_data = data[k][1]
    
    wTemp = data[k][2]
    wTemp = np.sum([wTemp == w for w in wavelength], axis = 0)==1
    
    # calibration file
    calib = []
    p = path.join(path_resources, files[k][1], files[k][3])
    with open(p, 'r') as f:
        csv_reader = csv.reader(f,delimiter='\t')
        
        for row in csv_reader:
            calib.append(row[1])
        calib = calib[1:]
    
    calib = np.array(calib).astype(float)[wTemp]
    
    # the spectra

    pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
    hyPIRFwd = np.array(my_data['files'])[pos][0]
    
    spc = np.reshape(hyPIRFwd['data'], (1,*hyPIRFwd['data'].shape))
    
    wTemp = np.reshape(np.repeat(wTemp , np.prod(spc.shape[:-1])), spc.shape, order = 'F')
    
    spc = np.reshape(spc[wTemp], (*spc.shape[:-1], len(wavelength)))
    
    # remove zero value spectra
    
    pos = np.sum(spc, axis = -1) == 0
    spc[pos] = spc[pos]+eps
    
    if use_calibration:
        calib = np.reshape(np.repeat(calib , np.prod(spc.shape[:-1])), spc.shape, order = 'F')
        spc = spc/calib
    
    spc = spc + 1E-10
    
    data[k] = [k[0], spc]

spc_data  = np.array([data[k][1][0] for k in keys])
spc_class = [data[k][0] for k in keys]
spc_keys  = keys
spc_wl    = wavelength

catshow(np.sum(spc_data, -1), ['intensity images: ',]+spc_keys, True)    


del(spc, keys, wTemp, calib, wavelength, data, amin, csv_reader, eps, f, files, hyPIRFwd, k, lengths, lines, my_data, p, pos, row, use_calibration, wl)
#if data.ndim < 4:
#    data = np.reshape(data, (*list(np.ones(4-data.ndim).astype(int)), *data.shape))

nZ,*imshape, vChan = spc_data.shape

if use_baseline:
    res = base_poly(np.reshape(spc_data, (np.prod(spc_data.shape[:-1]), vChan)), spc_wl = spc_wl, deg = 6, lam = 10, poly_sav = 2, window_sav = 3, return_filtered = False)
    res = np.reshape(res, spc_data.shape)
    
    d1 = np.reshape(spc_data.__copy__(), (np.prod(spc_data.shape[:-1]), vChan)) 
    d1 = np.asarray([d/s for d, s in zip(d1, np.sqrt(np.sum(d1**2, axis = -1)))])
    uuu1, uu1, u1, m1, o1, oo1, ooo1 = np.quantile(d1, q = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95), axis = 0)
    
    spc_data = res
    
    d2 = np.reshape(spc_data.__copy__(), (np.prod(spc_data.shape[:-1]), vChan)) 
    d2 = np.asarray([d/s for d, s in zip(d2, np.sqrt(np.sum(d2**2, axis = -1)))])
    uuu2, uu2, u2, m2, o2, oo2, ooo2 = np.quantile(d2, q = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95), axis = 0)
    
    
    plt.figure()
    plt.plot(spc_wl, m1, c = 'orange')
    plt.fill_between(spc_wl, u1, o1, alpha=0.3, facecolor = 'orange')
    plt.fill_between(spc_wl, uu1, oo1, alpha=0.3, facecolor = 'orange')
    plt.fill_between(spc_wl, uuu1, ooo1, alpha=0.3, facecolor = 'orange')
    plt.plot(spc_wl, m2, c = 'green')
    plt.fill_between(spc_wl, u2, o2, alpha=0.3, facecolor = 'green')
    plt.fill_between(spc_wl, uu2, oo2, alpha=0.3, facecolor = 'green')
    plt.fill_between(spc_wl, uuu2, ooo2, alpha=0.3, facecolor = 'green')
    plt.show()

else:
    d1 = np.reshape(spc_data.__copy__(), (np.prod(spc_data.shape[:-1]), vChan)) 
    d1 = np.asarray([d/s for d, s in zip(d1, np.sqrt(np.sum(d1**2, axis = -1)))])
    uuu1, uu1, u1, m1, o1, oo1, ooo1 = np.quantile(d1, q = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95), axis = 0)
    plt.figure()
    plt.plot(spc_wl, m1, c = 'orange')
    plt.fill_between(spc_wl, u1, o1, alpha=0.3, facecolor = 'orange')
    plt.fill_between(spc_wl, uu1, oo1, alpha=0.3, facecolor = 'orange')
    plt.fill_between(spc_wl, uuu1, ooo1, alpha=0.3, facecolor = 'orange')

#%% creating the lens

# downscaling and removal of some suspicios datapoints (ie spikes)

""" select parameters for generating the lens """
metric_name = 'cosine' # euclidean, seuclidean, correlation, cosine
nLensSteps = 100000    # how many spectra should be used while looking for the next lens value ( should be >= 1 )
thresh = 0.1           # threshold for spike removal via median filtering
ww = [1,5,5,1]         # downscaling paramters: z (ie image), y-axis, x-axis, spectral axis
""""""""""""""

if functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, list(map(int,ww)), np.ones(spc_data.ndim).astype(int)), True) : 
    distList, lensIm, lensList = fun_dist_lens(spc_data, metric_name, nLensSteps)
    
    plt.figure()
    plt.imshow(np.reshape(lensIm, (nZ,*imshape))[int((nZ/ww[0])//2)], cmap = 'hsv')
    plt.show()
else :
    pool_mean = poolNd(spc_data, ww, np.mean)
    pool_median = poolNd(spc_data, ww, np.median)
    
    d = np.abs((pool_mean-pool_median)/(pool_mean+pool_median+np.mean(pool_mean)*1E-6))
    print('amount of flattend positions: ',np.sum(d>=thresh)/len(d.flatten()))
    pool_mean[d>=thresh]=pool_median[d>=thresh]
    
    *imshape_red, vChan_red = pool_mean.shape

    data_red = np.reshape(pool_mean, (np.prod(pool_mean.shape[:-1]),pool_mean.shape[-1]))

    #catshow(np.sum(np.reshape(data_red,(*imshape_red, vChan)), -1), ['intensity images, pooling: ',]+spc_keys, True)    

    distList, lensIm, lensList = fun_dist_lens(data_red, metric_name, nLensSteps)
    
    #catshow(np.reshape(lensIm, imshape_red), namelist=['lens, pooling: ',]+spc_keys, normalize=False, cmap = 'hsv')  
    
    #interpolate data to data_red lens values

    iMax = len(distList)
    new_lens = []
    spc = np.reshape(spc_data, (nZ*np.prod(imshape), vChan))
    
    eps = 1E-10
    
    for i_s, s in enumerate(spc):
    
        s = np.reshape(s,(1,vChan))
        dist_s = cdist(s, data_red, metric_name)[0]
        
        min_s = np.argmin(dist_s)
        p_min = lensList.index(min_s)
        
        if p_min == 0:
            vec_min = [p_min, p_min+1]
            dist_s = dist_s[np.asarray(lensList)[vec_min]]
            dist_n = dist_s[1]-dist_s[0]
            
            o = dist_n / distList[1]
            if o < 1:
                dist_s = dist_s / (np.sum(dist_s) + eps)
                
                p = distList[vec_min[0]] * dist_s[1] + distList[vec_min[1]] * dist_s[0]
            else:
                dp = distList[1]-distList[0]
                p = distList[0] - (dist_n-dp)
                
        elif p_min == iMax-1:
            vec_min = [p_min-1, p_min]
            
            dist_s = dist_s[np.asarray(lensList)[vec_min]]
            dist_n = dist_s[0]-dist_s[1]
            o = dist_n / distList[-1]
            if o < 1:
                dist_s = dist_s / (np.sum(dist_s) + eps)
                p = distList[vec_min[0]] * dist_s[1] + distList[vec_min[1]] * dist_s[0]
            else:
                dp = distList[-1]-distList[-2]
                p = distList[-1] + np.abs(dist_n-dp)
        
        else:
            vec_min = [p_min-1, p_min, p_min+1]
            dist_s = dist_s[np.asarray(lensList)[vec_min]]
            
            amin = np.argsort(dist_s)[:2]
            
            dist_s = np.asarray(dist_s)[amin]
            dist_s = dist_s / (np.sum(dist_s) + eps)
            
            p = distList[vec_min[amin[0]]] * dist_s[1] + distList[vec_min[amin[1]]] * dist_s[0]
            
        new_lens.append(p)
        
    new_lens = np.array(new_lens)
    new_lens[np.isnan(new_lens)==True] = 0
    
    catshow(np.reshape(new_lens, (nZ,*imshape)), ['lens, interpolation: ']+spc_keys, False, cmap = 'hsv')    
    
    
dim = (nZ, *imshape)
id_im = np.reshape(np.arange(np.prod(dim)),dim)

#%% sorting the lens values (which are generated via downscaling) again
spc = np.reshape(spc_data.__copy__(), (nZ*np.prod(imshape), vChan))

""" select parameters for lens sorting """
extra_width = (0,8,8) # extra width while diving to the best fit (z,y,x) - dont use to big numbers!
step = 1              # how many values should be generated per iteration? bigger number makes it faster, but should be smaller than extra width
window = 5000         # how many pre sorted spectra should be analysed at the same time (together with extra width) - dont use to big numbers!
""""""""""""""
pre_sort = np.argsort(new_lens)

id_list = []
d_list = [0]
ds_list = [0]

d_old = 0
# das erste Element ist das in der ersten Zeile

i_old = 0

while len(pre_sort)>0:
    
    
    id_sort = pre_sort[:window+1]
    
    #### ND hinzufügen

    # um die vorgewählten Spektren werden die räumlichen Nachbarn hinzugefügt
    extra_id = [get_coords((nZ,*imshape), ids) for ids in id_sort]
    list_extra_id = [extract(id_im, tt, extra_width) for tt in extra_id]
    list_extra_id = flatten_unique(list_extra_id)
    # es werden nur die Spektren genommen, welche noch verfügbar sind
    list_extra_id = list(set(pre_sort).intersection(list_extra_id))
    
    id_sort = np.array(list_extra_id)

    #### Ende ND
    
    sort_dist = squareform(pdist(spc[id_sort], metric = metric_name))
    np.fill_diagonal(sort_dist, np.infty)

    for i_step in range(step):
        
        # alte ID hinzufügen
        t = id_sort[i_old]
        id_list.append(t)
        
        # 1 aktuelle Achse nehmen und auf neue Größe reduzieren
        
        ax = list(sort_dist[:,i_old])
        ax = ax[:i_old]+ax[i_old+1:]
        if len(ax) > 0:
            
            # Position neues Minimum auf reduzierter Achse
            i_new = np.argmin(ax)
            d_new = ax[i_new]
            
            
            # alte Einträge löschen
            sort_dist = np.delete(np.delete(sort_dist, i_old, axis = 0), i_old, axis = 1)
            id_sort = np.delete(id_sort, np.where(id_sort == t)[0][0])
            pre_sort = np.delete(pre_sort, np.where(pre_sort == t)[0][0])
            
            # Distanzen hinzufügen
            d_list.append(d_new)
            ds_list.append(d_old+d_new)
            
            # aktualiserung der Einträge
            d_old = d_old + d_new
            i_old = i_new
        else:
            # alte Einträge löschen
            pre_sort = np.delete(pre_sort, i_old)
            break
    if len(pre_sort)>0:
        # an der Stelle wurden step-1 Elemente hinzugefügt     
        # das nächste Element ist Startpunkt der nächsten step Elemente
        pre_sort = list(pre_sort)
        index = pre_sort.index(id_sort[i_old])
        pre_sort = np.array([pre_sort[index]] + pre_sort[:index] + pre_sort[index+1:])
    
    print(float(int(1000*len(id_list)/len(spc)))/10)

new_im = np.zeros(len(ds_list))
new_im[id_list] = ds_list
new_im = np.reshape(new_im, (nZ, *imshape))

catshow(np.reshape(new_im, (nZ,*imshape)), ['lens, sorting: ']+spc_keys, False, cmap = 'hsv')    

bins = 1000

b0 = np.round(bins * np.prod(imshape_red) / np.prod(imshape)*nZ).astype(int)

#%% flattening and cropping the histogram of the lens image (ie make it denser w/o changing the order)

metric_name = 'cosine'

sorting = np.argsort(new_im.flatten())
dl = np.sort(new_im.flatten())
dl = dl[1:] - dl[:-1]

fSize = 5
thresh = 0.1
q1 = 1
q2 = 2
eps = 1E-10

weight = np.full((fSize), 1.0/fSize)
dl_med = median_filter(dl, fSize)
dl_mea = convolve(dl, weight)
t = np.abs((dl_mea-dl_med)/(dl_med+dl_mea+eps))
dl_mea[t>thresh] = dl_med[t>thresh]
mea = np.mean(dl_mea)
std = np.std(dl_mea)
sel = dl_mea >= mea+q1*std
off = dl_mea[sel]
mi, ma = min(off), max(off)
off = ((off-mi)/(ma-mi)) * ((q2-q1)*std) + mea+q1*std
dl_mea[sel]=off

old = 0
lens = [0]
for d in dl_mea:
    old = old+d
    lens.append(old)


new_im_flat = np.zeros(len(lens))
new_im_flat[sorting] = lens
new_im_flat = np.reshape(new_im_flat, (nZ, *imshape))

#catshow(np.reshape(new_im, (nZ,*imshape)), ['lens, sorting: ']+spc_keys, False, cmap = 'hsv')  

#%% do actual tda
#%%

""" tda parameters """

resolution_windows = 1                  # how many windows should be used at the same time ( be careful, could generate cicle like networks )
resolution = 70*resolution_windows*nZ   # how many resolution windos should be used ( to many - noise, to less - no fine structures )
gain = 6                                # overlap of the windows

method = 'ward' # also possible: single, complete,...

t = 2.5 # threshold in cluster analysis

limit = 5 # minimum number of spectra per cluster

""""""""""""

my_lens = new_im_flat.flatten()

resolution = np.ceil(resolution/resolution_windows).astype(int)*resolution_windows

eps = 1E-3 # stretch the borders to get every point
overlap = (gain-1)/gain

data = np.reshape(spc_data,(np.prod(spc_data.shape[:-1]), spc_data.shape[-1]))
data_id = np.arange(len(data))

minmax = np.amax(my_lens, axis = 0) - np.amin(my_lens, axis = 0)
eps = (minmax/resolution)*eps
mins = np.amin(my_lens, axis = 0)

c_width = (minmax+2*eps)/(1+(resolution-1)*(1-overlap))
c_dist  = c_width*(1-overlap)

subset_borders = [[mins - eps + r*c_dist, mins - eps + r*c_dist + c_width] for r in range(resolution)]
border_select = [[i*resolution//resolution_windows+j for i in range(resolution_windows)] for j in range(resolution//resolution_windows)]

topo_graph = nx.Graph()
i_node = 0
ib = 0

for bs in border_select:
    ib = ib+1
    
    my_border = [ (my_lens>=subset_borders[i_sb][0])*(my_lens<=subset_borders[i_sb][1]) for i_sb in bs]
    my_border = np.sum(np.array(my_border), axis = 0)>0
    
    a = my_border
    
    if np.sum(a) == 1:
        spc_id = data_id[a]
        topo_graph.add_node(i_node, ids = spc_id, height = len(spc_id))
        i_node = i_node+1
    elif np.sum(a) > 1:
        spc_id = data_id[a]
        
        ## perform single linkage clustering
        # select correct values from distance matrix
       
        
        """ if n_spc is small, this is a much faster way!
        ids = data_red_id[a]
        idn = len(ids)
        ids = np.meshgrid(ids, ids)
        ida, idb = ids[0].flatten().astype(int), ids[1].flatten().astype(int)
        idd = squareform(np.reshape(dist[ida,idb], (idn,idn)))
        """
        idd = pdist(data[a], metric_name)
        idd[np.isnan(idd)] = 0
        Z = linkage(idd, method)
        
        #cluster analyse
        lens_dist = np.median(np.sort(my_lens[a])[1:]-np.sort(my_lens[a])[:-1])
        c = fcluster(Z, t = t*lens_dist, criterion='distance')
        n0=len(np.unique(c))
        
        # nur cluster nehmen, welche groß geng sind
        c_uni = np.unique(c)
        c_o = [np.sum(c==ic)<limit for ic in c_uni]
        c_n = [np.sum(c==ic)>=limit for ic in c_uni]
        c_out = c_uni[c_o]
        c_uni = c_uni[c_n]
        if len(c_uni)>0:
            for co in c_out:
                c[c==co] = 0
            
            # cluster mediane bilden und spektren spektral median filtern
            
            spc_filt = medfilt(data[a].__copy__(), (1,3))
            c_spc = np.array([np.median(spc_filt[c==cu], axis = 0) for cu in c_uni])
            
            co_id = np.arange(len(c))[c==0]
            co_spc = spc_filt[c==0]
            
            # übrige spektren einem der großen cluster zuordnen
            
            co_c_dist = cdist(co_spc, c_spc, metric = metric_name)
            mins = np.argmin(co_c_dist, axis = 1)
            c[co_id] = c_uni[mins]
            
            print(n0, " -> ", len(c_uni), ": ",[np.sum(c==cu) for cu in c_uni])
            
            """
            c_spc = [np.median(spc_filt[c==cu], axis = 0) for cu in c_uni]
            c_std = [np.std(spc_filt[c==cu], axis = 0) for cu in c_uni]
            plt.figure()
            for cs, ct in zip(c_spc, c_std):
                plt.plot(cs)
                #plt.fill_between(np.arange(len(cs)), cs+ct, cs-ct, alpha = 0.2)
            plt.show()
            """
            
        else:
            print(n0, ", no unique clusters")
        for i_c in np.unique(c):
            id_list = spc_id[c==i_c]
            topo_graph.add_node(i_node, ids = id_list, height = len(id_list))
            i_node = i_node+1
    print( ib , " / ", resolution//resolution_windows)

      
n_nodes = len(topo_graph)   

rm = []
for node_a in range(n_nodes):
    delete = 1
    l1 = topo_graph.nodes[node_a]['ids']
    for node_b in range(node_a+1, n_nodes):
        l2 = topo_graph.nodes[node_b]['ids']
        d = set(l1).intersection(l2)
        if (len(d)>0):
            topo_graph.add_edge(node_a, node_b, weight=len(d)) 
            #topo_graph.add_edge(node_a, node_b) 
            delete = 0
    if delete == 1:
        rm = rm+[node_a]
        
n_rm = len(rm)
rm = np.flip(rm)
for r in rm:
    topo_graph.remove_node(r)
print('n_nodes: ', len(topo_graph), " = ", n_nodes , ' - ', n_rm, ' (', n_nodes-n_rm, ') ')

n_nodes = n_nodes-n_rm

pos_topo = nx.spring_layout(topo_graph, weight = None)
fig = plt.figure()
title = 'TOPO - unweighted'
fig.canvas.set_window_title(title)
nx.draw_networkx_nodes(topo_graph, pos_topo, node_size=1)
nx.draw_networkx_edges(topo_graph, pos_topo, alpha=0.4)

plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
plt.title(title)
plt.show()

partition = community_louvain.best_partition(topo_graph, weight = 'height')
plt.figure()
nx.draw_networkx_nodes(topo_graph, pos_topo, partition.keys(), node_size=10,
                       cmap = cm.get_cmap('hsv', max(partition.values()) + 1), node_color=list(partition.values()))

nx.draw_networkx_edges(topo_graph, pos_topo, alpha=0.4)
plt.show()

#%% generate cluster maps

cl = set(partition.values())
part = np.asarray(list(partition.values()))
ids = nx.get_node_attributes(topo_graph, 'ids')
id_topo = np.asarray(list(ids.keys()))
clmap = []
for ic in cl:
    nodes_c = id_topo[part==ic]
    my_nodes = []
    for node in nodes_c: my_nodes.append(list(ids[node]))
    my_nodes = list(set(itertools.chain(*my_nodes)))
    if len(my_nodes)>1:
        
        as_map = np.zeros(nZ*np.prod(imshape))
        as_map[my_nodes] = 1
        
        as_map = as_map * np.sum(data, axis = -1)
        as_map = (as_map-np.amin(as_map))/(np.amax(as_map)-np.amin(as_map))
        as_map = exposure.equalize_hist(as_map)
        
        clmap.append(np.reshape(as_map, (nZ, *imshape)))
        
        catshow(np.reshape(as_map, (nZ, *imshape)), ['class ', str(ic)]+spc_keys, False)  
        """
        plt.figure()
        plt.imshow(np.reshape(as_map, (nZ, *imshape))[nZ//2])
        plt.show()
        """

#%% generate mean spectra per cluster (probably helpful: use normalization first or use median)
sList = []
for ic in cl:
    nodes_c = id_topo[part==ic]
    my_nodes = []
    for node in nodes_c: my_nodes.append(list(ids[node]))
    my_nodes = list(set(itertools.chain(*my_nodes)))
    if len(my_nodes)>1:
        
        as_map = np.zeros(np.prod((nZ, *imshape)))
        as_map[my_nodes] = 1
        sList.append(np.mean(data[as_map==1], axis = 0))
        plt.figure()
        plt.plot(spc_wl,np.mean(data[as_map==1], axis = 0))
        plt.show()
      
sList = np.array(sList)

#%% generate smoother cluster maps using nnf

H, W, n =  non_negative_factorization(np.abs(data), n_components = sList.shape[0], init='custom', update_H=False, H=np.abs(sList), max_iter = 10000)

res = H.__copy__().T
for ic, r in enumerate(res):
    catshow(np.reshape(r, (nZ, *imshape)), ['class ', str(ic)]+spc_keys, False)  



#%% save the results


# the lens
arr = np.reshape(res, (res.shape[0], nZ, *imshape))
nip.imsave(nip.image(arr[:, nZ//2]),r'M:\Downloads\im.tif')


#%% save the results
# lens #
plt.figure(' saving: lens ')
plt.imshow(new_im_flat[nZ//2])
plt.show()

np.save(path.join(path_project, r'temp_results\lens.npy'), new_im_flat)

# network #
nx.write_gpickle(topo_graph, path.join(path_project, r'temp_results\graph.gpickle'))
np.save(path.join(path_project, r'temp_results\pos_topo.npy'), pos_topo)
np.save(path.join(path_project, r'temp_results\partition.npy'), partition)

G = nx.read_gpickle(path.join(path_project, r'temp_results\graph.gpickle'))

tp = nx.spring_layout(topo_graph, weight = None)
plt.figure(' saving: topo ')
nx.draw_networkx_nodes(G, tp, node_size=1)
nx.draw_networkx_edges(G, tp, alpha=0.4)
plt.show()

# spectra
np.save(path.join(path_project, r'temp_results\sList.npy'), sList)

# cluster maps
np.save(path.join(path_project, r'temp_results\clmap.npy'), clmap)

# smooth maps
np.save(path.join(path_project, r'temp_results\res.npy'), res)

arr = np.reshape(res, (res.shape[0], nZ, *imshape))
nip.imsave(nip.image(arr[:, nZ//2]),path.join(path_project, r'temp_results\res.tif'))

