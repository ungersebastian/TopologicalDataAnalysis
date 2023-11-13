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
    lens_function = 'PCA', lens_axis = 0, lens_norm = 2,
    resolution = 3, gain = 4
    )
tda.fit(spc_train)

plt.figure()
plt.imshow(np.reshape(tda.lens.values, shape_im))
#%%


#%% generate absolute paths of ressource data

path_dir = os.path.dirname(os.path.abspath(__file__))

path_import = r'tda\resources\Ret240033'
headerfile = 'Ret240033.txt'

path_final = join(path_dir, path_import)


#%%

path_dir = os.path.dirname(os.path.abspath(__file__))

path_import33 = r'tda\resources\Ret240033'
headerfile33 = 'Ret240033.txt'

path_import20 = r'tda\resources\Ret240020'
headerfile20 = 'Ret240020.txt'

path_import12 = r'tda\resources\Ret240012'
headerfile12 = 'Ret240012.txt'

path_final33 = join(path_dir, path_import33)
path_final20 = join(path_dir, path_import20)
path_final12 = join(path_dir, path_import12)


#%% loads data and plots associated VistaScan parameter images

my_data12 = tda.pifm_image(path_final12, headerfile12) 
my_data20 = tda.pifm_image(path_final20, headerfile20) 
my_data33 = tda.pifm_image(path_final33, headerfile33) 

my_data12.plot_all()
my_data20.plot_all()
my_data33.plot_all()

#%% loads data and plots associated VistaScan parameter images

my_data = tda.pifm_image(path_final, headerfile) 

my_data.plot_all()

#%% Calibration using CaF

path_caf = join(path_dir, path_import20,'Ret24_CaF_2001_Tuner1349-1643.txt')

caf_file = pd.read_csv(path_caf, delimiter = '\t')
caf_spc = np.array(caf_file)[:,1]

my_spc12 = my_data12.return_spc()
my_spc20 = my_data20.return_spc()
my_spc33 = my_data33.return_spc()

my_wl  = my_data12['wavelength']

#%% removing zero intensity spectra, calibration and normalization for each set indipendently

pos12 =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data12['files']]
hyPIRFwd12 = np.array(my_data12['files'])[pos12][0]
data12 = np.reshape(hyPIRFwd12['data'], (hyPIRFwd12['data'].shape[0]*hyPIRFwd12['data'].shape[1], hyPIRFwd12['data'].shape[2]))
my_sum12 = np.sum(data12, axis = 1)

pos20 =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data20['files']]
hyPIRFwd20 = np.array(my_data20['files'])[pos20][0]
data20 = np.reshape(hyPIRFwd20['data'], (hyPIRFwd20['data'].shape[0]*hyPIRFwd20['data'].shape[1], hyPIRFwd20['data'].shape[2]))
my_sum20 = np.sum(data20, axis = 1)

pos33 =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data33['files']]
hyPIRFwd33 = np.array(my_data33['files'])[pos33][0]
data33 = np.reshape(hyPIRFwd33['data'], (hyPIRFwd33['data'].shape[0]*hyPIRFwd33['data'].shape[1], hyPIRFwd33['data'].shape[2]))
my_sum33 = np.sum(data33, axis = 1)

# also cat them together
AllData = np.array([[data12], [data20], [data33]])
AllDatas = np.squeeze(AllData, 1).T

zeros12 = np.zeros(len(my_sum12))
data12 = data12[my_sum12 != 0]
my_sum12 = my_sum12[my_sum12 != 0]

zeros20 = np.zeros(len(my_sum20))
data20 = data20[my_sum20 != 0]
my_sum20 = my_sum20[my_sum20 != 0]

zeros33 = np.zeros(len(my_sum33))
data33 = data33[my_sum33 != 0]
my_sum33 = my_sum33[my_sum33 != 0]

spc_norm12 = np.array([(spc/caf_spc)/s for spc, s in zip(data12, my_sum12)])
spc_norm20 = np.array([(spc/caf_spc)/s for spc, s in zip(data20, my_sum20)])
spc_norm33 = np.array([(spc/caf_spc)/s for spc, s in zip(data33, my_sum33)])

#%% removing zero intensity spectra, calibration and normalization for all sets together

my_sum = np.sum(AllDatas, axis = 0)
zeros = np.zeros(len(my_sum))
my_sum = my_sum[my_sum != 0]

spc_norm  = np.array([(spc/caf_spc)/s for spc, s in zip(AllDatas.T, my_sum)])
spc_norm2 = spc_norm.T.reshape(spc_norm.T.shape[0], -1)

my_sum = np.sum(spc_norm2, axis = 0)
coord = np.arange(len(my_sum))
zeros = np.zeros(len(my_sum))
my_sum = my_sum[my_sum != 0]

#%%
# Plot the mean spectra
mean_spc12 = np.mean(spc_norm12, axis = 0)
std_spc12 = np.std(spc_norm12, axis = 0)

mean_spc20 = np.mean(spc_norm20, axis = 0)
std_spc20 = np.std(spc_norm20, axis = 0)

mean_spc33 = np.mean(spc_norm33, axis = 0)
std_spc33 = np.std(spc_norm33, axis = 0)

mean_spc = np.mean(spc_norm, axis = 1).T

mean_spc2 = mean_spc[:, 0]
mean_spc2 = mean_spc2.T


my_fig = plt.figure()
ax = plt.subplot(111)
plt.gca().invert_xaxis() #inverts values of x-axis
ax.plot(my_data12['wavelength'], mean_spc)
ax.set_xlabel('wavenumber ['+my_data12['PhysUnitWavelengths']+']')
ax.set_ylabel('intensity (normalized)')
ax.set_yticklabels([])
plt.title('mean spectrum')
my_fig.tight_layout()


#%%

from sklearn.decomposition import PCA
ncomp = 2
model = PCA(n_components=ncomp)

transformed_data = model.fit(spc_norm2.T - mean_spc2).transform(spc_norm2.T - mean_spc2).T
Twelve = transformed_data[:, 0:1024]
Twenty = transformed_data[:, 1024:2048]
ThirtyThree = transformed_data[:, 2048:3072]

loadings = model.components_



my_fig = plt.figure()
ax = plt.subplot(111)
plt.gca().invert_xaxis() #inverts values of x-axis
for icomp in range(ncomp):
    ax.plot(my_data12['wavelength'], loadings[icomp], label='PC'+str(icomp+1) )
    ax.legend()
ax.set_xlabel('wavenumber ['+my_data20['PhysUnitWavelengths']+']')
ax.set_ylabel('intensity (normalized)')
ax.set_yticklabels([])
plt.title('PCA-Loadings')



my_fig = plt.figure()
ax = plt.subplot(111)
ax.plot(transformed_data[0], transformed_data[1], '.')
ax.set_xlim(np.quantile(transformed_data[0], 0.05),np.quantile(transformed_data[0], 0.95))
ax.set_ylim(np.quantile(transformed_data[1], 0.05),np.quantile(transformed_data[1], 0.95))
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.title('scatterplot')
my_fig.tight_layout()

maps = [zeros.copy() for icomp in range(ncomp)]
maps12 = [zeros12.copy() for icomp in range(ncomp)]
maps20 = [zeros20.copy() for icomp in range(ncomp)]
maps33 = [zeros33.copy() for icomp in range(ncomp)]

for icomp in range(ncomp):
    
    maps[icomp][coord] = transformed_data[icomp]
    maps[icomp] = np.reshape(maps[icomp],(32,96))
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps[icomp], cmap = 'coolwarm' , vmin = -2000, vmax = 10000))
    ax.set_xlabel('x scan ['+my_data20['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data20['YPhysUnit']+']')
    ax.legend()
    plt.title('factors PC'+str(icomp+1))
    my_fig.tight_layout()

    maps12 = maps[icomp][:, 0:32]  
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps12, cmap = 'coolwarm' , vmin = -2000, vmax = 10000))
    ax.set_xlabel('x scan ['+my_data20['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data20['YPhysUnit']+']')
    #ax.legend()
    plt.title('240012:factors PC'+str(icomp+1))
    my_fig.tight_layout()
    g =  maps[icomp][:, 0:32].reshape(maps[icomp][:, 0:32].shape[0], -1)
    np.savetxt(r'export\Twelve-PC'+str(icomp+1)+'.txt', g, delimiter = '\t')

    maps20 = maps[icomp][:, 32:64]
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps20, cmap = 'coolwarm' , vmin = -2000, vmax = 10000))
    ax.set_xlabel('x scan ['+my_data20['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data20['YPhysUnit']+']')
    #ax.legend()
    plt.title('240020:factors PC'+str(icomp+1))
    my_fig.tight_layout()
    g2 =  maps[icomp][:, 32:64].reshape(maps[icomp][:, 32:64].shape[0], -1)
    np.savetxt(r'export\Twenty-PC'+str(icomp+1)+'.txt', g2, delimiter = '\t')

    maps33 = maps[icomp][:, 64:96 ]     
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps33, cmap = 'coolwarm' , vmin = -2000, vmax = 10000))
    ax.set_xlabel('x scan ['+my_data20['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data20['YPhysUnit']+']')
    #ax.legend()
    plt.title('240033:factors PC'+str(icomp+1))
    my_fig.tight_layout()
    g3 =  maps[icomp][:, 64:96].reshape(maps[icomp][:, 64:96].shape[0], -1)
    np.savetxt(r'export\Thirty-PC'+str(icomp+1)+'.txt', g3, delimiter = '\t')
 
