
# coding: utf-8

# In[1]:


import os, sys
import numpy as np
import dask
#from dask_jobqueue import PBSCluster
import xarray as xr
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from mitequinox.utils import *


# In[2]:


dmethod = 1
#
if dmethod == 1:
    from dask.distributed import Client
    scheduler = os.getenv('DATAWORK')+'/dask/scheduler.json'
    client = Client(scheduler_file=scheduler)
elif dmethod == 2:
    from dask_jobqueue import PBSCluster
    # folder where data is spilled when RAM is filled up
    local_dir = os.getenv('TMPDIR')
    #
    cluster = PBSCluster(queue='mpi_1', local_directory=local_dir, interface='ib0', walltime='24:00:00',
                         threads=14, processes=2, memory='50GB', resource_spec='select=1:ncpus=28:mem=100g', 
                         death_timeout=100)
    w = cluster.start_workers(40)


# In[3]:


# you need to wait for workers to spin up
if dmethod == 2:
    cluster.scheduler


# In[4]:


# get dask handles and check dask server status
if dmethod == 2:
    from dask.distributed import Client
    client = Client(cluster)


# In[5]:


client


# --- 
# 
# # automatic rewriting of all variables

# In[ ]:


V = ['Eta', 'SST', 'SSS', 'SSU', 'SSV']
V = ['Eta']

transpose = False

#out_dir = datawork+'/mit_T/'
#out_dir = scratch+'/mit_nc_t/'

if transpose:
    Nt = 24*10 # time windows to consider
    out_dir = datawork+'/mit_nc_t/'
    fsize_bound = 15*1e9
else:
    Nt = 1
    out_dir = datawork+'/mit_nc/'
    fsize_bound = 60*1e6    

for v in V:
    #
    data_dir = root_data_dir+v+'/'
    iters, time = get_iters_time(v, data_dir, delta_t=25.)
    #
    it = np.arange(time.size/Nt-1).astype(int)*Nt
    #it = np.arange(10).astype(int)*Nt # tmp
    assert it[-1]+Nt<time.size
    #
    p = 'C'
    if v is 'SSU':
        p = 'W'
    elif v is 'SSV':
        p = 'S'
    #
    ds = get_compressed_data(v, data_dir, grid_dir, iters=iters, time=time, client=client, point=p)
    ds = ds.chunk({'face': 1})
    #
    for face in range(ds['face'].size):
        for i, t in enumerate(it):
            #
            file_out = out_dir+'/%s_f%02d_t%02d.nc'%(v,face,i)
            if not os.path.isfile(file_out) or os.path.getsize(file_out) < fsize_bound:            
                dv = ds[v].isel(time=slice(t,t+Nt), face=face) 
                # should store grid data independantly in a single file
                dv = dv.drop(['XC','YC','Depth','rA']).to_dataset()
                #
                if transpose:
                    dv = dv.chunk({'time': dv['time'].size, 'i': 432, 'j': 432})
                    dv = dv.transpose('i','j','time')
                    chunksizes = [432, 432, dv['time'].size]
                else:
                    dv = dv.chunk({'i': 432, 'j': 432})
                    chunksizes = [1, 432, 432]
                #print(dv)
                #
                while True:
                    try:
                        get_ipython().run_line_magic('time', "dv.to_netcdf(file_out, mode='w', unlimited_dims=['time'],                                            encoding={'Eta': {'chunksizes': chunksizes}})")
                    except:
                        print('Failure')
                    if os.path.isfile(file_out) and os.path.getsize(file_out)>fsize_bound:
                        #
                        print('face=%d / i=%d'%(face,i))
                        break
            else:
                print('face=%d / i=%d - allready processed'%(face,i))


# --- 
# 
# # standard data layout

# In[ ]:


V = ['Eta', 'SST', 'SSS', 'SSU', 'SSV']
V = ['Eta']

transpose = False # True untested

#out_dir = datawork+'/mit_T/'
#out_dir = scratch+'/mit_nc_t/'

if transpose:
    Nt = 24*10 # time windows to consider
    out_dir = datawork+'/mit_nc_t/'
    fsize_bound = 1e12
else:
    Nt = 1
    out_dir = datawork+'/mit_nc/'
    fsize_bound = 13*60*1e6

for v in V:
    #
    data_dir = root_data_dir+v+'/'
    iters, time = get_iters_time(v, data_dir, delta_t=25.)
    #
    it = np.arange(time.size/Nt-1).astype(int)*Nt
    #it = np.arange(10).astype(int)*Nt # tmp
    assert it[-1]+Nt<time.size
    #
    p = 'C'
    if v is 'SSU':
        p = 'W'
    elif v is 'SSV':
        p = 'S'
    #
    ds = get_compressed_data(v, data_dir, grid_dir, iters=iters, time=time, client=client, point=p)
    #ds = ds.chunk({'face': 1})
    #
    for i, t in enumerate(it):
        #
        file_out = out_dir+'/%s_t%04d.nc'%(v,i)
        if not os.path.isfile(file_out) or os.path.getsize(file_out) < fsize_bound:            
            dv = ds[v].isel(time=slice(t,t+Nt)) 
            # should store grid data independantly in a single file
            dv = dv.drop(['XC','YC','Depth','rA']).to_dataset()
            #
            if transpose:
                dv = dv.chunk({'time': dv['time'].size, 'i': 432, 'j': 432})
                dv = dv.transpose('face','i','j','time')
                #chunksizes = [1, 432, 432, dv['time'].size]
            else:
                #dv = dv.chunk({'i': 4320, 'j': 4320})
                #chunksizes = [1, 1, 432, 432]
                pass
            #print(dv)
            #
            while True:
                try:
                    get_ipython().run_line_magic('time', "dv.to_netcdf(file_out, mode='w')")
                    #%time dv.to_netcdf(file_out, mode='w', unlimited_dims=['time'])                    
                    #%time dv.to_netcdf(file_out, mode='w', unlimited_dims=['time'], \
                    #                   encoding={'Eta': {'chunksizes': chunksizes}})
                except:
                    print('Failure')
                #if os.path.isfile(file_out):
                if os.path.isfile(file_out) and os.path.getsize(file_out) > fsize_bound:
                    #
                    print('i=%d, iter=%d'%(i, iters[i].values))
                    break
        else:
            print('i=%d, iter=%d - allready processed'%(i, iters[i].values))


# ---
# 
# # preliminary attempts to transpose and store data in netcdf files

# In[ ]:


import netCDF4 as nc4


# In[ ]:


Nt = 24*10 # time windows to consider
v = 'Eta'
face = 1
out_dir = datawork+'/mit_tmp/'


data_dir = root_data_dir+v+'/'
iters, time = get_iters_time(v, data_dir, delta_t=25.)
p = 'C'
if v is 'SSU':
    p = 'W'
elif v is 'SSV':
    p = 'S'
ds = get_compressed_data(v, data_dir, grid_dir, iters=iters, time=time, client=client, point=p)
ds = ds.chunk({'face': 1})

dv = ds[v].isel(time=slice(0,Nt), face=face)

# xarray stores datasets preferentially
dv = dv.drop(['XC','YC','Depth','rA']).to_dataset()

dv0 = dv
print(dv)


# ---
# ### Default chunking

# In[ ]:


file_out = out_dir+'/%s_f%02d_0.nc'%(v,face)
get_ipython().run_line_magic('time', "dv.to_netcdf(file_out, mode='w', unlimited_dims=['time'])")


# In[ ]:


nc_eta = nc4.Dataset(file_out)['Eta']
print(nc_eta)
print(nc_eta.chunking())


# ---
# ### transposed dimensions

# In[ ]:


#
dv = dv0.transpose('i','j','time')
print(dv)
file_out = out_dir+'/%s_f%02d_1.nc'%(v,face)
get_ipython().run_line_magic('time', "dv.to_netcdf(file_out, mode='w', unlimited_dims=['time'])")


# In[ ]:


nc_eta = nc4.Dataset(file_out)['Eta']
print(nc_eta)
print(nc_eta.chunking())


# ---
# ### transpose and rechunk
# 
# the xarray rechunking does not affect the netcdf chunks

# In[ ]:


dv = dv0.chunk({'time': dv['time'].size})
dv = dv.transpose('i','j','time')
#dv = dv0.chunk({'time': dv['time'].size, 'i': 100, 'j': 100})
print(dv)


# In[ ]:


file_out = out_dir+'/%s_f%02d_2.nc'%(v,face)
get_ipython().run_line_magic('time', "dv.to_netcdf(file_out, mode='w', unlimited_dims=['time'])")
#%time dv.to_netcdf(file_out, mode='w') # leads to contiguous data


# In[ ]:


nc_eta = nc4.Dataset(file_out)['Eta']
print(nc_eta)
print(nc_eta.chunking())


# ---
# ### netcdf chunks passed with encoding option
# 
# takes very long time, not good

# In[ ]:


file_out = out_dir+'/%s_f%02d_3.nc'%(v,face)
dv = dv0
print(dv)
print(dv['time'].size)
#dv.to_netcdf(file_out, mode='w', encoding={'Eta': {'chunksizes': {'time': dv['time'].size, 'i': 432, 'j': 432}}})
get_ipython().run_line_magic('time', "dv.to_netcdf(file_out, mode='w', unlimited_dims=['time'],              encoding={'Eta': {'chunksizes': [432, 432, dv['time'].size]}})")


# In[ ]:


nc_eta = nc4.Dataset(file_out)['Eta']
print(nc_eta)
print(nc_eta.chunking())


# ---
# ### same but with xarray rechunking

# In[ ]:


#dv = dv0.chunk({'time': dv['time'].size}) # 3min 46s
dv = dv0.chunk({'time': dv['time'].size, 'i': 432, 'j': 432}) # 50.9 s
dv = dv.transpose('i','j','time')
print(dv)

file_out = out_dir+'/%s_f%02d_4.nc'%(v,face)
get_ipython().run_line_magic('time', "dv.to_netcdf(file_out, mode='w', unlimited_dims=['time'],              encoding={'Eta': {'chunksizes': [432, 432, dv['time'].size]}})")


# ---
# 

# In[ ]:


Nt = 24*10 # time windows to consider
v = 'Eta'
out_dir = datawork+'/mit_tmp/'


data_dir = root_data_dir+v+'/'
iters, time = get_iters_time(v, data_dir, delta_t=25.)
p = 'C'
if v is 'SSU':
    p = 'W'
elif v is 'SSV':
    p = 'S'
ds = get_compressed_data(v, data_dir, grid_dir, iters=iters, time=time, client=client, point=p)
#ds = ds.chunk({'face': 1})

it=22
dv = ds[v].isel(time=it)

# xarray stores datasets preferentially
dv = dv.drop(['XC','YC','Depth','rA']).to_dataset()

dv0 = dv
print(dv)


# In[ ]:


dv = dv0
file_out = out_dir+'/%s_t%04d.nc'%(v, it)
get_ipython().run_line_magic('time', "dv.to_netcdf(file_out, mode='w', unlimited_dims=['time'])")


# In[ ]:


client


# In[ ]:


client.restart()


# ---
# 
# ```
# aponte/mit_tmp% ncdump -sh Eta_f01_4.nc
# 
# netcdf Eta_f01_4 {
# dimensions:
# 	time = UNLIMITED ; // (24 currently)
# 	i = 4320 ;
# 	j = 4320 ;
# variables:
# 	int64 i(i) ;
# 		i:standard_name = "x_grid_index" ;
# 		i:axis = "X" ;
# 		i:long_name = "x-dimension of the t grid" ;
# 		i:swap_dim = "XC" ;
# 		i:_Storage = "contiguous" ;
# 		i:_Endianness = "little" ;
# 	int64 j(j) ;
# 		j:standard_name = "y_grid_index" ;
# 		j:axis = "Y" ;
# 		j:long_name = "y-dimension of the t grid" ;
# 		j:swap_dim = "YC" ;
# 		j:_Storage = "contiguous" ;
# 		j:_Endianness = "little" ;
# 	int64 face ;
# 		face:standard_name = "face_index" ;
# 		face:_Endianness = "little" ;
# 	double time(time) ;
# 		time:_FillValue = NaN ;
# 		time:_Storage = "chunked" ;
# 		time:_ChunkSizes = 512 ;
# 	float Eta(i, j, time) ;
# 		Eta:_FillValue = NaNf ;
# 		Eta:coordinates = "face" ;
# 		Eta:_Storage = "chunked" ;
# 		Eta:_ChunkSizes = 432, 432, 24 ;
# 
# // global attributes:
# 		:_NCProperties = "version=1|netcdflibversion=4.6.1|hdf5libversion=1.10.1" ;
# 		:_Format = "netCDF-4" ;
# }
# ```
