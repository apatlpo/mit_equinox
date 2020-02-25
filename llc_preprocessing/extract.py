
# coding: utf-8

# In[1]:


import os
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
# # try to transpose data

# In[ ]:


Nt = 24*10 # time windows to consider
V = ['Eta', 'SST', 'SSS', 'SSU', 'SSV']
V = ['Eta']
out_dir = '/home1/datawork/aponte/mit_T/'

#
it = np.arange(time.size/Nt-1).astype(int)*Nt
assert it[-1]+Nt<time.size

for v in V:
    #
    data_dir = data_rdir+v+'/'
    iters, time = get_iters_time(v, data_dir, delta_t=25.)
    p = 'C'
    if v is 'SSU':
        p = 'W'
    elif v is 'SSV':
        p = 'S'
    ds = get_compressed_data(v, data_dir, grid_dir, iters=iters, time=time, client=client, point=p)
    ds = ds.chunk({'face': 1})
    #
    for face in ds['face']:
        for i, t in enumerate(it):
            dv = ds[v].isel(time=slice(t,t+Nt), face=face)
            dv = dv.drop(['XC','YC','Depth','rA'])
            dv = dv.transpose('i','j','time')
            #
            file_out = out_dir+'/%s_f%02d_t%02d.nc'%(v,face,i)
            get_ipython().run_line_magic('time', "eta.to_netcdf(file_out, mode='w', unlimited_dims=['time'])")
            #
            print('face=%d / i=%d'%(face,i))

