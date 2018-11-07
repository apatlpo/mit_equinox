import os
from glob import glob
import numpy as np
import xarray as xr
from pandas import DataFrame

import datetime, dateutil

#------------------------------ paths ---------------------------------------



# datarmor
try:
    plateform='datarmor'
    #tmp = os.getenv('TMPDIR')
    datawork = os.getenv('DATAWORK')+'/'
    home = os.getenv('HOME')+'/'
    scratch = os.getenv('SCRATCH')+'/'
    osi = '/home/datawork-lops-osi/aponte/'
    #
    root_data_dir = '/home/datawork-lops-osi/data/mit4320/'
    grid_dir = root_data_dir+'grid/'
except:
    pass

# hal
try:
    plateform='hal'
    tmp = os.getenv('TMPDIR')
    home = os.getenv('HOME')+'/'
    scratch = os.getenv('HOME')+'/scratch/'
    #
    root_data_dir = '/work/ALT/swot/swotpub/LLC4320/'
    work_data_dir = '/work/ALT/swot/aval/syn/'
    grid_dir = root_data_dir+'grid/'
    grid_dir_nc = root_data_dir+'grid_nc/'
except:
    pass


#------------------------------ mit specific ---------------------------------------

def load_grdnc(V=None):
    _hasface = ['CS', 'SN', 'Depth', 
                'dxC', 'dxG', 'dyC', 'dyG', 
                'hFacC', 'hFacS', 'hFacW', 
                'rA', 'rAs', 'rAw', 
                'XC', 'YC', 'XG', 'YG']
    gfiles = glob(grid_dir_nc+'*.nc')
    gv = [f.split('/')[-1].split('.')[0] for f in gfiles]
    if V is not None:
        gfiles = [f for f, v in zip(gfiles, gv) if v in V]
        gv = V
    objs = []
    for f, v in zip(gfiles, gv):
        if v in _hasface:
            objs.append(xr.open_dataset(f, chunks={'face':1}))
        else:
            objs.append(xr.open_dataset(f))
    return xr.merge(objs, compat='equals').set_coords(names=gv)

def load_datanc(v, suff='_t*', files=None, **kwargs):
    #
    if type(v) is list:
        ds = xr.merge([load_datanc(v1, suff=suff, **kwargs) for v1 in v]
                        , compat='equals')
    else:
        default_kwargs = {'concat_dim': 'time',
                          'compat': 'equals', 
                          'chunks': {'face':1, 'i': 480, 'j':480}}
        if v is 'SSU':
            default_kwargs['chunks'] = {'face':1, 'i_g': 480, 'j':480}
        elif v is 'SSV':
            default_kwargs['chunks'] = {'face':1, 'i': 480, 'j_g':480}            
        default_kwargs.update(kwargs)
        #
        files_in = root_data_dir+v+'/'+v+suff
        if files is not None:
            files_in = files        
        ds = xr.open_mfdataset(files_in, 
                               **default_kwargs)
        ds = ds.assign_coords(dtime=xr.DataArray(iters_to_date(ds.iters), 
                                                 coords=[ds.time], 
                                                 dims=['time']))        
    return ds

def load_iters_date_files(v='Eta'):
    ''' For a given variable returns a dataframe of available data
    '''
    files = sorted(glob(root_data_dir+v+'/'+v+'_t*'))
    iters = [int(f.split('_t')[-1].split('.')[0]) for f in files]
    date = iters_to_date(iters)
    d = [{'date': d, 'iter':i, 'file':f} for d,i,f in zip(date, iters, files)]
    return DataFrame(d).set_index('date')

def iters_to_date(iters, delta_t=25.):
    t0 = datetime.datetime(2011,9,13)    
    ltime = delta_t * (np.array(iters)-10368)
    dtime = [t0+dateutil.relativedelta.relativedelta(seconds=t) for t in ltime]    
    return dtime

#------------------------------ misc ---------------------------------------
                        
def getsize(dir_path):
    ''' Returns the size of a directory in bytes
    '''
    process = os.popen('du -s '+dir_path)
    size = int(process.read().split()[0]) # du returns kb
    process.close()
    return size*1e3            