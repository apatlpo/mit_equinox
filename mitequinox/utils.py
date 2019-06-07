import os
from glob import glob
import numpy as np
import xarray as xr
from pandas import DataFrame

import datetime, dateutil


#------------------------------ parameters -------------------------------------

g = 9.81
omega_earth = 2.*np.pi/86164.0905
deg2rad = np.pi/180.

def coriolis(lat, signed=False):
    if signed:
        return 2.*omega_earth*np.sin(lat*deg2rad)
    else:
        return 2.*omega_earth*np.sin(np.abs(lat)*deg2rad)
        

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
    #grid_dir = root_data_dir+'grid/'
    #grid_dir_nc = root_data_dir+'grid_nc/'
    enatl60_data_dir = '/work/ALT/odatis/eNATL60/'
except:
    pass


#------------------------------ mit specific ---------------------------------------

def load_grd(V=None, ftype='zarr'):
    if ftype is 'zarr':
        return xr.open_zarr(root_data_dir+'zarr/grid.zarr')
    elif ftype is 'nc':
        return load_grdnc(V)

def load_grdnc(V):
    _hasface = ['CS', 'SN', 'Depth', 
                'dxC', 'dxG', 'dyC', 'dyG', 
                'hFacC', 'hFacS', 'hFacW', 
                'rA', 'rAs', 'rAw', 
                'XC', 'YC', 'XG', 'YG']
    gfiles = glob(grid_dir+'netcdf/grid/*.nc')
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

def load_data(V, ftype='zarr', **kwargs):
    if type(V) is list:
        #return xr.merge([load_data(v, ftype=ftype, **kwargs) for v in V]
        #               , compat='equals')
        return [load_data(v, ftype=ftype, **kwargs) for v in V]
    else:
        if ftype is 'zarr':
            return load_data_zarr(V, **kwargs)
        elif ftype is 'nc':
            return load_data_nc(V, **kwargs)

def load_data_zarr(v, suffix='_std'):
    return xr.open_zarr(work_data_dir+'rechunked/'+v+suffix+'.zarr')

def load_data_nc(v, suff='_t*', files=None, **kwargs):
    default_kwargs = {'concat_dim': 'time',
                      'compat': 'equals', 
                      'chunks': {'face':1, 'i': 480, 'j':480},
                      'parallel': True}
    if v is 'SSU':
        default_kwargs['chunks'] = {'face':1, 'i_g': 480, 'j':480}
    elif v is 'SSV':
        default_kwargs['chunks'] = {'face':1, 'i': 480, 'j_g':480}            
    default_kwargs.update(kwargs)
    #
    files_in = root_data_dir+'netcdf/'+v+'/'+v+suff
    if files is not None:
        files_in = files        
    ds = xr.open_mfdataset(files_in, 
                           **default_kwargs)
    ds = ds.assign_coords(dtime=xr.DataArray(iters_to_date(ds.iters.values), 
                                             coords=[ds.time], 
                                             dims=['time']))        
    return ds

def load_iters_date_files(v='Eta'):
    ''' For a given variable returns a dataframe of available data
    '''
    files = sorted(glob(root_data_dir+'netcdf/'+v+'/'+v+'_t*'))
    iters = [int(f.split('_t')[-1].split('.')[0]) for f in files]
    date = iters_to_date(iters)
    d = [{'date': d, 'iter':i, 'file':f} for d,i,f in zip(date, iters, files)]
    return DataFrame(d).set_index('date')

def iters_to_date(iters, delta_t=25.):
    t0 = datetime.datetime(2011,9,13)    
    ltime = delta_t * (np.array(iters)-10368)
    dtime = [t0+dateutil.relativedelta.relativedelta(seconds=t) for t in ltime]    
    return dtime

def load_common_timeline(V, verbose=True):
    df = load_iters_date_files(V[0]).rename(columns={'file': 'file_'+V[0]})
    for v in V[1:]:
        ldf = load_iters_date_files(v)
        ldf = ldf.rename(columns={'file': 'file_'+v})
        df = df.join(ldf['file_'+v], how='inner')
    if verbose:
        print(df.index[0], ' to ', df.index[-1])
    return df

#------------------------------ misc ---------------------------------------
                        
def getsize(dir_path):
    ''' Returns the size of a directory in bytes
    '''
    process = os.popen('du -s '+dir_path)
    size = int(process.read().split()[0]) # du returns kb
    process.close()
    return size*1e3

def rotate(u,v,ds):
    # rotate from grid to zonal/meridional directions
    return u*ds.CS-v*ds.SN, u*ds.SN+v*ds.CS
    
    
    
#------------------------------ enatl60 specific ---------------------------------------




