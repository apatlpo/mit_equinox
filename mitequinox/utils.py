import os
from glob import glob
import numpy as np
import xarray as xr
import pandas as pd
from pandas import DataFrame, Series

import dateutil
from datetime import timedelta, datetime


#------------------------------ parameters -------------------------------------

g = 9.81
omega_earth = 2.*np.pi/86164.0905
deg2rad = np.pi/180.
deg2m = 111319

def coriolis(lat, signed=False):
    if signed:
        return 2.*omega_earth*np.sin(lat*deg2rad)
    else:
        return 2.*omega_earth*np.sin(np.abs(lat)*deg2rad)

def dfdy(lat, units='1/s/m'):
    df = 2.*omega_earth*np.cos(lat*deg2rad)*deg2rad/deg2m
    if units=='cpd/100km':
        df = df *86400/2./np.pi *100*1e3
    return df

def fix_lon_bounds(lon):
    ''' reset longitudes bounds to (-180,180)'''
    if isinstance(lon, xr.DataArray): # xarray
        return lon.where(lon<180., other=lon-360)
    elif isinstance(lon, Series):    # pandas series
        out = lon.copy()
        out[out>180.] = out[out>180.] - 360.
        return out

#------------------------------ paths ---------------------------------------

if os.path.isdir('/home/datawork-lops-osi/'):
    # datarmor
    plateform='datarmor'
    datawork = os.getenv('DATAWORK')+'/'
    home = os.getenv('HOME')+'/'
    scratch = os.getenv('SCRATCH')+'/'
    osi = '/home/datawork-lops-osi/'
    #
    root_data_dir = '/home/datawork-lops-osi/equinox/mit4320/'
    #
    bin_data_dir = root_data_dir+'bin/'
    bin_grid_dir = bin_data_dir+'grid/'
    #
    zarr_data_dir = root_data_dir+'zarr/'
    zarr_grid = zarr_data_dir+'grid.zarr'
    mask_path = zarr_data_dir+'mask.zarr'
elif os.path.isdir('/work/ALT/swot/'):
    # hal
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

#------------------------------ mit specific ---------------------------------------

def load_grd(V=None, ftype='zarr'):
    if ftype == 'zarr':
        return xr.open_zarr(root_data_dir+'zarr/grid.zarr')
    elif ftype == 'nc':
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

def load_data(V, ftype='zarr', merge=True, **kwargs):
    if isinstance(V, list):
        out = [load_data(v, ftype=ftype, **kwargs) for v in V]
        if merge:
            return xr.merge(out)
        else:
            return out
        return 
    else:
        if ftype == 'zarr':
            return load_data_zarr(V, **kwargs)
        elif ftype == 'nc':
            return load_data_nc(V, **kwargs)

def load_data_zarr(v):
    return xr.open_zarr(root_data_dir+'zarr/'+v+'.zarr')

def load_data_nc(v, suff='_t*', files=None, **kwargs):
    default_kwargs = {'concat_dim': 'time',
                      'compat': 'equals', 
                      'chunks': {'face':1, 'i': 480, 'j':480},
                      'parallel': True}
    if v == 'SSU':
        default_kwargs['chunks'] = {'face':1, 'i_g': 480, 'j':480}
    elif v == 'SSV':
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
    t0 = datetime(2011,9,13)    
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

def load_enatl60_UV(chunks={'x': 480, 'y': 480}):
    ds_U = xr.open_zarr(enatl60_data_dir+'zarr/eNATL60-BLBT02-SSU-1h')
    ds_V = xr.open_zarr(enatl60_data_dir+'zarr/eNATL60-BLBT02-SSV-1h')
    ds_V = ds_V.drop('nav_lon')
    ds = xr.merge([ds_U, ds_V])
    ds = ds.rename({'time_counter':'time', 'nav_lon':'lon', 'nav_lat':'lat',
                    'sozocrtx': 'SSU', 'somecrty': 'SSV'})
    ds = ds.set_coords(['lon','lat'])
    ds = ds.chunk(chunks)
    return ds

def load_enatl60_grid(chunks={'x': 440, 'y': 440}):
    grd = (xr.open_dataset(enatl60_data_dir+'mesh_mask_eNATL60_3.6.nc', 
                           chunks=chunks)
           .squeeze()
           .rename({'nav_lon':'lon','nav_lat':'lat','nav_lev':'z'})
           .set_coords(['lon','lat','z']))
    return grd

    
#------------------------------ time relative ---------------------------------------

        
def np64toDate(dt64, tzinfo=None):
        """
        Converts a Numpy datetime64 to a Python datetime.
        :param dt64: A Numpy datetime64 variable, type dt64: numpy.datetime64
        :param tzinfo: The timezone the date / time value is in, type tzinfo: pytz.timezone
        :return: A Python datetime variablertype: datetime
        """
        ts = pd.to_datetime(dt64)
        if tzinfo is not None:
            return datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, tzinfo=tzinfo)
        return datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)  
    
def dateRange(date1, date2, dt=timedelta(days=1.)):
    for n in np.arange(date1,date2, dt):
        yield np64toDate(n)
    