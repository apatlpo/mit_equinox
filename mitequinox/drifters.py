
from glob import glob
import pickle

import numpy as np
import pandas as pd
import geopandas
import dask.bag as db
import xarray as xr

from functools import partial

#--------------------------------------  pair I/O ------------------------------------------------

data_dir = '/work/ALT/swot/aval/syn/drifters/'

def load_from_ID(ID):
    try:
        return pickle.load(open(data_dir+'single/argos_%09d.p'%ID, 'rb'))
    except:
        try:
            return pickle.load(open(data_dir+'single/gps_%09d.p'%ID, 'rb'))
        except:
            print('ID=%d has not data file in %ssingle/ directory'%(ID,data_dir))

def load_pair(p, data_dir):
    d0, id0 = load_from_ID(p[0])
    d1, id1 = load_from_ID(p[1])
    # get rid of gaps and interpolate if necessary
    d0 = d0[~pd.isnull(d0.index)]
    d1 = d1[~pd.isnull(d1.index)]
    #
    d0, d1 = d0.align(d1, join='inner')
    # fill gaps, should keep track of this
    d0, d1 = compute_vector(d0, d1)
    # should go through vectors for interpolation
    #d0 = d0.resample('H').interpolate('linear')
    #d1 = d1.resample('H').interpolate('linear')
    d0 = d0.resample('H').asfreq()
    d1 = d1.resample('H').asfreq()
    d0, d1 = compute_lonlat(d0, d1)
    # converts to geopandas
    gd0 = to_gdataframe(d0)
    gd1 = to_gdataframe(d1)
    return gd0, gd1, p

def _to_dict(x):
    out = x[0]
    out['ID'] = x[1]
    return out.to_dict()

def load_single_df(npartitions=100, gps=None):
    ''' could also load directly from netcdf files
    '''
    if gps is None:
        files = sorted(glob(data_dir+'single/*.p'))
    elif gps==0:
        files = sorted(glob(data_dir+'single/argos_*.p'))
    elif gps==1:
        files = sorted(glob(data_dir+'single/gps_*.p'))
    b = ( db.from_sequence(files[:], npartitions=npartitions) \
         .map(lambda f: pickle.load(open(f, 'rb'))) )
    return b.map(_to_dict).to_dataframe()
    
#--------------------------------------  binning ------------------------------------------------

# leverages apply_ufunc with numpy binning code
# https://github.com/pydata/xarray/issues/2817
# an alternative would be to create xarray dataarrays and use groupby_bins
# https://github.com/pydata/xarray/issues/1765

def _bin1d(bins, v, vbin, weights=True):

    if weights:
        w = v
    else:
        w = np.ones_like(v)
    h, edges = np.histogram(vbin, bins=bins, weights=w, density=False)
    return h[None,:]

def bin1d(v, vbin, bins, weights, bin_dim='bin_dim', name='binned_array'):
    # wrapper around apply_ufunc
    dims = ['TIME'] # core dim
    bins_c = (bins[:-1]+bins[1:])*.5
    out = xr.apply_ufunc(partial(_bin1d, bins), v, vbin, kwargs={'weights': weights},
                    output_core_dims=[[bin_dim]], output_dtypes=[np.float64], 
                    dask='parallelized',
                    input_core_dims=[dims, dims],
                    output_sizes={bin_dim: len(bins_c)})
    out = out.assign_coords(**{bin_dim: bins_c}).rename(name)
    return out

def _bin2d(bins1, bins2, v, vbin1, vbin2, weights=True):
    # wrapper around apply_ufunc
    if weights:
        w = v
    else:
        w = np.ones_like(v)
    h, edges1, edges2 = np.histogram2d(vbin1.flatten(), vbin2.flatten(), 
                                       bins=[bins1, bins2], weights=w.flatten(), density=False)
    return h[None,...]

def bin2d(v, vbin1, bins1, vbin2, bins2, weights, 
          bin_dim1='bin_dim1', bin_dim2='bin_dim2', name='binned_array'):
    # wrapper around apply_ufunc
    dims = ['TIME'] # core dim
    bins1_c = (bins1[:-1]+bins1[1:])*.5
    bins2_c = (bins2[:-1]+bins2[1:])*.5
    out = xr.apply_ufunc(partial(_bin2d, bins1, bins2), v, vbin1, vbin2, 
                       kwargs={'weights': weights},
                       output_core_dims=[[bin_dim1,bin_dim2]], output_dtypes=[np.float64], 
                       dask='parallelized',
                       input_core_dims=[dims, dims, dims],
                       output_sizes={bin_dim1: len(bins1_c), bin_dim2: len(bins2_c)})
    out = out.assign_coords(**{bin_dim1: bins1_c, bin_dim2: bins2_c}).rename(name)
    return out
    
#--------------------------------------  utils ------------------------------------------------


RADIUS_EARTH = 6378.0
deg2rad = np.pi / 180.
def haversine(lon1, lat1, lon2, lat2):
    """Computes the Haversine distance in kilometres between two points
    :param x: first point or points as array, each as array of latitude, longitude in degrees
    :param y: second point or points as array, each as array of latitude, longitude in degrees
    :return: distance between the two points in kilometres
    """
    llat1 = lat1 * deg2rad
    llat2 = lat2 * deg2rad
    llon1 = lon1 * deg2rad
    llon2 = lon2 * deg2rad
    arclen = 2 * np.arcsin(np.sqrt((np.sin((llat2 - llat1) / 2)) ** 2 +
                                   np.cos(llat1) * np.cos(llat2) * (np.sin((llon2 - llon1) / 2)) ** 2))
    return arclen * RADIUS_EARTH

# geodesy with vectors
# https://www.movable-type.co.uk/scripts/latlong-vectors.html
def compute_vector(*args, lon_key='LON', lat_key='LAT'):
    if len(args)==1:
        df = args[0]
        df['v0'] = np.cos(deg2rad*df[lat_key])*np.cos(deg2rad*df[lon_key])
        df['v1'] = np.cos(deg2rad*df[lat_key])*np.sin(deg2rad*df[lon_key])
        df['v2'] = np.sin(deg2rad*df[lat_key])
        return df
    else:
        return [compute_vector(df) for df in args]

def drop_vector(*args, v0='v0', v1='v1', v2='v2'):
    # should test for type in order to accomodate for xarray types
    if len(args)==1:
        return args[0].drop(columns=[v0,v1,v2])
    else:
        return [df.drop(columns=[v0,v1,v2]) for df in args]
    
def compute_lonlat(*args, dropv=True, 
                   v0='v0', v1='v1', v2='v2',
                   lon_key='LON', lat_key='LAT'):
    if len(args)==1:
        df = args[0]
        # renormalize vectors
        n = np.sqrt(df[v0]**2+df[v1]**2+df[v2]**2)
        df[v0], df[v1], df[v2] = df[v0]/n, df[v1]/n, df[v2]/n
        # estimate LON/LAT
        df[lon_key] = np.arctan2(df[v1],df[v0])/deg2rad
        df[lat_key] = np.arctan2(df[v2],np.sqrt(df[v0]**2+df[v1]**2))/deg2rad
        if dropv:
            df = drop_vector(df, v0=v0, v1=v1, v2=v2)
        return df
    else:
        return [compute_lonlat(df, dropv=dropv, v0=v0, v1=v1, v2=v2, 
                               lon_key=lon_key, lat_key=lat_key ) for df in args]
    
def to_gdataframe(*args):
    if len(args)==1:
        df = args[0]
        return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(
                    df.LON, df.LAT))
    else:
        return [geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(
                    df.LON, df.LAT)) for df in args]
