import os
import numpy as np
from scipy.signal import welch
import dask
import xarray as xr
import threading

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cmocean import cm

import datetime, dateutil

import xmitgcm as xm

from .utils import *


            
#------------------------------ zarr ---------------------------------

def store_zarr(ds, filename, encoding):
    # tmp, waiting for xarray release
    for v in ds.variables:
        if hasattr(ds[v],'encoding') and 'chunks' in ds[v].encoding:
            del ds[v].encoding['chunks']    
    #
    ds.to_zarr(filename, mode='w', encoding=encoding)

def zarr_standard(V, client, F=None, out_dir=None, compressor=None):
    """ rechunk variables
    
    Parameters
    ----------
    V: list of str
        variables to rechunk

    face: list of int
        faces to consider, default is all (0-12)
        
    out_dir: str
        output directory
        
    """

    if out_dir is None:
        out_dir = scratch+'mit/standard/'

    if F is None:
        F = range(13)
    elif type(F) is int:
        F = [F]
        
    for v in V:
        #
        data_dir = root_data_dir+v+'/'
        iters, time = get_iters_time(v, data_dir, delta_t=25.)
        #
        p = 'C'
        if v is 'SSU':
            p = 'W'
        elif v is 'SSV':
            p = 'S'
        #
        ds = get_compressed_data(v, data_dir, grid_dir, iters=iters, 
                                 time=time, client=client, point=p)
        #
        # should store grid data independantly in a single file
        ds = ds.drop(['XC','YC','Depth','rA'])
        #
        ds = ds.sel(face=F)
        ds = ds.chunk({'face': 1})
        #
        dv = ds[v].to_dataset()
        #
        file_out = out_dir+'/%s.zarr'%(v)
        try:
            dv.to_zarr(file_out, mode='w', \
                       encoding={key: {'compressor': compressor} for key in dv.variables})
        except:
            print('Failure')
        dsize = getsize(file_out)
        print(' %s converted to zarr,  data is %.1fGB ' %(v, dsize/1e9))

        
#------------------------------ rechunking ---------------------------------

def rechunk(V, F=None, out_dir=None, Nt = 24*10, Nc = 96, compressor=None):
    """ rechunk variables
    
    Parameters
    ----------
    V: list of str
        variables to rechunk

    F: list of int
        faces to consider, default is all (0-12)
        
    out_dir: str
        output directory
        
    Nt: int
        temporal chunk size, default is 24x10
    
    Nc; int, tuple
        i, j chunk size, default is (96, 96) (96x45=4320)
        
    """

    if out_dir is None:
        out_dir = scratch+'mit/rechunked/'

    if F is None:
        F=range(13)
    elif type(F) is int:
        F = [F]
        
    if type(Nc) is int:
        Ni, Nj = Nc
    elif type(Nc) is tuple:
        Ni, Nj = Nc[0], Nc[1]
        
    for v in V:

        file_in = scratch+'/mit/standard/%s.zarr'%(v)
        ds0 = xr.open_zarr(file_in)
                
        for face in F:

            ds = ds0.sel(face=face)
            #
            ds = ds.isel(time=slice(len(ds.time)//Nt *Nt))
            #
            ds = ds.chunk({'time': Nt, 'i': Ni, 'j': Nj})
            #
            # tmp, xarray zarr backend bug: 
            # https://github.com/pydata/xarray/issues/2278
            del ds['face'].encoding['chunks']
            del ds[v].encoding['chunks']

            file_out = out_dir+'%s_f%02d.zarr'%(v,face)
            try:
                if compressor is 'default':
                    ds.to_zarr(file_out, mode='w')
                else:
                    # specify compression:
                    ds.to_zarr(file_out, mode='w', \
                                encoding={key: {'compressor': compressor} for key in ds.variables})
            except:
                print('Failure')
            dsize = getsize(file_out)
            print(' %s face=%d  data is %.1fGB ' %(v, face, dsize/1e9))
        