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


#------------------------------ momentum balance -----------------------------------

#
def get_mbal(term, ds, grid):

    # add tests?    
    if term is 'u_coriolis_linear':
        dxgSSV_j = grid.interp(ds.dxG * ds['SSV'],'Y') 
        dxgSSV_ji = grid.interp(dxgSSV_j,'X')    # SSV at (i_g,j)
        bterm = ds.f_j * dxgSSV_ji /ds.dxC    #f*SSV
        bterm = bterm.chunk({'i_g':None,'j':None})
    elif term is 'u_gradp':    
        bterm = grid.diff(g*ds['Eta'],'X')/ds.dxC    # d(Eta*g)/dx
        bterm = bterm.chunk({'i_g':None,'j':None})
    elif term is 'v_coriolis_linear':
        dygSSU_i = grid.interp(ds.dyG * ds['SSU'],'X') 
        dxgSSU_ij = grid.interp(dygSSU_i,'Y')    # SSU at (i,j_g)
        bterm = -ds.f_i * dxgSSU_ij /ds.dyC    #-f*SSU
        bterm = bterm.chunk({'i':None,'j_g':None})
    elif term is 'v_gradp':
        bterm = grid.diff(g*ds['Eta'],'Y')/ds.dyC     # d(Eta*g)/dy
        bterm = bterm.chunk({'i':None,'j_g':None})
    return bterm.rename(term)