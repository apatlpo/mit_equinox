import os
import numpy as np
from scipy import signal
import dask
import xarray as xr
import threading

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cmocean import cm

import datetime, dateutil

#import xmitgcm as xm

from .utils import *

#------------------------------ temporal filters ---------------------------------------

def gen_filter(band, numtaps=24*10, dt=1., lat=None, domega=None, ndomega=None):
    ''' Wrapper around scipy.signal.firwing
    dt: float
        hours
    '''
    params = {}
    pass_zero = False
    if band=='semidiurnal':
        omega = 1/12.
    elif band=='diurnal':
        omega = 1/24.
    elif band=='inertial':
        try:
            omega = coriolis(lat)*3600/2./np.pi
        except:
            print('latitude needs to be provided to gen_filter')
    #
    if domega is not None:
        cutoff = [omega-domega, omega+domega]
    elif ndomega is not None:
        cutoff = [omega*(1-ndomega), omega*(1.+ndomega)]
    elif band is not 'subdiurnal':
        print('domega or ndomega needs to be provided to gen_filter')
    #
    if band=='subdiurnal':
        pass_zero = True
        cutoff = [1./30.]
    h = signal.firwin(numtaps, cutoff=cutoff, pass_zero=pass_zero, 
                      nyq=1./2/dt, scale=True)
    return h
        
def filter_response(h, dt=1.):
    ''' Returns the frequency response
    '''
    w, hh = signal.freqz(h, worN=8000)
    return hh, (w/np.pi)/2/dt*24

#------------------------------ spectrum ---------------------------------------
          
#[psi,lambda] = sleptap(size(uv2,1),2,1);
#% calculate all spectra, linearly detrend
#tic;[f,spp2,snn2] = mspec(1/24,detrend(uv2,1),psi);toc         
#[PSI,LAMBDA]=SLEPTAP(N,P,K) calculates the K lowest-order Slepian
#    tapers PSI of length N and time-bandwidth product P, together with
#    their eigenvalues LAMBDA. PSI is N x K and LAMBDA is K x 1.

# scipy.signal.windows.dpss(M, NW, Kmax=None, sym=True, norm=None, return_ratios=False)          
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.dpss.html#scipy.signal.windows.dpss
            
def _get_E(x, ufunc=True, **kwargs):
    ax = -1 if ufunc else 0
    #
    dkwargs = {'window': 'hann', 'return_onesided': False, 
               'detrend': 'linear', 'scaling': 'density'}
    dkwargs.update(kwargs)
    f, E = signal.welch(x, fs=24., axis=ax, **dkwargs)
    #
    if ufunc:
        return E
    else:
        return f, E

def get_E(v, f=None, **kwargs):
    v = v.chunk({'time': len(v.time)})
    if 'nperseg' in kwargs:
        Nb = kwargs['nperseg']
    else:
        Nb = 80*24
        kwargs['nperseg']= Nb
    if f is None:
        f, E = _get_E(v.values, ufunc=False, **kwargs)
        return f, E
    else:
        E = xr.apply_ufunc(_get_E, v,
                    dask='parallelized', output_dtypes=[np.float64],
                    input_core_dims=[['time']],
                    output_core_dims=[['freq_time']],
                    output_sizes={'freq_time': Nb}, kwargs=kwargs)
        return E.assign_coords(freq_time=f).sortby('freq_time')

#------------------------------ misc ---------------------------------------
                        
def getsize(dir_path):
    ''' Returns the size of a directory in bytes
    '''
    process = os.popen('du -s '+dir_path)
    size = int(process.read().split()[0]) # du returns kb
    process.close()
    return size*1e3            