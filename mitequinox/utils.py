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

#------------------------------ paths ---------------------------------------

#tmp = os.getenv('TMPDIR')
datawork = os.getenv('DATAWORK')+'/'
home = os.getenv('HOME')+'/'
scratch = os.getenv('SCRATCH')+'/'
osi = '/home/datawork-lops-osi/aponte/'
#
root_data_dir = '/home/datawork-lops-osi/data/mit4320/'
grid_dir = root_data_dir+'grid/'


#------------------------------ mit specific ---------------------------------------

#
def get_compressed_level_index(grid_dir, index_fname='llc4320_compressed_level_index.nc', geometry='llc'):
    ''' Some doc
    '''
    #
    ds = xm.open_mdsdataset('', grid_dir=grid_dir,
                             iters=None, geometry=geometry, read_grid=True,
                             default_dtype=np.dtype('>f4'),
                             ignore_unknown_vars=True)
    
    # get shape
    #nz, nface, ny, nx = ds.hFacC.shape
    #shape = (1, nface, ny, nx)
    
    try:
        ds_index = xr.open_dataset(grid_dir+index_fname)
    except OSError:
        # compute and save mask indices
        print('Create llc4320_compressed_level_index.nc in grid_dir')
        ds_index = ((ds.reset_coords()[['hFacC', 'hFacW','hFacS']] > 0).sum(axis=(1, 2, 3)))
        ds_index.coords['k'] = ds.k
        ds_index.load().to_netcdf(grid_dir+index_fname)
        print('done')

    return ds_index, ds
    

def load_level_from_3D_field(data_dir, varname, inum, offset, count, mask, dtype):
    ''' Some doc
    '''

    # all iters in one directory:
    inum_str = '%010d' % inum
    if 'Eta' in varname:
        suff = '.data.shrunk'
    else:
        suff = '.shrunk'            
    fname = os.path.join(data_dir, '%s.%s' % (varname, inum_str) +suff)
    
    with open(fname, mode='rb') as file:
        file.seek(offset * dtype.itemsize)
        data = np.fromfile(file, dtype=dtype, count=count)
    
    data_blank = np.full_like(mask, np.nan, dtype='f4')
    data_blank[mask] = data
    data_blank.shape = mask.shape
    data_llc = xm.utils._reshape_llc_data(data_blank, jdim=0).compute(get=dask.get)
    data_llc.shape = (1,) + data_llc.shape
    return data_llc


def lazily_load_level_from_3D_field(data_dir, varname, inum, offset, count, mask, shape, dtype):
    ''' Some doc
    '''
    return dask.array.from_delayed(dask.delayed(load_level_from_3D_field)
                            (data_dir, varname, inum, offset, count, mask, dtype),
                            shape, dtype)


def get_compressed_data(varname, data_dir, grid_dir, ds_index=None, ds=None, iters='all', 
                        time=None, client=None, k=0, point='C', **kwargs):
    ''' Get mitgcm compressed data
    
    Parameters
    ----------
    varname: string
        Variable name to load (should allow for list?)
    data_dir: string
        Path to the directory where the mds .data and .meta files are stored
    grid_dir: string
        Path to the directory where grid files are stored
    ds_index: xarray Dataset, optional
        Contains compressed file
    iters: list, 'all', optional
        The iterations numbers of the files to be read. If 'all' (default), all iterations 
        will be read.
    k: int
        vertical level loaded
    point: string
        grid point used for the mask
    '''
    dtype = np.dtype('>f4')
    
    if ds_index is None or shape is None or ds is None:
        ds_index, ds = get_compressed_level_index(grid_dir, **kwargs)
        # get shape
        nz, nface, ny, nx = ds.hFacC.shape
        shape = (1, nface, ny, nx)        
        
    strides = [0,] + list(ds_index['hFac' + point].data)
    offset = strides[k]
    count = strides[k+1]
    
    if iters is 'all':
        iters = xm.mds_store._get_all_iternums(data_dir, file_prefixes=varname, 
                                               file_format='*.??????????.data.shrunk')
    
    # load mask from raw data
    hfac = xm.utils.read_mds(grid_dir + 'hFac' + point,
                            use_mmap=True, dask_delayed=False, force_dict=False)
    mask = hfac[k]>0
    if client is None:
        mask_future = mask
    else:
        mask_future = client.scatter(mask)
    
    data = dask.array.concatenate([lazily_load_level_from_3D_field
                            (data_dir, varname, i, offset, count, mask_future, shape, dtype)
                            for i in iters], axis=0)

    ds[varname] = xr.Variable(['time', 'face', 'j', 'i'], data)   
    if time is not None:
        ds['time'] = time.sel(iters=iters).values
        #ds['dtime'] = iters_to_date(iters)
        ds = ds.assign_coords(dtime=xr.DataArray(iters_to_date(iters), dims=['time']))
        #ds = ds.assign(dtime=)
        
    return ds


def get_iters_time(varname, data_dir, delta_t=25.):
    ''' get iteration numbers and derives corresponding time
    Parameters
    ----------
    varname: string
        Variable name to load (should allow for list?)
    data_dir: string
        Path to the directory where the mds .data and .meta files are stored
    delta_t: float
        Model time step
    
    Returns
    -------
    iters: xarray DataArray
        iteration numbers indexed by time
    time: xarray DataArray
        time in seconds
    '''
    file_suff = '.shrunk'
    if varname is 'Eta':
        file_suff = '.data.shrunk'
    #
    iters = xm.mds_store._get_all_iternums(data_dir, file_prefixes=varname, 
                                           file_format='*.??????????'+file_suff)
    time = delta_t * np.array(iters)
    
    iters = xr.DataArray(iters, coords=[time], dims=['time'])
    time = xr.DataArray(time, coords=[iters.values], dims=['iters'])
    
    return iters, time

def iters_to_date(iters, delta_t=25.):
    t0 = datetime.datetime(2011,9,13)    
    ltime = delta_t * (np.array(iters)-10368)
    dtime = [t0+dateutil.relativedelta.relativedelta(seconds=t) for t in ltime]    
    return dtime


#------------------------------ plot ---------------------------------------

#
def plot_scalar(v, colorbar=False, title=None, vmin=None, vmax=None, savefig=None, 
                offline=False, coast_resolution='110m', figsize=(10,10), cmmap='thermal'):
    #
    if vmin is None:
        vmin = v.min()
    if vmax is None:
        vmax = v.max()
    #
    MPL_LOCK = threading.Lock()
    with MPL_LOCK:
        if offline:
            plt.switch_backend('agg')
        #
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        #ax = fig.add_subplot(111)
        cmap = getattr(cm, cmmap)
        try:
            im = v.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax,
                                   x='XC', y='YC', add_colorbar=colorbar, cmap=cmap)
            #im = v.plot.pcolormesh(ax=ax, add_colorbar=False, cmap=cmap)
            fig.colorbar(im)
            gl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='k', 
                            alpha=0.5, linestyle='--')
            gl.xlabels_top = False
            ax.coastlines(resolution=coast_resolution, color='k')
        except:
            pass
        #
        if title is not None:
            ax.set_title(title)
        #
        if savefig is not None:
            fig.savefig(savefig, dpi=150)
            plt.close(fig)
        #
        if not offline:
            plt.show()


#------------------------------ zarr ---------------------------------

def zarr(V, client, F=None, out_dir=None, compressor=None):
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
        ds = get_compressed_data(v, data_dir, grid_dir, iters=iters, time=time, client=client, point=p)
        #
        # should store grid data independantly in a single file
        ds = ds.drop(['XC','YC','Depth','rA'])
        #
        ds = ds.isel(face=F)
        ds = ds.chunk({'face': 1})
        #
        dv = ds[v].to_dataset()
        #
        file_out = out_dir+'/%s.zarr'%(v)
        try:
            dv.to_zarr(file_out, mode='w')                    
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
    
    Nc; int
        i, j chunk size, default is 96 (96x45=4320)
        
    """

    if out_dir is None:
        out_dir = scratch+'mit/rechunked/'

    if F is None:
        F=range(13)
    elif type(F) is int:
        F = [F]
        
    Nt = len(ds.time) if Nt == 0 else Nt

    for v in V:

        file_in = scratch+'/mit/standard/%s.zarr'%(v)
        ds0 = xr.open_zarr(file_in)

        for face in F:

            ds = ds0.isel(face=face)
            #
            ds = ds.isel(time=slice(len(ds.time)//Nt *Nt))
            #
            ds = ds.chunk({'time': Nt, 'i': Nc, 'j': Nc})
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


#------------------------------ spectrum ---------------------------------------
            
def _get_E(x, ufunc=True, **kwargs):
    ax = -1 if ufunc else 0
    #
    dkwargs = {'window': 'hann', 'return_onesided': False, 
               'detrend': 'linear', 'scaling': 'density'}
    dkwargs.update(kwargs)
    f, E = welch(x, fs=24., axis=ax, **dkwargs)
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