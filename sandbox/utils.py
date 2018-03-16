
import os

import numpy as np
import dask
import xarray as xr

import xmitgcm as xm



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

    inum_str = '%010d' % inum
    # all iters in one directory:
    fname = os.path.join(data_dir, '%s.%s.data.shrunk' % (varname, inum_str))
    
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
    iters = xm.mds_store._get_all_iternums(data_dir, file_prefixes=varname, file_format='*.??????????.data.shrunk')
    time = delta_t * np.array(iters)
    
    iters = xr.DataArray(iters, coords=[time], dims=['time'])
    time = xr.DataArray(time, coords=[iters.values], dims=['iters'])
    
    return iters, time
    