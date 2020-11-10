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
        
        
#------------------------------ parquet relative ---------------------------------------

def store_parquet(run_dir, 
              df,
              partition_size='100MB', 
              index='trajectory',
              overwrite=False,
              engine = 'auto',
              compression='ZSTD',
             ):
    """ store data under parquet format

    Parameters
    ----------
    run_dir: str, path to the simulation
    df: dask dataframe to store
    partition_size: str, optional
        size of each partition that will be enforced
        Default is '100MB' which is dask recommended size
    index: str, which index to set before storing the dataframe
    overwrite: bool, can overwrite or not an existing archive
    engine: str, engine to store the parquet format
    compression: str, type of compression to use when storing in parquet format
    """
    
    # check if right value for index
    columns_names = df.columns.tolist()+[df.index.name]
    if index not in columns_names:
        print('Index must be in ', columns_names)
        return
    
    parquet_path = os.path.join(run_dir,'drifters',index)

    # check wether an archive already exists
    if os.path.isdir(parquet_path):
        if overwrite:
            print('deleting existing archive: {}'.format(parquet_path))
            shutil.rmtree(parquet_path)
        else:
            print('Archive already existing: {}'.format(parquet_path))
            return

    # create archive path   
    os.system('mkdir -p %s' % parquet_path)
    print('create new archive: {}'.format(parquet_path))
    
    # change index of dataframe
    if df.index.name != index:
        df = df.reset_index()
        df = df.set_index(index).persist()

    # repartition such that each partition is 100MB big
    df.to_parquet(parquet_path, engine=engine,compression=compression)

def load_parquet(run_dir,index='trajectory'):
        """ load data into a dask dataframe
        
        Parameters
        ----------
            run_dir: str, path to the simulation (containing the drifters directory)
            index: str, to set the path and load a dataframe with the right index
        """
        parquet_path = os.path.join(run_dir,'drifters',index) 
        
        # test if parquet
        if os.path.isdir(parquet_path):
            return dd.read_parquet(parquet_path,engine='fastparquet')
        else:
            print("load_parquet error: directory not found ",parquet_path)
            return None
        
#------------------------------ h3 relative ---------------------------------------

def h3_index(df, resolution=2):
    """
    Add an H3 geospatial indexing system column to the dataframe
    parameters:
    ----------
    df : dask dataframe to which the new column is added
    resolution : int, cell areas for H3 Resolution (0-15)
                  see https://h3geo.org/docs/core-library/restable for more information
    """
    def get_hex(row, resolution, *args, **kwargs):
        return h3.geo_to_h3(row["lat"], row["lon"], resolution)

    # resolution = 2 : 86000 km^2
    df['hex_id'] = (df.apply(get_hex, axis=1, 
                            args=(resolution,), meta='string') # use 'category' instead?
                    )
    return df

def add_lonlat(df, reset_index=False):
    if reset_index:
        df = df.reset_index()
    df['lat'] = df['hex_id'].apply(lambda x: h3.h3_to_geo(x)[0])
    df['lon'] = df['hex_id'].apply(lambda x: h3.h3_to_geo(x)[1])
    return df
    
def id_to_bdy(hex_id):
    hex_boundary = h3.h3_to_geo_boundary(hex_id) # array of arrays of [lat, lng]                    
    hex_boundary = hex_boundary+[hex_boundary[0]]
    return [[h[1], h[0]] for h in hex_boundary]

def plot_h3_simple(df, metric_col, x='lon', y='lat', marker='o', alpha=1, 
                 figsize=(16,12), colormap='viridis'):
    df.plot.scatter(x=x, y=y, c=metric_col, title=metric_col
                    , edgecolors='none', colormap=colormap, 
                    marker=marker, alpha=alpha, figsize=figsize);
    plt.xticks([], []); plt.yticks([], [])
    
        
#------------------------------ netcdf floats relative ---------------------------------------

def load_cdf(run_dir, step_tile='*',index='trajectory'):
    """
    Load floats netcdf files from a parcel simulation
    run_dir: str, directory of the simulation
    step_tile: str, characteristic string to select the floats to load. (default=*)
               name of the files are floats_xxx_xxx.nc, first xxx is step, second is tile
               ex: floats_002_023.nc step step 2 of tile 23
               step_tile can be 002_* for step 2 of every tile or *_023 for every step of the tile 23
    index : str, column to set as index for the returned dask dataframe ('trajectory','time',
            'lat','lon', or 'z')
    """
    def xr2df(file):
        return xr.open_dataset(file).to_dataframe().set_index(index)

    # find list of tile directories
    tile_dir = os.path.join(run_dir,'tiling/')
    tl = pa.tiler(tile_dir=tile_dir) 
    tile_data_dirs = [os.path.join(run_dir,'data_{:03d}'.format(t)) 
                      for t in range(tl.N_tiles)
                     ]
    
    # find list of netcdf float files
    float_files = []
    for _dir in tile_data_dirs:
        float_files.extend(sorted(glob.glob(_dir+"/floats_"+step_tile+".nc")))
    
    # read the netcdf files and store in a dask dataframe
    lazy_dataframes = [delayed(xr2df)(f) for f in float_files]
    _df = lazy_dataframes[0].compute()
    df = dd.from_delayed(lazy_dataframes, meta=_df).repartition(partition_size='100MB').persist()
    return df