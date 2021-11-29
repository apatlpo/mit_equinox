import os
from glob import glob

import numpy as np
import xarray as xr
import pandas as pd

import geopandas as gpd
from shapely.geometry import Polygon

import dateutil
from datetime import timedelta, datetime

from dask.distributed import wait

from tqdm import tqdm

# ideal dask size
_chunk_size_threshold = 4000*3000

# ------------------------------ parameters, geometry -------------------------------------

g = 9.81
omega_earth = 2.0 * np.pi / 86164.0905
deg2rad = np.pi / 180.0
deg2m = 111319
earth_radius = 6378.0

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
    arclen = 2 * np.arcsin(
        np.sqrt(
            (np.sin((llat2 - llat1) / 2)) ** 2
            + np.cos(llat1) * np.cos(llat2) * (np.sin((llon2 - llon1) / 2)) ** 2
        )
    )
    return arclen * earth_radius

def coriolis(lat, signed=True):
    if signed:
        return 2.0 * omega_earth * np.sin(lat * deg2rad)
    else:
        return 2.0 * omega_earth * np.sin(np.abs(lat) * deg2rad)


def dfdy(lat, units="1/s/m"):
    df = 2.0 * omega_earth * np.cos(lat * deg2rad) * deg2rad / deg2m
    if units == "cpd/100km":
        df = df * 86400 / 2.0 / np.pi * 100 * 1e3
    return df


def fix_lon_bounds(lon):
    """ reset longitudes bounds to (-180,180)"""
    if isinstance(lon, xr.DataArray):  # xarray
        return lon.where(lon < 180.0, other=lon - 360)
    elif isinstance(lon, pd.Series):  # pandas series
        out = lon.copy()
        out[out > 180.0] = out[out > 180.0] - 360.0
        return out


# ------------------------------ paths ---------------------------------------

if os.path.isdir("/home/datawork-lops-osi/"):
    # datarmor
    platform = "datarmor"
    datawork = os.getenv("DATAWORK") + "/"
    home = os.getenv("HOME") + "/"
    scratch = os.getenv("SCRATCH") + "/"
    osi = "/home/datawork-lops-osi/"
    #
    root_data_dir = "/home/datawork-lops-osi/equinox/mit4320/"
    ref_data_dir = "/dataref/ocean-analysis/intranet/LLC4320_surface/"
    work_data_dir = root_data_dir
    #
    bin_data_dir = root_data_dir + "bin/"
    bin_grid_dir = bin_data_dir + "grid/"
    #
    zarr_data_dir = ref_data_dir
    zarr_grid = zarr_data_dir + "grid.zarr"
    mask_path = zarr_data_dir + "mask.zarr"
    #
    diag_dir = os.path.join(root_data_dir, "diags/")
elif os.path.isdir("/work/ALT/swot/"):
    # hal
    platform = "hal"
    tmp = os.getenv("TMPDIR")
    home = os.getenv("HOME") + "/"
    scratch = os.getenv("HOME") + "/scratch/"
    #
    root_data_dir = "/work/ALT/swot/swotpub/LLC4320/"
    work_data_dir = "/work/ALT/swot/aval/syn/"
    #
    diag_dir = os.path.join(work_data_dir, "diags/")
    #
    enatl60_data_dir = "/work/ALT/odatis/eNATL60/"
elif os.path.isdir("/Users/aponte"):
    # laptop
    platform = "laptop"


# ------------------------------ mit specific ---------------------------------------


def load_grd(V=None, ftype="zarr", **kwargs):
    """ Load llc4320 grid
    Parameters
    ----------
        V: str, list, optional
        List of coordinates to select
    """
    if ftype == "zarr":
        ds = xr.open_zarr(ref_data_dir + "grid.zarr", **kwargs)
        if V is not None:
            ds = ds.reset_coords()[V].set_coords(names=V)
    elif ftype == "nc":
        ds = _load_grd_nc(V, **kwargs)
    return ds


def _load_grd_nc(V):
    _hasface = [
        "CS",
        "SN",
        "Depth",
        "dxC",
        "dxG",
        "dyC",
        "dyG",
        "hFacC",
        "hFacS",
        "hFacW",
        "rA",
        "rAs",
        "rAw",
        "XC",
        "YC",
        "XG",
        "YG",
    ]
    gfiles = glob(grid_dir + "netcdf/grid/*.nc")
    gv = [f.split("/")[-1].split(".")[0] for f in gfiles]
    if V is not None:
        gfiles = [f for f, v in zip(gfiles, gv) if v in V]
        gv = V
    objs = []
    for f, v in zip(gfiles, gv):
        if v in _hasface:
            objs.append(xr.open_dataset(f, chunks={"face": 1}))
        else:
            objs.append(xr.open_dataset(f))
    return xr.merge(objs, compat="equals").set_coords(names=gv)


def load_data(V, ftype="zarr", merge=True, **kwargs):
    if isinstance(V, list):
        out = [load_data(v, ftype=ftype, **kwargs) for v in V]
        if merge:
            return xr.merge(out, join='inner')
        else:
            return out
        return
    else:
        if ftype == "zarr":
            return load_data_zarr(V, **kwargs)
        elif ftype == "nc":
            return load_data_nc(V, **kwargs)


def load_data_zarr(v, **kwargs):
    return xr.open_zarr(ref_data_dir + v + ".zarr", **kwargs)


def load_data_nc(v, suff="_t*", files=None, **kwargs):
    default_kwargs = {
        "concat_dim": "time",
        "compat": "equals",
        "chunks": {"face": 1, "i": 480, "j": 480},
        "parallel": True,
    }
    if v == "SSU":
        default_kwargs["chunks"] = {"face": 1, "i_g": 480, "j": 480}
    elif v == "SSV":
        default_kwargs["chunks"] = {"face": 1, "i": 480, "j_g": 480}
    default_kwargs.update(kwargs)
    #
    files_in = root_data_dir + "netcdf/" + v + "/" + v + suff
    if files is not None:
        files_in = files
    ds = xr.open_mfdataset(files_in, **default_kwargs)
    ds = ds.assign_coords(
        dtime=xr.DataArray(
            iters_to_date(ds.iters.values), coords=[ds.time], dims=["time"]
        )
    )
    return ds

def load_llc(v, dij, t_start, t_end):
    """ load llc data and subsample
    """
    if isinstance(v, list):
        return xr.merge([load_llc(_v, dij, t_start, t_end) for _v in v])
    ds = (load_data(V=[v])
          .sel(time=slice(str(t_start), str(t_end)))
         )

    i, j = get_ij_dims(ds[v])
    ds = ds.rename({i: 'i', j: 'j'})
    ds = ds.isel(i=slice(0, None, dij),
                 j=slice(0, None, dij),
                )
    coords = ['XC', 'YC']
    if v in ['SSU', 'SSV']:
        coords = coords + ['CS', 'SN']
    grd = (load_grd()[coords]
           .isel(i=slice(0, None, dij),
                 j=slice(0, None, dij),
                )
          )
    #       .load_grd()[['XC', 'YC', 'XG', 'YG']]
    llc = xr.merge([ds, grd])
    return llc

def load_iters_date_files(v="Eta"):
    """For a given variable returns a dataframe of available data"""
    files = sorted(glob(root_data_dir + "netcdf/" + v + "/" + v + "_t*"))
    iters = [int(f.split("_t")[-1].split(".")[0]) for f in files]
    date = iters_to_date(iters)
    d = [{"date": d, "iter": i, "file": f} for d, i, f in zip(date, iters, files)]
    return pd.DataFrame(d).set_index("date")


def iters_to_date(iters, delta_t=25.0):
    t0 = datetime(2011, 9, 13)
    ltime = delta_t * (np.array(iters) - 10368)
    dtime = [t0 + dateutil.relativedelta.relativedelta(seconds=t) for t in ltime]
    return dtime


def load_common_timeline(V, raw=False, verbose=True):
    df = load_iters_date_files(V[0]).rename(columns={"file": "file_" + V[0]})
    for v in V[1:]:
        ldf = load_iters_date_files(v)
        ldf = ldf.rename(columns={"file": "file_" + v})
        df = df.join(ldf["file_" + v], how="inner")
    if verbose:
        print(df.index[0], " to ", df.index[-1])
    return df

def find_ijface(lon, lat, face=None, radius=2):
    """ Find indexes based on lon/lat within a given radius

    Parameters
    ----------
    lon, lat: floats
    face: int
    radius: float
        in km !!
    """

    grd = load_grd(["XC", "YC"])

    if face is not None:
        grd = grd.sel(face=face)

    #d = (np.cos(np.pi/180*grd.)*(grd.XC-lon))**2 + (grd.XC-lat)**2
    d = haversine(lon, lat, grd.XC, grd.YC)
    dmin = d.min().compute()

    d_loc = d.where(d<dmin+radius, drop=True).compute()
    i = {d: slice(int(min(d_loc[d])), int(max(d_loc[d]))+1) for d in ["i", "j", "face"]}
    i["i_g"] = i["i"]
    i["j_g"] = i["j"]

    return i

# ------------------------------ diagnosics -----------------------------------

def store_diagnostic(name, data,
                     overwrite=False,
                     file_format=None,
                     directory=None,
                     auto_rechunk=True,
                     **kwargs
                    ):
    """ Write diagnostics to disk
    Parameters
    ----------
    name: str
        Name of a diagnostics to store on disk
    data: xr.Dataset, xr.DataArray (other should be implemented)
        Data to be stored
    overwrite: boolean, optional
        Overwrite an existing diagnostic. Default is False
    file_format: str, optional
        Storage file format (supported at the moment: zarr, netcdf)
    directory: str, optional
        Directory path where diagnostics will be stored (absolute or relative to output directory).
        Default is 'diagnostics/'
    auto_rechunk: boolean, optional
        Automatically rechunk diagnostics such as o ensure they are not too small.
        Default is True.
    **kwargs:
        Any keyword arguments that will be passed to the file writer
    """
    if directory is None:
        directory = diag_dir
    # create diagnostics dir if not present
    _dir = _check_diagnostic_directory(directory)
    if auto_rechunk:
        data = _auto_rechunk(data)
    #
    if isinstance(data, xr.DataArray):
        store_diagnostic(name, data.to_dataset(),
                         overwrite=overwrite,
                         file_format=file_format,
                         directory=directory,
                         auto_rechunk=True,
                         **kwargs,
                        )
    elif isinstance(data, xr.Dataset):
        success=False
        if file_format is None or file_format.lower() in ['zarr', '.zarr']:
            _file = os.path.join(_dir, name+'.zarr')
            write_kwargs = dict(kwargs)
            if overwrite:
                write_kwargs.update({'mode': 'w'})
            data = _move_singletons_as_attrs(data)
            data = _reset_chunk_encoding(data)
            data.to_zarr(_file, **write_kwargs)
            success=True
        elif file_format.lower() in ['nc', 'netcdf']:
            _file = os.path.join(_dir, name+'.nc')
            write_kwargs = dict(kwargs)
            if overwrite:
                write_kwargs.update({'mode': 'w'})
            data.to_netcdf(_file, **write_kwargs)
            success=True
        if success:
            print('data stored in {}'.format(_file))

def is_diagnostic(name,
                  directory=None,
                 ):
    """ Indicates whether diagnostic zarr archive exists
    Obviously does not indicate wether it is complete
    """
    if directory is None:
        directory = diag_dir
    zarr_dir = os.path.join(directory, name+'.zarr')
    return os.path.isdir(zarr_dir)

def load_diagnostic(name,
                    directory=None,
                    **kwargs):
    """ Load diagnostics from disk
    Parameters
    ----------
    name: str, list
        Name of a diagnostics or list of names of diagnostics to load
    directory: str, optional
        Directory where diagnostics will be stored (absolute or relative to output directory).
        Default is 'diagnostics/'
    **kwargs:
        Any keyword arguments that will be passed to the file reader
    """
    if directory is None:
        directory = diag_dir
    _dir = _check_diagnostic_directory(directory)
    # find the diagnostic file
    _file = glob(os.path.join(_dir,name+'.*'))
    assert len(_file)==1, 'More that one diagnostic file {}'.format(_file)
    _file = _file[0]
    # get extension
    _extension = _file.split('.')[-1]
    if _extension=='zarr':
        return xr.open_zarr(_file, **kwargs)
    elif _extension=='nc':
        return xr.open_dataset(_file, **kwargs)
    else:
        raise NotImplementedError('{} extension not implemented yet'
                                  .format(_extension))

def _check_diagnostic_directory(directory,
                                create=True,
                               ):
    """ Check existence of a directory and create it if necessary

    Parameters
    ----------
    directory: str
        Path of directory
    create: boolean
        Create directory if not there, default is True
    """
    # create diagnostics dir if not present
    if not os.path.isdir(directory):
        if create:
            # need to create the directory
            os.mkdir(directory)
            print('Create new diagnostic directory {}'.format(directory))
        else:
            raise OSError('Directory does not exist')
    return directory

def _move_singletons_as_attrs(ds, ignore=[]):
    """ change singleton variables and coords to attrs
    This seems to be required for zarr archiving
    """
    for c,co in ds.coords.items():
        if co.size==1 and ( len(co.dims)==1 and co.dims[0] not in ignore or len(co.dims)==0 ):
            value = ds[c].values
            if isinstance(value, np.datetime64):
                value = str(value)
            ds = ds.drop_vars(c).assign_attrs({c: value})
    for v in ds.data_vars:
        if ds[v].size==1 and ( len(v.dims)==1 and v.dims[0] not in ignore or len(v.dims)==0 ):
            ds = ds.drop_vars(v).assign_attrs({v: ds[v].values})
    return ds

def _reset_chunk_encoding(ds):
    ''' Delete chunks from variables encoding.
    This may be required when loading zarr data and rewriting it with different chunks

    Parameters
    ----------
    ds: xr.DataArray, xr.Dataset
        Input data
    '''
    if isinstance(ds, xr.DataArray):
        return _reset_chunk_encoding(ds.to_dataset()).to_array()
    #
    for v in ds.coords:
        if 'chunks' in ds[v].encoding:
            del ds[v].encoding['chunks']
    for v in ds:
        if 'chunks' in ds[v].encoding:
            del ds[v].encoding['chunks']
    return ds

def _check_chunks_sizes(da):
    """ checks that chunk sizes are above the _chunk_size_threshold
    """
    averaged_chunk_size, total_size = _get_averaged_chunk_size(da)
    assert averaged_chunk_size==total_size or averaged_chunk_size>_chunk_size_threshold, \
        '{} chunks are two small, rechunk such that chunk sizes'.format(da.name) \
        + ' exceed {} elements on average,'.format(_chunk_size_threshold) \
        + ' there are currently ' \
        + '{} points per chunks on average'.format(averaged_chunk_size)

def _get_averaged_chunk_size(da):
    """ returns the averaged number of elements in the dataset
    """
    # total number of elements
    total_size = int(np.array(list(da.sizes.values())).prod())
    # averaged number of chunks along each dimension:
    if da.chunks:
        chunk_size_dims = np.array([np.max(d) for d in da.chunks])
        chunk_size = int(chunk_size_dims.prod())
    else:
        chunk_size = total_size
    return chunk_size, total_size

def _auto_rechunk_da(da):
    """ Automatically rechunk a DataArray such as chunks number of elements
    exceeds _chunk_size_threshold
    """
    dims = ['i', 'i_g', 'j', 'j_g', 'face',
            'k', 'lon', 'lat',
            'time']
    for d in dims:
        # gather da number of elements and chunk sizes
        averaged_chunk_size, total_size = _get_averaged_chunk_size(da)
        # exit if there is one chunk
        if averaged_chunk_size==total_size:
            break
        # rechunk along dimenion d
        if (d in da.dims) and averaged_chunk_size<_chunk_size_threshold:
            dim_chunk_size = np.max(da.chunks[da.get_axis_num(d)])
            # simple rule of 3
            factor = max(1, np.ceil(_chunk_size_threshold/averaged_chunk_size))
            new_chunk_size = int( dim_chunk_size * factor )
            # bounded by dimension size
            new_chunk_size = min(da[d].size, new_chunk_size)
            da = da.chunk({d: new_chunk_size})
    return da

def _auto_rechunk(ds):
    """ Wrapper around _auto_rechunk_da for datasets
    Accepts DataArrays as well however.
    """
    if isinstance(ds, xr.DataArray):
        return _auto_rechunk_da(ds)
    for k, da in ds.items(): # data_vars
        ds = ds.assign(**{k: _auto_rechunk_da(da)})
    for k, da in ds.coords.items():
        ds = ds.assign_coords(**{k: _auto_rechunk_da(da)})
    return ds

# ------------------------------ misc ---------------------------------------

face_connections = {'face': {0: {'X': ((12, 'Y', False), (3, 'X', False)),
              'Y': (None, (1, 'Y', False))},
          1: {'X': ((11, 'Y', False), (4, 'X', False)),
              'Y': ((0, 'Y', False), (2, 'Y', False))},
          2: {'X': ((10, 'Y', False), (5, 'X', False)),
              'Y': ((1, 'Y', False), (6, 'X', False))},
          3: {'X': ((0, 'X', False), (9, 'Y', False)),
              'Y': (None, (4, 'Y', False))},
          4: {'X': ((1, 'X', False), (8, 'Y', False)),
              'Y': ((3, 'Y', False), (5, 'Y', False))},
          5: {'X': ((2, 'X', False), (7, 'Y', False)),
              'Y': ((4, 'Y', False), (6, 'Y', False))},
          6: {'X': ((2, 'Y', False), (7, 'X', False)),
              'Y': ((5, 'Y', False), (10, 'X', False))},
          7: {'X': ((6, 'X', False), (8, 'X', False)),
              'Y': ((5, 'X', False), (10, 'Y', False))},
          8: {'X': ((7, 'X', False), (9, 'X', False)),
              'Y': ((4, 'X', False), (11, 'Y', False))},
          9: {'X': ((8, 'X', False), None),
              'Y': ((3, 'X', False), (12, 'Y', False))},
          10: {'X': ((6, 'Y', False), (11, 'X', False)),
               'Y': ((7, 'Y', False), (2, 'X', False))},
          11: {'X': ((10, 'X', False), (12, 'X', False)),
               'Y': ((8, 'Y', False), (1, 'X', False))},
          12: {'X': ((11, 'X', False), None),
               'Y': ((9, 'Y', False), (0, 'X', False))}}}

def getsize(dir_path):
    """Returns the size of a directory in bytes"""
    process = os.popen("du -s " + dir_path)
    size = int(process.read().split()[0])  # du returns kb
    process.close()
    return size * 1e3


def rotate(u, v, ds):
    # rotate from grid to zonal/meridional directions
    return u * ds.CS - v * ds.SN, u * ds.SN + v * ds.CS


def get_ij_dims(da):
    i = next((d for d in da.dims if d[0] == "i"))
    j = next((d for d in da.dims if d[0] == "j"))
    return i, j

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def custom_distribute(ds, op,
                      tmp_dir=None,
                      suffix="tmp",
                      overwrite=True,
                      append=False,
                      _root=True,
                      op_kwargs={},
                      **kwargs,
                     ):
    """ Distribute an embarrasingly parallel calculation manually and store chunks to disk
    Data can be written temporarily to disk with a restart capability or directly persisted
    in memory.

    Example usages:
    ds_out = custom_distribute(ds, lambda ds: ds.mean("time"), dim_0=2) # persists data
    ds_out = custom_distribute(ds, lambda ds: ds.mean("time"), dim_0=2, tmp_dir="/path/to/tmp/") # writes data on disk

    Parameters
    ----------
    ds: xr.Dataset
        Input data
    op: func
        Process the data and return a dataset
    tmp_dir: str, optional
        temporary output directory, persist data in memory
    suffix: str, optional
        suffix employed for temporary files
    overwrite: optional, boolean
        set to False if you do not want to overwrite existing data. Default is True
    append: optional, boolean
        append to a single zarr archive 
        ! only works for distribution only a single dimension at the moment !
    op_kwargs: dict, optional
        pass kwargs to op
    **kwargs:
        dimensions with chunk size, e.g. (..., dim_0=2) processes data sequentially in chunks
        of size 2 along dimension dim_0
    """
    
    d = list(kwargs.keys())[0]
    c = kwargs[d]

    new_kwargs = removekey(kwargs, d)
    dim = ds[d].values
    chunks = [dim[i*c:(i+1)*c] for i in range((dim.size + c - 1) // c )]

    # various conditions should be fullfilled:
    assert not overwrite or tmp_dir is not None, "tmp_dir is required if `overwrite=False`"
    assert not append or not new_kwargs, "Appending data to a single zarr archive is"\
        +" not implemented for multiple dimensions"
    assert overwrite or not append, "overwrite=False and append=True should not make sense"\
        +" at the moment"
    assert not append and c>1, "the size of the chunks along the selected dimension needs to"\
        +" be greater than 1 if you want to append to a single zarr archive, see "\
        +" https://github.com/pydata/xarray/issues/4084"
    
    D, Z = [], []
    iterator = zip(chunks, range(len(chunks)))
    if _root:
        iterator = tqdm(iterator)
        # tqdm may be redirected to the logger and a file:
        # https://github.com/tqdm/tqdm/issues/313
    for c, i in iterator:
        #_ds = ds.isel(**{d: slice(c[0], c[-1]+1)})
        _ds = ds.sel(**{d: c})
        _suffix = suffix+"_{}".format(i)
        if new_kwargs:
            ds_out, _Z = custom_distribute(_ds, op,
                                           tmp_dir=tmp_dir,
                                           suffix=_suffix,
                                           _root=False,
                                           op_kwargs=op_kwargs,
                                           **new_kwargs,
                                          )
            D.append(ds_out)
            Z.append(_Z)
        else:
            ds_out = op(_ds, **op_kwargs)
            if tmp_dir is not None:
                # store
                if append:
                    zarr = os.path.join(tmp_dir, suffix)
                    if not os.path.isdir(zarr): # not sure this is necessary
                        ds_out.to_zarr(zarr, mode="w")
                    else:
                        ds_out.to_zarr(zarr, append_dim=d)
                else:
                    zarr = os.path.join(tmp_dir, _suffix)
                    if overwrite or not os.path.isdir(zarr):
                        ds_out.to_zarr(zarr, mode="w")
                    Z.append(zarr)
                    # load
                    ds_out = xr.open_zarr(zarr)
            else:
                # persist data and wait for completion
                ds_out = ds_out.persist()
                _ = wait(ds_out)
            if not append:
                D.append(ds_out)

    if append:
        ds = xr.open_zarr(zarr)
        Z = zarr
    else:
        # concatenate results and return
        ds = xr.concat(D, d)

    return ds, Z

def custom_distribute_concat(ds,
                             tmp_dir,
                             suffix="tmp",
                             _root=True,
                             **kwargs,
                            ):
    """ Load and concatenate data tempory files

    Parameters
    ----------
    ds: xr.Dataset
        Input data
    tmp_dir: str
        temporary output directory, persist data in memory
    suffix: str, optional
        suffix employed for temporary files
    **kwargs:
        dimensions with chunk size, e.g. (..., dim_0=2) processes data sequentially in chunks
        of size 2 along dimension dim_0
    """

    d = list(kwargs.keys())[0]
    c = kwargs[d]

    new_kwargs = removekey(kwargs, d)

    dim = ds[d].values
    chunks = [dim[i*c:(i+1)*c] for i in range((dim.size + c - 1) // c )]

    D = []
    Z = []
    iterator = zip(chunks, range(len(chunks)))
    for c, i in iterator:
        _ds = ds.sel(**{d: c})        
        _suffix = suffix+"_{}".format(i)
        if new_kwargs:
            ds_out, _Z = custom_distribute_concat(_ds,
                                                  tmp_dir,
                                                  suffix=_suffix,
                                                  _root=False,
                                                  **new_kwargs,
                                                 )
            D.append(ds_out)
            Z.append(_Z)
        else:
            # load
            zarr = os.path.join(tmp_dir, _suffix)
            Z.append(zarr)
            ds_out = xr.open_zarr(zarr)
            D.append(ds_out)

    # merge results back and return
    ds = xr.concat(D, d)

    return ds, Z

def filter_llc_regionally(t_range,
                          v=None,
                          faces=None,
                          dij=None,
                          zarr=None,
                          overwrite=True,
                          background=None,
                         ):
    """ Select
    """

    if background is not None:
        v = background["v"]
        faces = background["region"]["faces"]
        dij = background["dij"]
    else:
        assert v is not None
        assert faces is not None
        assert dij is not None

    if isinstance(t_range, pd.Index):
        t_range = tuple(t_range[[0,-1]])

    if v in ['SSU', 'SSV']:
        V = ['SSU', 'SSV']
    else:
        V = v
    llc = load_llc(V, dij, t_range[0], t_range[1])
    llc = llc.sel(face=faces)

    llc = llc.chunk({'time': 1, 'i': -1, 'j': -1})
    if 'chunks' in llc.niter.encoding:
        del llc.niter.encoding['chunks']

    if zarr is None:
        zarr = os.path.join(scratch, 'zoom_llc')
    #
    if overwrite:
        mode='w'
    else:
        mode='w-'
    llc.to_zarr(zarr, mode=mode)
    return zarr

# ------------------------------ enatl60 specific ---------------------------------------


def load_enatl60_UV(chunks={"x": 480, "y": 480}):
    ds_U = xr.open_zarr(enatl60_data_dir + "zarr/eNATL60-BLBT02-SSU-1h")
    ds_V = xr.open_zarr(enatl60_data_dir + "zarr/eNATL60-BLBT02-SSV-1h")
    ds_V = ds_V.drop("nav_lon")
    ds = xr.merge([ds_U, ds_V])
    ds = ds.rename(
        {
            "time_counter": "time",
            "nav_lon": "lon",
            "nav_lat": "lat",
            "sozocrtx": "SSU",
            "somecrty": "SSV",
        }
    )
    ds = ds.set_coords(["lon", "lat"])
    ds = ds.chunk(chunks)
    return ds


def load_enatl60_grid(chunks={"x": 440, "y": 440}):
    grd = (
        xr.open_dataset(enatl60_data_dir + "mesh_mask_eNATL60_3.6.nc", chunks=chunks)
        .squeeze()
        .rename({"nav_lon": "lon", "nav_lat": "lat", "nav_lev": "z"})
        .set_coords(["lon", "lat", "z"])
    )
    return grd


# ------------------------------ time relative ---------------------------------------


def np64toDate(dt64, tzinfo=None):
    """
    Converts a Numpy datetime64 to a Python datetime.
    :param dt64: A Numpy datetime64 variable, type dt64: numpy.datetime64
    :param tzinfo: The timezone the date / time value is in, type tzinfo: pytz.timezone
    :return: A Python datetime variablertype: datetime
    """
    ts = pd.to_datetime(dt64)
    if tzinfo is not None:
        return datetime(
            ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, tzinfo=tzinfo
        )
    return datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)


def dateRange(date1, date2, dt=timedelta(days=1.0)):
    for n in np.arange(date1, date2, dt):
        yield np64toDate(n)


# ------------------------------ misc data ---------------------------------------

def load_bathy(subsample=None, extent=None, land=False):
    """ Load bathymetry (etopo1)

    Parameters
    ----------
        subsample: int, optional
            subsampling parameter:
                30 leads to 1/2 deg resolution
                15 leads to 1/4 deg resolution
    """
    if platform=="datarmor":
        path = os.path.join(osi, "equinox/misc/bathy/ETOPO1_Ice_g_gmt4.grd")
    ds = xr.open_dataset(path)
    if subsample is not None:
        ds = ds.isel(x=slice(0, None, subsample),
                     y=slice(0, None, subsample),
                    )
    if extent is not None:
        ds = ds.sel(x=slice(extent[0], extent[1]),
                    y=slice(extent[2], extent[3]),
                   )
    da = -ds['z'].rename("h")
    if not land:
        da = da.where(da>=0)
    da = da.rename({'x':'lon', 'y':'lat',})
    da.attrs["long_name"] = "depth"
    # long_name (plot)
    return da


def load_oceans(database="IHO", features=["oceans"]):
    """ Load Oceans, Seas and other features shapes

    Usage
    -----
    oceans = load_oceans()
    oceans.loc[oceans.name == 'North Atlantic Ocean'].plot()

    Parameters
    ----------
    database: str, optional,
        Database to load, available: "IHO" (default), "ne_110m_ocean"
    features: tuple, optional
        Features to output: ("oceans", "seas", "other")

    Notes
    -----
    IHO ocean seas shapefiles from https://marineregions.org/downloads.php
    Ocean seas shapefiles from: http://www.naturalearthdata.com/downloads/

    """
    if platform=="datarmor":
        root_shape_dir = os.path.join(osi, "equinox/misc/")
    elif platform=="laptop":
        root_shape_dir = "/Users/aponte/Data/shapefiles/"
    if database=="IHO":
        path = os.path.join(root_shape_dir,
                            "World_Seas_IHO_v3/World_Seas_IHO_v3.shp"
                           )
        gdf = gpd.read_file(path)
        gdf = gdf.rename(columns={c: c.lower() for c in gdf.columns})
        out = {}
        if "oceans" in features:
            out["oceans"] = gdf.loc[gdf.name.str.contains('Ocean')]
        if "seas" in features:
            out["seas"] = gdf.loc[gdf.name.str.contains('Sea')]
        if "other" in features:
            out["other"] = gdf.loc[~gdf.name.str.contains('Ocean|Sea')]
        if len(out)==1:
            return list(out.values())[0]
        else:
            return out
    elif database=="ne_110m_ocean":
        path = os.path.join(root_shape_dir,
                            "ne_110m_ocean/ne_110m_ocean.shp",
                           )
        out = gpd.read_file(path)
        return out

def load_swot_tracks(phase="calval", resolution=None, bbox=None, **kwargs):
    """ Load SWOT tracks

    Parameters
    ----------
    phase: str, optional
        "calval" or "science"
    resolution: str, optional
        Specify resolution, for example "10s", default is "30s"
    """

    if platform=="datarmor":
        tracks_dir = "/home/datawork-lops-osi/equinox/misc/swot"
    else:
        tracks_dir = "/Users/aponte/Data/swot"
    #
    files = glob(os.path.join(tracks_dir, "*.shp"))
    files = [f for f in files if phase in f]
    if resolution is not None:
        files = [f for f in files if resolution in f]
    dfiles = {f.split("_")[-1].split(".")[0]: f for f in files}
    out = {key: gpd.read_file(f, **kwargs) for key, f in dfiles.items()}

    if bbox is None:
        return out

    central_lon = (bbox[0]+bbox[1])*0.5
    central_lat = (bbox[2]+bbox[3])*0.5

    polygon = Polygon([(bbox[0], bbox[2]),
                       (bbox[1], bbox[2]),
                       (bbox[1], bbox[3]),
                       (bbox[0], bbox[3]),
                       (bbox[0], bbox[2]),
                      ])
    out = {key: gpd.clip(gdf, polygon) for key, gdf in out.items()}

    return out
