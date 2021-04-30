from glob import glob
import pickle

import numpy as np
import pandas as pd
import dask.bag as db
import xarray as xr

import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely import wkt
import pyproj
crs_wgs84 = pyproj.CRS("EPSG:4326")
import pyinterp.geohash as geohash

from functools import partial

# from .utils import fix_lon_bounds, work_data_dir
from .utils import fix_lon_bounds, load_oceans


# --------------------------------------  pair I/O ------------------------------------------------

dr_data_dir = "/work/ALT/swot/aval/syn/drifters/"


def load_from_ID(ID):
    try:
        return pickle.load(open(dr_data_dir + "single/argos_%09d.p" % ID, "rb"))
    except:
        try:
            return pickle.load(open(dr_data_dir + "single/gps_%09d.p" % ID, "rb"))
        except:
            print("ID=%d has not data file in %ssingle/ directory" % (ID, data_dir))


def load_pair(p, data_dir):
    d0, id0 = load_from_ID(p[0])
    d1, id1 = load_from_ID(p[1])
    # get rid of gaps and interpolate if necessary
    d0 = d0[~pd.isnull(d0.index)]
    d1 = d1[~pd.isnull(d1.index)]
    #
    d0, d1 = d0.align(d1, join="inner")
    # fill gaps, should keep track of this
    d0, d1 = compute_vector(d0, d1)
    # should go through vectors for interpolation
    # d0 = d0.resample('H').interpolate('linear')
    # d1 = d1.resample('H').interpolate('linear')
    d0 = d0.resample("H").asfreq()
    d1 = d1.resample("H").asfreq()
    d0, d1 = compute_lonlat(d0, d1)
    # converts to geopandas
    gd0 = to_gdataframe(d0)
    gd1 = to_gdataframe(d1)
    return gd0, gd1, p


def _to_dict(x):
    out = x[0]
    out["ID"] = x[1]
    return out.to_dict()


def load_single_df(npartitions=100, gps=None):
    """could also load directly from netcdf files"""
    if gps is None:
        files = sorted(glob(dr_data_dir + "single/*.p"))
    elif gps == 0:
        files = sorted(glob(dr_data_dir + "single/argos_*.p"))
    elif gps == 1:
        files = sorted(glob(dr_data_dir + "single/gps_*.p"))
    b = db.from_sequence(files[:], npartitions=npartitions).map(
        lambda f: pickle.load(open(f, "rb"))
    )
    return b.map(_to_dict).to_dataframe()


def load_single_df_fromnc(npartitions=100, gps=None):

    if gps is None or gps == 1:
        ds = xr.open_dataset(dr_data_dir + "raw/driftertrajGPS_1.02.nc")
        ds = ds.chunk({"TIME": 24 * 1000})
        ds["GPS"] = (1 + ds.U * 0.0).astype(int)
        ds_GPS = ds

    if gps is None or gps == 0:
        ncfile = dr_data_dir + "raw/driftertrajWMLE_1.02_block1.nc"
        ds = xr.open_mfdataset(dr_data_dir + "raw/driftertrajWMLE_1.02_block*.nc")
        ds = ds.chunk({"TIME": 24 * 1000})
        ds["GPS"] = (0.0 + ds.U * 0.0).astype(int)
        ds_ARGOS = ds

    if gps is None:
        ds = xr.concat([ds_GPS, ds_ARGOS], dim="TIME")

    ds["LON"] = fix_lon_bounds(ds["LON"])

    return ds.to_dask_dataframe()

# ---------------------------  lon/lat grid - geohash --------------------------

def generate_grid_and_geohashes(dl,
                                lat_bounds=(-80,80),
                                geohash=False,
                                plot=False,
                                buffered=False,
                                ):
    """ Generate a regular lon/lat grid
    """

    # generate grid
    lon = np.arange(-180, 180, dl)
    lat = np.arange(lat_bounds[0], lat_bounds[1], dl)
    lon2, lat2 = np.meshgrid(lon, lat)

    # associated shapely objects and geopandas dataframe
    def get_poly(lon, lat, dlon, dlat):
        return ((lon, lat),
                (lon+dlon, lat),
                (lon+dlon, lat+dlat),
                (lon, lat+dlat), (lon, lat),
                )
    _polygons = [Polygon(get_poly(_lon,_lat, dl, dl))
                 for _lon, _lat in zip(lon2.flatten(), lat2.flatten())]
    #
    gdf = gpd.GeoDataFrame(
        None, geometry=_polygons, crs=crs_wgs84
    )

    # filter oceans out
    ocean = load_oceans(database="ne_110m_ocean")

    # select only boxes that are within ocean
    gdf = gpd.sjoin(gdf, ocean, how='inner', op='intersects')
    gdf = gdf.drop(columns=["index_right",
                            "scalerank",
                            "featurecla",
                            "min_zoom",
                           ])
    if plot:
        gdf.plot()

    if geohash:
        if isinstance(geohash, int):
            geohash_resolution=geohash
        else:
            # default resolution
            geohash_resolution=2
        hash_boxes = get_geohashes(geohash_resolution)
        gdf = (gpd
               .sjoin(gdf, hash_boxes, how='left', op='intersects')
               .rename(columns=dict(index_right="geohash"))
               )

    if not buffered:
        return gdf

    # buffer each polygons
    gdf_buff = gdf.copy()
    gdf_buff['geometry'] = gdf.buffer(dl)

    if geohash:
        gdf_buff = (gpd
                      .sjoin(gdf_buff, hash_boxes, how='left', op='intersects')
                      .rename(columns=dict(index_right="geohash"))
                     )

    return gdf, gdf_buff


def get_geohashes(resolution):

    # generate all geohases for one resolution
    codes = geohash.bounding_boxes(None, precision=2)

    hash_boxes = gpd.GeoDataFrame(
        {'geohash': codes},
        geometry=[wkt.loads(geohash.bounding_box(code).wkt()) for code in codes],
        crs=crs_wgs84,
    ).set_index("geohash")

    return hash_boxes


# --------------------------------------  binning ------------------------------------------------

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
    return h[None, :]


def bin1d(
    v, vbin, bins, weights, bin_dim="bin_dim", name="binned_array", core_dim="TIME"
):
    # wrapper around apply_ufunc
    dims = [core_dim]  # core dim
    bins_c = (bins[:-1] + bins[1:]) * 0.5
    out = xr.apply_ufunc(
        partial(_bin1d, bins),
        v,
        vbin,
        kwargs={"weights": weights},
        output_core_dims=[[bin_dim]],
        output_dtypes=[np.float64],
        dask="parallelized",
        input_core_dims=[dims, dims],
        output_sizes={bin_dim: len(bins_c)},
    )
    out = out.assign_coords(**{bin_dim: bins_c}).rename(name)
    return out


def _bin2d(bins1, bins2, v, vbin1, vbin2, weights=True):
    # wrapper around apply_ufunc
    if weights:
        w = v
    else:
        w = np.ones_like(v)
    h, edges1, edges2 = np.histogram2d(
        vbin1.flatten(),
        vbin2.flatten(),
        bins=[bins1, bins2],
        weights=w.flatten(),
        density=False,
    )
    return h[None, ...]


def bin2d(
    v,
    vbin1,
    bins1,
    vbin2,
    bins2,
    weights,
    bin_dim1="bin_dim1",
    bin_dim2="bin_dim2",
    name="binned_array",
):
    # wrapper around apply_ufunc
    dims = ["TIME"]  # core dim
    bins1_c = (bins1[:-1] + bins1[1:]) * 0.5
    bins2_c = (bins2[:-1] + bins2[1:]) * 0.5
    out = xr.apply_ufunc(
        partial(_bin2d, bins1, bins2),
        v,
        vbin1,
        vbin2,
        kwargs={"weights": weights},
        output_core_dims=[[bin_dim1, bin_dim2]],
        output_dtypes=[np.float64],
        dask="parallelized",
        input_core_dims=[dims, dims, dims],
        output_sizes={bin_dim1: len(bins1_c), bin_dim2: len(bins2_c)},
    )
    out = out.assign_coords(**{bin_dim1: bins1_c, bin_dim2: bins2_c}).rename(name)
    return out


# ------------------------------- time window  processing -----------------------------------------


def time_window_processing(
    df,
    myfun,
    columns,
    T,
    N,
    spatial_dims=None,
    Lx=None,
    overlap=0.5,
    id_label="id",
    dt=None,
    **myfun_kwargs,
):
    """Break each drifter time series into time windows and process each windows

    Parameters
    ----------
        df: Dataframe
            This dataframe represents a drifter time series
        T: float
            Length of the time windows, must be in the same units that column "time"
        myfun
            Method that will be applied to each window
        columns: list of str
            List of columns of df that will become inputs of myfun
        N: int
            Length of myfun outputs
        spatial_dims: tuple, optional
            Tuple indicating column labels for spatial coordinates.
            Guess otherwise
        Lx: float
            Domain width for periodical domains in x direction
        overlap: float
            Amount of overlap between temporal windows.
            Should be between 0 and 1.
            Default is 0.5
        id_label: str, optional
            Label used to identify drifters
        dt: float, str
            Conform time series to some time step, if string must conform to rule option of
            pandas resample method
        **myfun_kwargs
            Keyword arguments for myfun

    """
    if hasattr(df, id_label):
        dr_id = df[id_label].unique()[0]
    elif df.index.name == id_label:
        dr_id = df.index.unique()[0]
    elif hasattr(df, "name"):
        # when mapped after groupby
        dr_id = df.name
    else:
        assert False, "Cannot find float id"
    #
    dim_x, dim_y, geo = guess_spatial_dims(df)
    if geo:
        df = compute_vector(df, lon_key=dim_x, lat_key=dim_y)
    #
    # drop duplicated values
    df = df.drop_duplicates(subset="date")
    #p = p.where(p.time.diff() != 0).dropna() # duplicates - old
    #
    df = df.sort_values("time")
    #
    if dt is not None:
        if isinstance(dt, float):
            # enforce regular sampling
            tmin, tmax = df.index[0], df.index[-1]
            tmax = tmin+int((tmax-tmin)/dt)*dt
            regular_time = np.arange(tmin, tmax, dt)
            df = df.reindex(regular_time).interpolate()
        elif isinstance(dt, str):
            df = df.set_index("date").resample(dt).pad().reset_index()
            # by default converts to days then
            dt = pd.Timedelta(dt)/pd.Timedelta('1D')
        if geo:
            df = compute_lonlat(
                df,
                lon_key=dim_x,
                lat_key=dim_y,
            )
    #
    df = df.set_index("time")
    tmin, tmax = df.index[0], df.index[-1]
    #
    # need to create an empty dataframe, in case the loop below is empty
    # get column names from fake output:
    myfun_out = myfun(*[None for c in columns], N, dt=dt, **myfun_kwargs)
    #
    columns_out = [dim_x, dim_y] + ["id"] + list(myfun_out.index)
    out = pd.DataFrame({c: [] for c in columns_out})
    t = tmin
    while t + T < tmax:
        #
        _df = df.loc[t : t + T].iloc[:-1, :]
        # note: iloc is here because pandas include the last date
        # compute average position
        x, y = mean_position(_df, Lx=Lx)
        # apply myfun
        myfun_out = myfun(*[_df[c] for c in columns], N, dt=dt, **myfun_kwargs)
        # combine with mean position and time
        out.loc[t + T / 2.0] = [x, y] + [dr_id] + list(myfun_out)
        t += T * (1 - overlap)
    return out


def mean_position(df, Lx=None):
    """Compute the mean position of a dataframe

    Parameters:
    -----------
        df: dafaframe
            dataframe containing position data
        Lx: float, optional
            Domain width for periodical domains
    """
    # guess grid type
    dim_x, dim_y, geo = guess_spatial_dims(df)
    #lon = next((c for c in df.columns if "lon" in c.lower()), None)
    #lat = next((c for c in df.columns if "lat" in c.lower()), None)
    if geo:
        lon, lat = dim_x, dim_y
        if "v0" not in df:
            df = compute_vector(df, lon_key=lon, lat_key=lat)
        mean = compute_lonlat(
            df.mean(),
            dropv=True,
            lon_key=lon,
            lat_key=lat,
        )
        return mean[lon], mean[lat]
    else:
        if Lx is not None:
            x = (
                (
                    np.angle(np.exp(1j * (df[dim_x] * 2.0 * np.pi / L - np.pi)).mean())
                    + np.pi
                )
                * Lx
                / 2.0
                / np.pi
            )
        else:
            x = df[dim_x].mean()
        y = df[dim_y].mean()
        return x, y


def get_spectrum(v, N, dt=None, method="welch", detrend="linear", **kwargs):
    """Compute a lagged correlation between two time series
    These time series are assumed to be regularly sampled in time
    and along the same time line.

    Parameters
    ----------

        v: ndarray, pd.Series
            Time series, the index must be time if dt is not provided

        N: int
            Length of the output

        dt: float, optional
            Time step

        method: string
            Method that will be employed for spectral calculations.
            Default is 'welch'

        detrend: boolean, optional
            Turns detrending on or off. Default is 'linear'.

    See:
        - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.periodogram.html
        - https://krischer.github.io/mtspec/
        - http://nipy.org/nitime/examples/multi_taper_spectral_estimation.html
    """
    if v is None:
        _v = np.random.randn(N)
    else:
        _v = v.iloc[:N]
    if dt is None:
        dt = _v.reset_index()["index"].diff().mean()
    if detrend and not method == "welch":
        print("!!! Not implemented yet except for welch")
    if method == "welch":
        from scipy import signal
        dkwargs = {
            "window": "hann",
            "return_onesided": False,
            "detrend": None,
            "scaling": "density",
        }
        dkwargs.update(kwargs)
        f, E = signal.periodogram(_v, fs=1 / dt, axis=0, **dkwargs)
    elif method == "mtspec":
        from mtspec import mtspec
        lE, f = mtspec(
            data=_v, delta=dt, time_bandwidth=4.0, number_of_tapers=6, quadratic=True
        )
    elif method == "mt":
        import nitime.algorithms as tsa
        dkwargs = {"NW": 2, "sides": "twosided", "adaptive": False, "jackknife": False}
        dkwargs.update(kwargs)
        lf, E, nu = tsa.multi_taper_psd(_v, Fs=1 / dt, **dkwargs)
        f = fftfreq(len(lf)) * 24.0
        # print('Number of tapers = %d' %(nu[0]/2))
    return pd.Series(E, index=f)


# --------------------------------------  utils ------------------------------------------------

earth_radius = 6378.0
deg2rad = np.pi / 180.0

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


# geodesy with vectors
# https://www.movable-type.co.uk/scripts/latlong-vectors.html
def compute_vector(*args, lon_key="LON", lat_key="LAT"):
    if len(args) == 1:
        df = args[0]
        df.loc[:, "v0"] = np.cos(deg2rad * df[lat_key]) * np.cos(deg2rad * df[lon_key])
        df.loc[:, "v1"] = np.cos(deg2rad * df[lat_key]) * np.sin(deg2rad * df[lon_key])
        df.loc[:, "v2"] = np.sin(deg2rad * df[lat_key])
        return df
    else:
        return [compute_vector(df) for df in args]


def drop_vector(*args, v0="v0", v1="v1", v2="v2"):
    # should test for type in order to accomodate for xarray types
    if len(args) == 1:
        return args[0].drop(columns=[v0, v1, v2])
    else:
        return [df.drop(columns=[v0, v1, v2]) for df in args]


def compute_lonlat(
    *args,
    dropv=True,
    v0="v0",
    v1="v1",
    v2="v2",
    lon_key="LON",
    lat_key="LAT",
):
    if len(args) == 1:
        df = args[0]
        # renormalize vectors
        n = np.sqrt(df[v0] ** 2 + df[v1] ** 2 + df[v2] ** 2)
        df[v0], df[v1], df[v2] = df[v0] / n, df[v1] / n, df[v2] / n
        # estimate LON/LAT
        df[lon_key] = np.arctan2(df[v1], df[v0]) / deg2rad
        df[lat_key] = np.arctan2(df[v2], np.sqrt(df[v0] ** 2 + df[v1] ** 2)) / deg2rad
        if dropv:
            df = drop_vector(df, v0=v0, v1=v1, v2=v2)
        return df
    else:
        return [
            compute_lonlat(
                df, dropv=dropv, v0=v0, v1=v1, v2=v2, lon_key=lon_key, lat_key=lat_key
            )
            for df in args
        ]


def to_gdataframe(*args):
    if len(args) == 1:
        df = args[0]
        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.LON, df.LAT)
        )
    else:
        return [
            gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.LON, df.LAT)
            )
            for df in args
        ]


# def load_depth():
#    ds = xr.open_dataset(work_data_dir+'bathy/ETOPO1_Ice_g_gmt4.grd')
#    ds = ds.sortby(ds.x)
#    ds['x2'] = (ds.x+0.*ds.y).transpose()
#    ds['y2'] = (0.*ds.x+ds.y).transpose()
#    return ds


def compute_depth(lon, lat):
    depth = load_depth()
    lon = (lon - 180) % 360 - 180
    return -depth.z.sel(x=lon, y=lat, method="nearest")


def guess_spatial_dims(df):
    """Guess spatial dimensions
    Detects longitude, latitude first
    Serach then for x/y
    """
    lon = next((c for c in df.columns if "lon" in c.lower()), None)
    lat = next((c for c in df.columns if "lat" in c.lower()), None)
    if lon is not None and lat is not None:
        return lon, lat, True
    if "x" in df and "y" in df:
        return "x", "y", False
