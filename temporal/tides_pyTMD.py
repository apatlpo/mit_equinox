import os
from glob import glob

import numpy as np
import xarray as xr
import pandas as pd

import pyTMD


r2d = 180/np.pi
d2r = np.pi/180
cpd = 86400/2/np.pi


#fes_dir = "/home/datawork-lops-osi/equinox/misc/tides/FES2014"
fes_dir = "/Users/aponte/Data/tides/fes2014/"
TIDE_MODEL = "FES2014"
GZIP = False
model_format = 'FES'


# ------------------------------------- constituents and equilibrium tides -------------------------------

default_fes_constituents = \
            ['2n2','eps2','j1','k1','k2','l2','lambda2','m2','m3','m4','m6',
            'm8','mf','mks2','mm','mn4','ms4','msf','msqm','mtm','mu2','n2',
            'n4','nu2','o1','p1','q1','r2','s1','s2','s4','sa','ssa','t2']
semidiurnal = ["m2", "s2", "n2", "k2", "nu2", "mu2", "l2", "t2", "2n2"]
diurnal = ["k1", "o1", "p1", "q1", "j1"]
major_semidiurnal = ["m2", "s2"]
major_diurnal = ["k1", "o1"]

def load_constituents_properties(c=None):
    """
    Parameters
    ----------
    c: str, list
        constituent or list of constituent

    Returns
    -------
    amplitude: amplitude of equilibrium tide in m for tidal constituent
    phase: phase of tidal constituent
    omega: angular frequency of constituent in radians
    alpha: load love number of tidal constituent
    species: spherical harmonic dependence of quadrupole potential
    """
    if c is None:
        c = constituents
    if isinstance(c, list):
        df = (pd.DataFrame({_c: load_constituents_properties(_c) for _c in c}).T)
        df = df.sort_values("omega")
        return df
    elif isinstance(c, str):
        p_names = ["amplitude", "phase", "omega", "alpha", "species"] 
        p = pyTMD.load_constituent(c)
        s = pd.Series({_n: _p for _n, _p in zip(p_names, p)})
        s["omega_cpd"] = s["omega"]*cpd
        return s

# load all constituents properties and hold them
cproperties = load_constituents_properties(default_fes_constituents)
    
def plot_equilibrium_amplitudes(c, ax=None, xlim=None, ylim=None):

    _c = c.loc[c.amplitude>0]
    if xlim is not None:
        _c = _c.loc[ (c.omega*cpd>xlim[0]) & (c.omega*cpd<xlim[1]) ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))
        #plt.xticks(rotation=90)
    
    ax.stem(_c.omega*cpd, _c.amplitude, markerfmt=' ')

    # annotate lines
    #vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
    for d, l, r in zip(_c.omega*cpd, _c.amplitude, _c.index):
        ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),
                    textcoords="offset points", va="top", ha="right")

    ax.set_ylabel("[m]")
    ax.set_title("Equilibrium tide amplitudes")
    ax.grid()
    ax.set_xlabel("frequency [cpd]")
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax    
    

def get_tidal_arguments(time):
    """ compute tidal arguments for predictions """

    if isinstance(time, dict):
        time = pd.date_range(**time)

    # https://github.com/tsutterley/pyTMD/blob/main/pyTMD/predict_tidal_ts.py 
    # https://github.com/tsutterley/pyTMD/blob/main/pyTMD/load_nodal_corrections.py

    #-- convert from calendar date to days relative to Jan 1, 1992 (48622 MJD)
    tide_time = pyTMD.time.convert_calendar_dates(time.year, time.month,
                                                  time.day, hour=time.hour, 
                                                  minute=time.minute,
                                                 )
    #-- delta time (TT - UT1) file
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])
    #deltat = pyTMD.calc_delta_time(delta_file, tide_time)
    deltat = pyTMD.time.interpolate_delta_time(delta_file, tide_time)

    pu, pf, G = pyTMD.load_nodal_corrections(tide_time + 48622.0, default_fes_constituents,
                                             deltat=deltat, corrections=model_format,
                                            )
    # pu, pf are nodal corrections
    # G is the equilibrium argument and common to all models

    ds = xr.Dataset(None,
                    coords=dict(time=("time", time),
                                constituents=("constituents", default_fes_constituents)),
                   )
    ds["pu"] = (("time","constituents"), pu)
    ds["pf"] = (("time","constituents"), pf)
    ds["G"] = (("time","constituents"), G)
    
    params = (cproperties
              .to_xarray()
              .rename(index="constituents")
             )
    ds = xr.merge([ds, params])
    
    ds["th"] = ds.G*np.pi/180.0 + ds.pu
    
    ds["ht_no_hc"] = ds.pf*np.exp(1j*ds.th)
    
    return ds    
    
    
    
# ------------------------------------- amplitudes & predictions -------------------------------

def load_tidal_amplitudes(lon, lat, vtype, **kwargs):
    """ wrapper in order to produce an xarray dataset

    https://pytmd.readthedocs.io/en/latest/api_reference/io/FES.html

    Parameters:
    lon: tuple, np.array
        Longitude. Tuple that can be fed to np.arange, or array
    lat: tuple, np.array
        Latitude. Tuple that can be fed to np.arange, or array
    broadcast: boolean, optinal
    time: dict, pd.DatetimeIndex
        Dates when tides need to be predicted. 
        Dict that can be fed to pd.date_range (keys are start, end, freq)
        or DatetimeIndex
    
    """    
    if isinstance(vtype, list):
        return xr.merge([load_tidal_amplitudes(lon, lat, v, **kwargs) for v in vtype])

    ds, lon_np, lat_np, stacked = prepare_output_dataset(lon, lat)

    hc, c = load_tidal_amplitudes_np(lon_np, lat_np, vtype, **kwargs)

    ds[vtype+"_amplitude"] = (ds.lon.dims+("constituents",), hc)    
    ds = ds.assign_coords(constituents=c)
    
    if stacked:
        ds = ds.unstack()
    
    return ds

def load_tidal_amplitudes_np(lon, lat, vtype, constituents=None):
    """ Load tidal amplitudes
    """
    
    model_directory, scale = get_dirs_scale(vtype)

    # flatten arrays
    ll_shape = lon.shape
    LON=lon.flatten().astype(float)
    LAT=lat.flatten().astype(float)
    
    # set path and constituents
    if constituents is not None:
        c = constituents
    else:
        c = default_fes_constituents
    model_files = [_c+".nc" for _c in c]
    for i in range(len(model_files)):
        if model_files[i]=="lambda2.nc":
            model_files[i]="la2.nc"
    model_files = [os.path.join(model_directory, m) for m in model_files]

    constituents = pyTMD.io.FES.read_constants(model_files,
        type=vtype, version=TIDE_MODEL,
        compressed=GZIP)
    
    #-- read tidal constants and interpolate to grid points
    kw = dict(type=vtype,
       version=TIDE_MODEL,
       method='spline',
       compressed=GZIP,
       scale=scale,
    )
    
    amp,ph = pyTMD.io.FES.interpolate_constants(
            np.atleast_1d(LON), np.atleast_1d(LAT),
            constituents, **kw)
    #TYPE = 'z' # variable type, can be "u", "v"
    #SCALE = 1.0/100.0
    #GZIP = False
    #amp,ph = pyTMD.extract_FES_constants(LON, LAT,
    #    model_file, TYPE=TYPE, VERSION=TIDE_MODEL, METHOD='spline',
    #    EXTRAPOLATE=True, SCALE=SCALE, GZIP=GZIP)
    #-- calculate complex phase in radians for Euler's
    cph = -1j*ph*np.pi/180.0
    #-- calculate constituent oscillation
    hc = amp*np.exp(cph)

    return np.reshape(hc, ll_shape+(len(c),)), c

def load_raw_tidal_amplitudes(vtype, lon=None, lat=None, constituents=None):
    """ wrapper in order to produce an xarray dataset

    https://pytmd.readthedocs.io/en/latest/api_reference/io/FES.html

    Parameters:
    lon: tuple, np.array
        Longitude. Tuple that can be fed to np.arange, or array
    lat: tuple, np.array
        Latitude. Tuple that can be fed to np.arange, or array
    time: dict, pd.DatetimeIndex
        Dates when tides need to be predicted. 
        Dict that can be fed to pd.date_range (keys are start, end, freq)
        or DatetimeIndex
    
    """
    if isinstance(vtype, list):
        return xr.merge([load_raw_tidal_amplitudes(v, lon=lon, lat=lat, 
                                                   constituents=constituents
                                                  ) 
                         for v in vtype]
                       )
    
    model_directory, scale = get_dirs_scale(vtype)
    
    # set path and constituents
    if constituents is None:
        constituents = default_fes_constituents
    model_files = [c+".nc" for c in constituents]
    for i in range(len(model_files)):
        if model_files[i]=="lambda2.nc":
            model_files[i]="la2.nc"
    model_files = [os.path.join(model_directory, m) for m in model_files]

    if isinstance(lon, tuple):
        lon = slice(lon[0], lon[1])
    elif lon is None:
        lon = slice(None, None)
    if isinstance(lat, tuple):
        lat = slice(lat[0], lat[1])
    elif lat is None:
        lat = slice(None, None)
    
    ds = xr.concat([load_fes_file(f, lon, lat, c)
                    for f, c  in zip(model_files, constituents)
                   ],
                   "constituents",
                  )
    # add vtype suffix
    if vtype=="z":
        ds = ds.rename(amplitude="sea_level_amplitude", phase="sea_level_phase")
    elif vtype=="u":
        ds = ds.rename(Ua="zonal_current_amplitude", Ug="zonal_current_phase")        
    elif vtype=="v":
        ds = ds.rename(Va="meridional_current_amplitude", Vg="meridional_current_phase")
    
    return ds


# ------------------------------------- predictions ------------------------------------

def tidal_prediction(lon, lat, time, vtype, constituents=None, minor=True):
    """ tidal prediction:
    Parameters
    ----------
    lon, lat: tuple, list, np.array, xr.DataArray
    
    """
    if isinstance(vtype, list):
        return xr.merge([tidal_prediction(lon, lat, time, v, 
                                             constituents=constituents, minor=minor) 
                         for v in ["z", "u", "v"]
                        ])
    
    ds, lon_np, lat_np, stacked = prepare_output_dataset(lon, lat)

    if constituents is None:
        constituents = default_fes_constituents
        
    v_tide = tidal_prediction_np(lon_np, lat_np, time, vtype, constituents, minor)    
    ds[vtype+"_tide"] = (ds.lon.dims+("time",), v_tide)
    ds = ds.assign_coords(time=time)
    
    if stacked:
        ds = ds.unstack()

    return ds

def tidal_prediction_np(lon, lat, time, vtype, constituents, minor):
    """ Predict tidal fluctuations with pyTMD
    
    Parameters:
    lon, lat: longitude, latitude arrays, should be flat
    time: time array
    vtype: str, "z", "u", or "v"
    constituents: list
    minor: boolean, infer minor constituents
    """
    
    model_directory, scale = get_dirs_scale(vtype)
    
    model_files = [c+".nc" for c in constituents]
    for i in range(len(model_files)):
        if model_files[i]=="lambda2.nc":
            model_files[i]="la2.nc"
    model_files = [os.path.join(model_directory, m) for m in model_files]
    
    # flatten arrays
    LON=lon.astype(float)
    LAT=lat.astype(float)
    # reset longitudes to adequate convention
    LON %= 360

    #-- convert from calendar date to days relative to Jan 1, 1992 (48622 MJD)
    tide_time = pyTMD.time.convert_calendar_dates(time.year, time.month,
        time.day, hour=time.hour, minute=time.minute)
    #-- delta time (TT - UT1) file
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])
    #-- interpolate delta times from calendar dates to tide time
    #deltat = pyTMD.calc_delta_time(delta_file, tide_time)
    deltat = pyTMD.time.interpolate_delta_time(delta_file, tide_time)
    
    # interpolate harmonic amplitudes
    amp,ph = pyTMD.io.FES.extract_constants(lon, lat, model_files,
            type=vtype, version=TIDE_MODEL, method='spline',
            scale=scale, compressed=GZIP)
        
    # calculate complex phase in radians for Euler's
    cph = -1j*ph*np.pi/180.0
    # calculate constituent oscillation
    hc = amp*np.exp(cph)

    #-- predict tidal variable at time 1 and infer minor corrections    
    
    # timeseries based approach
    TIDE = np.ma.zeros((LON.size, tide_time.size))
    for i in range(len(LON)):
        TIDE[i, :] = pyTMD.predict.time_series(tide_time, hc[[i],:], constituents,
                                      deltat=deltat, corrections=model_format)
        if minor:
            MINOR = pyTMD.predict.infer_minor(tide_time, hc[[i],:], constituents,
                deltat=deltat, corrections=model_format)
            TIDE[i, :] += MINOR[:]
    tide_out = TIDE
    
    # map based approach
    # allocate for tide current map calculated every hour
    #tide_out = np.ma.zeros((lon.size, time.size))
    #for t in range(time.size):
    #    # predict tidal elevations at time and infer minor corrections
    #    TIDE = pyTMD.predict.map(tide_time[t], hc, constituents, deltat=deltat[t],
    #        corrections=model_format)
    #    tide_out[:, t] = TIDE
    #    if False:
    #        MINOR = pyTMD.predict.infer_minor(tide_time[t], hc, constituents, deltat=deltat[t], 
    #            corrections=model_format)
    #        # add major and minor components and reform grid
    #        #tide[TYPE][:,:,t] = np.reshape((TIDE+MINOR),(lon.size))
    #        tide_out[:, t] += MINOR

    return tide_out



# ------------------------------- common methods -----------------------------------


def get_dirs_scale(vtype):
    """ get FES directories and variables scales based on variable type: 
                "z", "u", or "v" 
    """
    if vtype=="z":
        tide_dir = os.path.join(fes_dir, "fes2014_elevations_and_load/fes2014b_elevations")
        #model_directory = os.path.join(tide_dir,'fes2014','ocean_tide')
        model_directory = os.path.join(tide_dir,'ocean_tide')
        scale = 1/100
    elif vtype=="u":
        tide_dir = os.path.join(fes_dir, "fes2014a_currents")
        model_directory = os.path.join(tide_dir,'eastward_velocity')
        scale = 1/100
    elif vtype=="v":
        tide_dir = os.path.join(fes_dir, "fes2014a_currents")
        model_directory = os.path.join(tide_dir,'northward_velocity')
        scale = 1/100
    return model_directory, scale

def load_fes_file(f, lon, lat, c):
    ds = xr.open_dataset(f)
    #
    ds["lon"] = (ds["lon"]+180)%360 - 180
    ds = ds.sortby("lon")
    ds = (ds.sel(lon=lon, lat=lat)
          .expand_dims(constituents=[c])
    )
    #
    ds = ds.drop_dims("nv", errors="ignore")
    if "crs" in ds:
        ds = ds.drop("crs")
    ds = ds.assign_coords(**{v: ("constituents", cproperties[v].loc[ds.constituents])
                    for v in ["omega", "omega_cpd"]}
                )
    return ds

def prepare_output_dataset(lon, lat):
    """ prepare an xarray output dataset """
    
    stacked=False
    
    if isinstance(lon, xr.DataArray) and isinstance(lat, xr.DataArray):
        # no broadcasting, stacks dimensions
        if sorted(lon.dims)==sorted(lat.dims) and len(lon.dims)>1:
            dims = lon.dims
            lon = lon.stack(points_tmp=[...])
            lat = lat.stack(points_tmp=[...])
            stacked=True
        ds = xr.merge([lon, lat])
    elif isinstance(lon, tuple) and isinstance(lat, tuple):
        # broadcasting
        print("broadcasting lon/lat")
        lon = np.arange(*lon)
        lat = np.arange(*lat)
        ds = xr.Dataset(None, 
                        coords=dict(lon=("lon", lon),
                                    lat=("lat", lat),
                                   )
                       )
        ds = ds.stack(points_tmp=[...])
        stacked=True
        #lon_np = (ds.lon + ds.lat*0).values
        #lat_np = (ds.lon*0 + ds.lat).values
    elif len(lon)==len(lat):
        # no broadcasting
        print("not broadcasting because len(lon)==len(lat)")
        ds = xr.Dataset(None, 
                        coords=dict(lon=("lon", lon),
                                    lat=("lat", lat),
                                   ),
                       )
    lon_np = ds.lon.values
    lat_np = ds.lat.values

    return ds, lon_np, lat_np, stacked
