import os
import numpy as np
import dask
import xarray as xr

import datetime, dateutil

import xmitgcm as xm

from .utils import *

# ------------------------------ mit specific ---------------------------------------

_vars_datashrunk = ["Eta", "oceTAUX", "oceTAUY", "KPPhbl"]


def get_iters_time(varname, data_dir, delta_t=25.0):
    """get iteration numbers and derives corresponding time
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
    """
    file_suff = ".shrunk"
    if varname in _vars_datashrunk:
        file_suff = ".data.shrunk"
    #
    iters = xm.mds_store._get_all_iternums(
        data_dir, file_prefixes=varname, file_format="*.??????????" + file_suff
    )
    time = delta_t * np.array(iters)

    iters = xr.DataArray(iters, coords=[time], dims=["time"])
    time = xr.DataArray(time, coords=[iters.values], dims=["iters"])

    return iters, time
