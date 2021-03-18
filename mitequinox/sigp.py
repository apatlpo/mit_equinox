import os
from shutil import rmtree

import numpy as np
from scipy import signal
import dask
import xarray as xr
import threading

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cmocean import cm

import datetime, dateutil

# import xmitgcm as xm
from rechunker import rechunk

from .utils import *

# ---------------------------------- rechunk --------------------------------------------


def transpose_rechunk(
    ds,
    chunks,
    iters,
    face=None,
    subsampling=None,
    name=None,
    out_dir=work_data_dir + "rechunked/",
    overwrite=True,
    max_mem="25GB",
    verbose=0,
    debug=False,
):
    """Rechunk a data array

    cluster = PBSCluster(cores=2, processes=1, walltime='03:00:00')

    Parameters:
        ds: xarray.DataArray
            input data array
        chunks: tuple
            (Nt, Nj, Ni)
        iters: xarray.DataArray
            mitgcm iterations to consider
        face: int, optional
            face to consider
        out_dir: str, optional
            output path
        max_mem: str, optional
            rechunker parameter
        verbose: turn on/off verbose

    """

    if name is None:
        vnames = list(ds)
        assert len(vnames) == 1, "You should have only one variablie in the xr dataset"
        v = vnames[0]

    if face is not None:
        ds = ds.sel(face=face)
        suff = "_f{:02d}.zarr".format(int(face))
        print(" face={}".format(int(face)))
    else:
        suff = ".zarr"

    # rechunker outputs
    target_store = out_dir + v + suff
    temp_store = out_dir + "tmp.zarr"

    # clean archives if necessary
    rmtree(temp_store, ignore_errors=True)
    if os.path.isdir(target_store):
        if overwrite:
            rmtree(target_store)
        else:
            print("Do not overwrite")
            # assert False, 'Archive exists and you do not want to overwrite'
            return

    # select common time line
    t0 = ds["time"].where(ds.iters == iters[0], drop=True).values[0]
    t1 = ds["time"].where(ds.iters == iters[-1], drop=True).values[0]
    ds = ds.sel(time=slice(t0, t1))
    ds["dtime"] = ds["dtime"].compute()
    ds["iters"] = ds["iters"].compute()

    if subsampling is not None:
        i_dim, j_dim = get_ij_dims(ds[v])
        ds = ds.isel(
            **{
                i_dim: slice(0, None, subsampling),
                j_dim: slice(0, None, subsampling),
            }
        )

    # deal with the time dimension
    Nt = len(ds.time) - 1 if chunks[0] == 0 else chunks[0]
    chunks = (Nt, chunks[1], chunks[2])
    # -1 is to obtain 8784 which you can divide by 4**2
    # necessary ? yes
    ds = ds.isel(time=slice(len(ds.time) // Nt * Nt))

    # init rechunker
    target_chunks = get_chunks(chunks, v, 1, verbose=verbose)
    r = rechunk(ds, target_chunks, max_mem, target_store, temp_store=temp_store)

    if verbose > 0:
        print_rechunk(r, v)

    # exec
    if debug:
        return r
    result = r.execute()

    # clean up intermediate file
    rmtree(temp_store, ignore_errors=True)

    print(" rechunking over")


def get_chunks(N, v, power=1, verbose=0):
    """get target chunks that may be ingested into rechunker
    Parameters:
        N: tuple
            (Nt, Nj, Ni)
        v: str
            variable name
        power: int
            rescaling power: N = (int(N[0]/4**power), N[1]*2**power, N[2]*2**power)
    """

    N = (int(N[0] / 4 ** power), N[1] * 2 ** power, N[2] * 2 ** power)

    chunks = dict(time=N[0], face=1)
    if v == "SSU":
        chunks = dict(**chunks, i_g=N[1], j=N[2])
    elif v == "SSV":
        chunks = dict(**chunks, i=N[1], j_g=N[2])
    else:
        chunks = dict(**chunks, i=N[1], j=N[2])

    target_chunks = dict(
        **{v: chunks},
        **{c: None for c in chunks},
        dtime=(-1,),
        iters=(-1,),
    )
    size_mb = np.prod(list(chunks.values())) * 4 / 1e6

    if verbose > 0:
        print(chunks)
        print("Individual chunk size = {:.1f} MB".format(size_mb))

    return target_chunks


def print_rechunk(r, v):
    """
    Parameters:
        r: rechunker instance
        v: variable name
    """

    s = r._source[v]
    i = r._intermediate[v]
    t = r._target[v]

    # source size
    print(
        "Source data size: \t\t "
        + "x".join("{}".format(v) for v in s.shape)
        + " \t {:.1f}GB".format(s.nbytes / 1e9)
    )

    print(
        "Source chunk size: \t\t "
        + "x".join("{}".format(v) for v in s.data.chunksize)
        + "\t\t {:.1f}MB".format(np.prod(s.data.chunksize) * 4 / 1e6)
    )

    print(
        "Source number of files: \t\t {:d}".format(
            int(np.prod(s.shape) / np.prod(s.data.chunksize))
        )
    )

    #
    print(
        "Intermediate chunk size: \t "
        + "x".join("{}".format(v) for v in i.chunks)
        + "\t\t {:.1f}MB".format(np.prod(i.chunks) * 4 / 1e6)
    )
    print(
        "Intermediate number of files: \t\t {:d}".format(
            int(np.prod(s.shape) / np.prod(i.chunks))
        )
    )

    #
    print(
        "Target chunk size: \t\t "
        + "x".join("{}".format(v) for v in t.chunks)
        + " \t\t {:.1f}MB".format(np.prod(t.chunks) * 4 / 1e6)
    )
    print(
        "Target number of files: \t\t {:d}".format(
            int(np.prod(s.shape) / np.prod(t.chunks))
        )
    )


# ------------------------------ temporal filters ---------------------------------------


def gen_filter(band, numtaps=24 * 10, dt=1.0, lat=None, domega=None, ndomega=None):
    """Wrapper around scipy.signal.firwing
    dt: float
        hours
    """
    params = {}
    pass_zero = False
    if band == "semidiurnal":
        omega = 1 / 12.0
    elif band == "diurnal":
        omega = 1 / 24.0
    elif band == "inertial":
        try:
            omega = coriolis(lat) * 3600 / 2.0 / np.pi
        except:
            print("latitude needs to be provided to gen_filter")
    #
    if domega is not None:
        cutoff = [omega - domega, omega + domega]
    elif ndomega is not None:
        cutoff = [omega * (1 - ndomega), omega * (1.0 + ndomega)]
    elif band != "subdiurnal":
        print("domega or ndomega needs to be provided to gen_filter")
    #
    if band == "subdiurnal":
        pass_zero = True
        cutoff = [1.0 / 30.0]
    h = signal.firwin(
        numtaps, cutoff=cutoff, pass_zero=pass_zero, nyq=1.0 / 2 / dt, scale=True
    )
    return h


def filter_response(h, dt=1.0):
    """Returns the frequency response"""
    w, hh = signal.freqz(h, worN=8000)
    return hh, (w / np.pi) / 2 / dt * 24


# ------------------------------ spectrum ---------------------------------------

# [psi,lambda] = sleptap(size(uv2,1),2,1);
#% calculate all spectra, linearly detrend
# tic;[f,spp2,snn2] = mspec(1/24,detrend(uv2,1),psi);toc
# [PSI,LAMBDA]=SLEPTAP(N,P,K) calculates the K lowest-order Slepian
#    tapers PSI of length N and time-bandwidth product P, together with
#    their eigenvalues LAMBDA. PSI is N x K and LAMBDA is K x 1.

# scipy.signal.windows.dpss(M, NW, Kmax=None, sym=True, norm=None, return_ratios=False)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.dpss.html#scipy.signal.windows.dpss


def _get_E(x, ufunc=True, **kwargs):
    ax = -1 if ufunc else 0
    #
    dkwargs = {
        "window": "hann",
        "return_onesided": False,
        "detrend": None,
        "scaling": "density",
    }
    dkwargs.update(kwargs)
    f, E = signal.welch(x, fs=24.0, axis=ax, **dkwargs)
    #
    if ufunc:
        return E
    else:
        return f, E


def get_E(v, f=None, **kwargs):
    v = v.chunk({"time": len(v.time)})
    if "nperseg" in kwargs:
        Nb = kwargs["nperseg"]
    else:
        Nb = 60 * 24
        kwargs["nperseg"] = Nb
    if "return_onesided" in kwargs and kwargs["return_onesided"]:
        Nb = int(Nb/2)+1
    if f is None:
        f, E = _get_E(v.values, ufunc=False, **kwargs)
        return f, E
    else:
        E = xr.apply_ufunc(
            _get_E,
            v,
            dask="parallelized",
            output_dtypes=[np.float64],
            input_core_dims=[["time"]],
            output_core_dims=[["freq_time"]],
            dask_gufunc_kwargs={"output_sizes": {"freq_time": Nb}},
            kwargs=kwargs,
        )
        E = E.assign_coords(freq_time=f).sortby("freq_time")
        return E


# ------------------------------ misc ---------------------------------------


def getsize(dir_path):
    """Returns the size of a directory in bytes"""
    process = os.popen("du -s " + dir_path)
    size = int(process.read().split()[0])  # du returns kb
    process.close()
    return size * 1e3
