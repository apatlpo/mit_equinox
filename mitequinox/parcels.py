import os, shutil
from glob import glob
import yaml
import pickle

from itertools import product
from tqdm import tqdm

import numpy as np
import xarray as xr
import pandas as pd
from heapq import nsmallest

import geopandas
from shapely.geometry import Polygon, Point
import pyproj
from datetime import timedelta, datetime

from h3 import h3

import dask
import dask.dataframe as dd
from dask.delayed import delayed

from xmitgcm.llcreader import llcmodel as llc

from parcels import FieldSet, ParticleSet, ParticleFile, plotTrajectoriesFile, Variable
from parcels import JITParticle, ScipyParticle
from parcels import ErrorCode, NestedField, AdvectionEE, AdvectionRK4

from .drifters import haversine

# ------------------------------- dir tree management ------------------------------

def create_dir_tree(root_dir, run_name, overwrite=False, verbose=True):
    """ Create run directory tree based on a root_dir and run_name
    Parameters
    ----------
    
    """
    run_root_dir = os.path.join(root_dir, run_name)
    _create_dir(run_root_dir, overwrite, verbose=verbose)
    dirs = {"run_root": run_root_dir}
    #
    for sub_dir in ["run", "tiling", "parquets", "diagnostics"]:
        dirs[sub_dir] = os.path.join(run_root_dir, sub_dir)
        _create_dir(dirs[sub_dir], overwrite, verbose=verbose)
    return dirs

def _create_dir(directory, overwrite, verbose=True):
    if os.path.isdir(directory) and not overwrite:
        if verbose:
            print('Not overwriting {}'.format(directory))
    else:
        shutil.rmtree(directory, ignore_errors=True)
        os.mkdir(directory)    

# ------------------------------- llc tiling -------------------------------------

_mates = [
    ["maskW", "maskS"],
    ["TAUX", "TAUY"],
    ["SSU", "SSV"],
    ["dxC", "dyC"],
    ["dxG", "dyG"],
    ["hFacW", "hFacS"],
    ["rAw", "rAs"],
]

crs_wgs84 = pyproj.CRS("EPSG:4326")

def default_projection(lon, lat):
    # https://proj.org/operations/projections/laea.html
    return "+proj=laea +lat_0={:d} +lon_0={:d} +units=m".format(int(lat), int(lon))


class tiler(object):
    def __init__(
        self,
        ds=None,
        factor=(4, 4),
        overlap=(500, 500),
        tile_dir=None,
        name="tiling",
        N_extra=1000,
    ):
        """ Object handling a custom tiling of global llc data
        Parameters
        ----------
            ds: xr.Dataset, optional
            factor: tuple, optional
                Reduction factor in i and j directions
            overlap: tuple, optional
                Number of points that overlap over each domain
            tile_dir: 
            name: str
                
            N_extra: int
                number of extra points added around the dateline
        """
        if ds is not None:
            self._build(ds, factor, overlap, N_extra)
            self.name = name
        elif tile_dir is not None:
            self._load(tile_dir)
        else:
            assert False, "Either ds or tile_dir are required."
        self.crs_wgs84 = crs_wgs84
        self.N_extra = N_extra

    def _build(self, ds, factor, overlap, N_extra, projection=None):
        """Generate tiling

        Parameters
        ----------
        ds: xarray.Dataset
            contains llc model output
        factor: tuple
            Number of tiles along each dimensions, e.g. (4,4)
        overlap: tuple
            Fraction of overlap in number of points
        N_extra: int
            number of extra points added around the dateline
        """

        if "time" in ds.dims:
            ds = ds.isel(time=0)

        # pair vector variables
        for m in _mates:
            if m[0] in ds:
                ds[m[0]].attrs["mate"] = m[1]
                ds[m[1]].attrs["mate"] = m[0]

        # concatenate face data
        ds = llc.faces_dataset_to_latlon(ds)

        # store initial grid size
        global_domain_size = ds.i.size, ds.j.size

        # add N_extra points along longitude to allow wrapping around dateline
        ds_extra = ds.isel(i=slice(0, N_extra), i_g=slice(0, N_extra))
        for dim in ["i", "i_g"]:
            ds_extra[dim] = ds_extra[dim] + ds[dim][-1] + 1
        ds = xr.merge([xr.concat([ds[v], ds_extra[v]], ds[v].dims[-1]) for v in ds])

        # start tiling
        tiles_1d, boundaries_1d = {}, {}
        tiles_1d["i"], boundaries_1d["i"] = tile_domain(
            global_domain_size[0], factor[0], overlap[0], wrap=True
        )
        tiles_1d["j"], boundaries_1d["j"] = tile_domain(
            global_domain_size[1], factor[1], overlap[1]
        )
        tiles = list(product(tiles_1d["i"], tiles_1d["j"]))
        boundaries = list(product(boundaries_1d["i"], boundaries_1d["j"]))
        N_tiles = len(tiles)

        data = {
            d: [ds.isel(i=s[0], j=s[1], i_g=s[0], j_g=s[1]) for s in slices]
            for d, slices in zip(["tiles", "boundaries"], [tiles, boundaries])
        }
        # tile data used for parcel advections
        # only boundary data will be used for reassignments

        # we need tile centers for geographic projections (dateline issues)
        centers = [
            ds.reset_coords()[["XC", "YC"]]
            .isel(
                i=[int((t[0].start + t[0].stop) / 2)],
                j=[int((t[1].start + t[1].stop) / 2)],
            )
            .squeeze()
            for t in tiles
        ]
        centers = [
            {"lon": float(c.XC), "lat": float(c.YC), "i": int(c.i), "j": int(c.j)}
            for c in centers
        ]

        # generate list of projections
        if not projection:
            projection = default_projection
        crs_strings = [
            projection(c["lon"], c["lat"]) for c in centers
        ]  # for storage purposes
        CRS = list(map(pyproj.CRS, crs_strings))

        # build geometrical boundaries geopandas and shapely objects from boundaries
        S, G = {}, {}
        for key, _D in data.items():

            S[key], G[key] = [], []

            for d, crs in zip(_D, CRS):

                lon = _get_boundary(d["XC"], stride=1)
                lat = _get_boundary(d["YC"], stride=1)

                # generate shapely object
                _polygon = Polygon(list(zip(lon, lat)))
                # geopandas dataframe with projection immediately transformed to deal with dateline issues
                polygon_gdf = geopandas.GeoDataFrame(
                    [1], geometry=[_polygon], crs=crs_wgs84
                ).to_crs(crs)
                # update shapely polygon on new projections
                polygon = polygon_gdf.geometry[0]
                S[key].append(polygon)
                G[key].append(polygon_gdf)

        # store useful data
        V = [
            "global_domain_size",
            "N_tiles",
            "tiles",
            "boundaries",
            "CRS",
            "crs_strings",
            "S",
            "G",
        ]
        for v in V:
            setattr(self, v, eval(v))

    def _load(self, tile_dir):
        """Load tiler from tile_dir"""

        # load various dict
        ds = xr.open_dataset(os.path.join(tile_dir, "info.nc")) 
        self.global_domain_size = (ds.attrs["global_domain_size_0"], 
                                   ds.attrs["global_domain_size_1"])
        self.N_tiles = ds.attrs["N_tiles"]
        
        # regenerate projections
        self.CRS = list(map(pyproj.CRS, list(ds["crs_strings"].values)))

        # rebuild slices (tiles, boundaries)
        D = {}
        for t in ["tiles", "boundaries"]:
            D[t] = [
                (
                    slice(
                        int(ds["i_start_" + t].isel(tile=i)),
                        int(ds["i_end_" + t].isel(tile=i)),
                    ),
                    slice(
                        int(ds["j_start_" + t].isel(tile=i)),
                        int(ds["j_end_" + t].isel(tile=i)),
                    ),
                )
                for i in range(self.N_tiles)
            ]
        self.tiles = D["tiles"]
        self.boundaries = D["boundaries"]

        # rebuild S and G
        S, G = {}, {}
        for key in ["tiles", "boundaries"]:
            S[key], G[key] = [], []
            df = pd.read_csv(os.path.join(tile_dir, key + "_bdata.csv"))
            for i, crs in enumerate(self.CRS):
                polygon = Polygon(
                    list(zip(df["x{:03d}".format(i)], df["y{:03d}".format(i)]))
                )
                polygon_gdf = geopandas.GeoDataFrame([1], geometry=[polygon], crs=crs)
                S[key].append(polygon)
                G[key].append(polygon_gdf)
        self.S = S
        self.G = G

    def store(self, tile_dir, create=True):
        """Store tile to tile_dir"""

        _check_tile_directory(tile_dir, create)

        # tiles and boundaries and scales
        df_tiles = slices_to_dataframe(self.tiles)
        df_boundaries = slices_to_dataframe(self.boundaries)
        df = pd.concat(
            [df_tiles.add_suffix("_tiles"),
             df_boundaries.add_suffix("_boundaries")],
            axis=1,
        )
        ds = (df
              .to_xarray()
              .rename(dict(index="tile"))
             )
        ds["crs_strings"]=("tile", self.crs_strings) # was list
        ds.attrs["global_domain_size_0"] = self.global_domain_size[0]
        ds.attrs["global_domain_size_1"] = self.global_domain_size[1]
        ds.attrs["N_tiles"] = self.N_tiles
        # store in netcdf file
        ds.to_netcdf(os.path.join(tile_dir, "info.nc"), mode="w")
        #
        # boundary data
        for key in ["tiles", "boundaries"]:
            df = pd.concat(
                [
                    polygon_to_dataframe(gdf.geometry[0], suffix="%03d" % i)
                    for i, gdf in enumerate(self.G[key])
                ],
                axis=1,
            )
            df.to_csv(os.path.join(tile_dir, key + "_bdata.csv"))

        print("Tiler stored in {}".format(tile_dir))

    def assign(
        self,
        lon=None,
        lat=None,
        pid=None,
        gs=None,
        inner=True,
        tiles=None,
    ):
        """assign data to tiles
        Parameters
        ----------
        lon, lat: iterables, optional
            longitude and latitudes
        gs: dataframe, optional
            contains long
        inner: boolean
            assigns to inner part of the domain only
        tiles: list, optional
            search in a subset of tiles
        """
        if lon is not None and lat is not None:
            pts = geopandas.points_from_xy(lon, lat)
            gs = geopandas.GeoSeries(pts, crs=crs_wgs84)
        if inner:
            polygons = self.S["boundaries"]
        else:
            polygons = self.S["tiles"]

        if tiles is None:
            tiles = np.arange(self.N_tiles)
        elif isinstance(tiles, list):
            tiles = np.array(tiles)

        df = pd.DataFrame([gs.to_crs(self.CRS[t]).within(polygons[t]) for t in tiles]).T
        
        # purpose of code below: Sylvie?
        dummy = [abs(-35.703821 - lon[i]) < 1.0e-5 for i in range(len(lon))]
        if any(dummy):
            print("assign:3000002", lon[2], lat[2], df[df.index == 2])

        def _find_column(v):
            out = tiles[v]
            if out.size == 0:
                return -1
            else:
                return out[0]

        tiles = df.apply(_find_column, axis=1)
        return tiles

    # assignment is slow, could we improve this?
    # https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon

    def tile(self, ds, tile=None, rechunk=True, persist=False):
        """Load zarr archive and tile

        Parameters
        ----------
        ds: xr.Dataset
            llc dataset that will be tiled
        tile: int, optional
            select one tile, returns a list of datasets otherwise (default)
        rechunk: boolean, optional
            set spatial chunks to -1
        """
        if "face" in ds.dims:
            # pair vector variables
            for m in _mates:
                if m[0] in ds:
                    ds[m[0]].attrs["mate"] = m[1]
                    ds[m[1]].attrs["mate"] = m[0]
            ds = llc.faces_dataset_to_latlon(ds).drop("face")
        if tile is None:
            tile = list(range(self.N_tiles))
        if isinstance(tile, list):
            return [
                self.tile(ds, tile=i, rechunk=rechunk, persist=persist) for i in tile
            ]
        ds = ds.isel(
            i=self.tiles[tile][0],
            j=self.tiles[tile][1],
        )
        #
        _default_rechunk = {"i": -1, "j": -1}
        if rechunk == True:
            ds = ds.chunk(_default_rechunk)
        elif isinstance(rechunk, dict):
            _default_rechunk.update(rechunk)
            ds = ds.chunk(_default_rechunk)
        #
        if persist:
            ds = ds.persist()
        return ds
    
    def create_tile_run_tree(self, run_dir, overwrite=False):
        """ Create a tree of directory for the run
        run_root_dir/run/data_000/ , ...
        """
        dirs = {t: os.path.join(run_dir, "data_{:03d}".format(t))
                for t in range(self.N_tiles)}
        for t, tile_dir in dirs.items():
            _create_dir(tile_dir, overwrite)
        self.run_dirs = dirs
        #return dirs
        
    def clean_up(self, run_dir, step):
        """ Clean up netcdf files older than step
        """
        for t in range(self.N_tiles):
            ncfiles = glob(os.path.join(run_dir, "data_{:03d}/*.nc".format(t)))
            for f in ncfiles:
                f_step = f.split('/')[-1].split('_')[1]
                if int(f_step)>=step:
                    print('delete {}'.format(f))
                    os.remove(f)        


def tile_domain(N, factor, overlap, wrap=False):
    """tile a 1D dimension into factor tiles with some overlap

    Parameters
    ----------
        N: int
            size of the dimension
        factor: int
            number of tiles
        overlap: float
            fraction of overlap
        wrap: boolean, optional
            True means dimension is periodical (default if False)

    Returns
    -------
        slices: list
            List of slices for each tile
        boundaries: list
            List of "boundaries" which are defined as the midway point within overlapped areas
    """
    if wrap:
        _N = N + overlap
    else:
        _N = N
    n = np.ceil((_N + (factor - 1) * overlap) / factor)
    slices = []
    boundaries = []
    for f in range(factor):
        slices.append(
            slice(int(f * (n - overlap)), int(min(f * (n - overlap) + n, _N)))
        )
        # boundaries do need to be exact
        lower = 0
        if f > 0 or wrap:
            lower = int(f * (n - overlap) + 0.5 * overlap)
        upper = N
        if f < factor - 1 or wrap:
            # upper = int(f*(n-overlap)+n-0.5*overlap)
            upper = int(f * (n - overlap) + n - 0.5 * overlap + 1)
        boundaries.append(slice(lower, upper))
    return slices, boundaries


def _get_boundary(da, i="i", j="j", stride=4):
    """return array along boundaries with optional striding"""
    return np.hstack(
        [
            da.isel(**{j: 0}).values[::stride],
            da.isel(**{i: -1}).values[::stride],
            da.isel(**{j: -1}).values[::-stride],
            da.isel(**{i: 0}).values[::-stride],
        ]
    )


def slices_to_dataframe(slices):
    """transform list of slices into dataframe for storage"""
    i_start = [t[0].start for t in slices]
    i_stop = [t[0].stop for t in slices]
    j_start = [t[1].start for t in slices]
    j_stop = [t[1].stop for t in slices]
    return pd.DataFrame(
        map(list, zip(i_start, i_stop, j_start, j_stop)),
        columns=["i_start", "i_end", "j_start", "j_end"],
    )


def polygon_to_dataframe(polygon, suffix=""):
    _coords = polygon.boundary.coords
    return pd.DataFrame(_coords, columns=["x", "y"]).add_suffix(suffix)


def _check_tile_directory(tile_dir, create):
    """Check existence of a directory and create it if necessary"""
    # create diagnostics dir if not present
    if not os.path.isdir(tile_dir):
        if create:
            # need to create the directory
            os.mkdir(tile_dir)
            print("Create new tile directory {}".format(tile_dir))
        else:
            raise OSError("Directory does not exist")


# def tile(tiler, ds, tile=None, rechunk=True):
#    ''' Load zarr archive and tile
#
#    Parameters
#    ----------
#    ds: xr.Dataset
#        llc dataset that will be tiled
#    tile: int, optional
#        select one tile, returns a list of datasets otherwise (default)
#    rechunk: boolean, optional
#        set spatial chunks to -1
#    '''
#    if 'face' in ds.dims:
#        # pair vector variables
#        for m in _mates:
#            if m[0] in ds:
#                ds[m[0]].attrs['mate'] = m[1]
#                ds[m[1]].attrs['mate'] = m[0]
#        ds = llc.faces_dataset_to_latlon(ds).drop('face')
#    ds = ds.isel(i=tiler.tiles[tile][0],
#                 j=tiler.tiles[tile][1],
#                 i_g=self.tiles[tile][0],
#                 j_g=self.tiles[tile][1],
#                )
#    if rechunk:
#        ds = ds.chunk({'i': -1, 'j': -1, 'i_g': -1, 'j_g': -1})
#    return ds


def generate_randomly_located_data(lon=(0, 180), lat=(0, 90), N=1000):
    """Generate randomly located points"""
    # generate random points for testing
    points = [
        Point(_lon, _lat)
        for _lon, _lat in zip(
            np.random.uniform(lon[0], lon[1], (N,)),
            np.random.uniform(lat[0], lat[1], (N,)),
        )
    ]
    return geopandas.GeoSeries(points, crs=crs_wgs84)


def tile_store_llc(ds, time_slice, tl, persist=False, netcdf=False):
    """ 
    """

    # tslice = slice(t, t+dt_window*24, None)
    # ds_tsubset = ds.isel(time=time_slice)
    # extract time slice for dt_window
    ds_tsubset = ds.sel(time=time_slice)
    # convert faces structure to global lat/lon structure
    if "face" in ds_tsubset.dims:
        # pair vector variables
        for m in _mates:
            if m[0] in ds:
                ds_tsubset[m[0]].attrs["mate"] = m[1]
                ds_tsubset[m[1]].attrs["mate"] = m[0]
        ds_tsubset = llc.faces_dataset_to_latlon(ds_tsubset).drop("face")

    # add N_extra points along longitude to allow wrapping around dateline
    ds_extra = ds_tsubset.isel(i=slice(0, tl.N_extra), 
                               i_g=slice(0, tl.N_extra)
                              )
    for dim in ["i", "i_g"]:
        ds_extra[dim] = ds_extra[dim] + ds_tsubset[dim][-1] + 1
    ds_tsubset = xr.merge(
        [
            xr.concat([ds_tsubset[v], ds_extra[v]], ds_tsubset[v].dims[-1])
            for v in ds_tsubset
        ]
    )

    # shift horizontal grid to match parcels (NEMO) convention
    ds_tsubset = fuse_dimensions(ds_tsubset)

    if persist:
        ds_tsubset = ds_tsubset.persist()

    D = tl.tile(ds_tsubset, persist=False)

    ds_tiles = []
    #for tile, ds_tile in enumerate(tqdm(D)):
    for tile, ds_tile in enumerate(D):
        if netcdf:
            nc_file = os.path.join(tl.run_dirs[tile], "llc.nc")
            ds_tile.to_netcdf(nc_file, mode="w")
            ds_tiles.append(None)
        else:
            ds_tiles.append(ds_tile.chunk(chunks={"time": 1}))
    return ds_tiles


# ------------------------------- parcels specific code ----------------------------


def fuse_dimensions(ds, shift=True):
    """rename i_g and j_g into i and j
    Shift horizontal grid such as to match parcels (NEMO) convention
    see https://github.com/OceanParcels/parcels/issues/897
    """
    coords = list(ds.coords)
    for c in ["i_g", "j_g"]:
        coords.remove(c)
    ds = ds.reset_coords()
    D = []
    for v in ds:
        _da = ds[v]
        if shift and "i_g" in ds[v].dims:
            _da = _da.shift(i_g=-1)
        if shift and "j_g" in ds[v].dims:
            _da = _da.shift(j_g=-1)
        _da = _da.rename({d: d[0] for d in _da.dims if d != "time"})
        D.append(_da)
    ds = xr.merge(D)
    ds = ds.set_coords(coords)
    return ds


# should create a class with code below
class run(object):
    
    def __init__(
        self,
        tile,
        tl,
        ds,
        step,
        starttime,
        endtime,
        dt_window,
        dt_step,
        dt_out,
        pclass="jit",
        id_max=None,
        netcdf=False,
        verbose=0,
    ):

        self.empty = True
                
        self.tile = tile
        self.tl = tl
        self.run_dir = tl.run_dirs[tile]
        self._tile_run_dirs = tl.run_dirs
        self.pclass = pclass
        self.step = step

        self.starttime = starttime
        self.endtime = endtime
        
        self.dt_step = dt_step
        self.dt_window = dt_window
        self.dt_out = dt_out
                
        self.id_max = id_max
        
        self.verbose = verbose

        if self.verbose>0:
            print('starttime =', starttime)
            print('endtime =', endtime)
            print('dt_window =', dt_window)
            print('dt_step =', dt_step)
                
        # load fieldset
        self.init_field_set(ds, netcdf)
        # step_window would call such object

        self.particle_class = _get_particle_class(pclass)

    def __getitem__(self, item):
        if self.empty:
            return
        if item in ["lon", "lat", "id"]:
            return self.pset.collection.data[item]
        elif item == "size":
            return self.pset.size

    def init_field_set(self, ds, netcdf):

        self._fieldset_netcdf = netcdf

        variables = {
            "U": "SSU",
            "V": "SSV",
        }
        #    "waterdepth": "Depth",
        _standard_dims = {"lon": "XC", "lat": "YC", "time": "time"}
        dims = {
            "U": _standard_dims,
            "V": _standard_dims,
        }
        #    "waterdepth": {"lon": "XC", "lat": "YC"},
        if self.pclass=='extended':
            variables.update({"sea_level": "Eta",
                             "temperature": "SST",
                             "salinity": "SSS"}
                            )
            dims.update({"sea_level": _standard_dims,
                         "temperature": _standard_dims,
                         "salinity": _standard_dims,
                        })
        if netcdf:
            fieldset = FieldSet.from_netcdf(
                netcdf,
                variables=variables,
                dimensions=dims,
                interp_method="cgrid_velocity",
            )
        else:
            #ds = ds.compute() # test to see if it decreases memory usage, nope
            fieldset = FieldSet.from_xarray_dataset(
                ds,
                variables=variables,
                dimensions=dims,
                interp_method="cgrid_velocity",
                allow_time_extrapolation=True,
            )
        self.fieldset = fieldset

    def init_particles(self, ds, dij, debug=None):
        """Initial particle positions"""
        tile, tl, fieldset = self.tile, self.tl, self.fieldset

        # first step, create drifters positions
        x = (
            ds[["XC", "YC"]]
            .isel(i=slice(0, None, dij), j=slice(0, None, dij))
            .stack(drifter=("i", "j"))
            .reset_coords()
        )
        x = x.where(x.Depth > 0, drop=True)
        # use tile to select points within the tile (most time conssuming operation)
        xv, yv = x.XC.values, x.YC.values
        in_tile = tl.assign(lon=xv, lat=yv, tiles=[tile])
        xv, yv = xv[in_tile[in_tile == tile].index], yv[in_tile[in_tile == tile].index]
        # store initial positions along with relevant surounding area
        # will be store in the first file, just need a mechanism to read them back
        #
        pset = None
        if xv.size > 0:
            self.particle_class.setLastID(0)
            pset = ParticleSet(
                fieldset=fieldset,
                pclass=self.particle_class,
                lon=xv.flatten(),
                lat=yv.flatten(),
            )
            # pid_orig = np.arange(xv.flatten().size)+(tile*100000),
        # offset parcel id's such as to make sure they do not overlap
        if pset is not None:
            if pset.size > 0:
                pset.collection.data["id"] = pset.collection.data["id"] + int(tile * 1e6)           
            # initial value of id_max
            self.id_max = max(pset.collection.data["id"])
            self.empty = False
        else:
            self.id_max = -1

        self.pset = pset

    def init_particles_restart(self, seed=False):
        """Reload data from previous runs"""
        tile, tl, fieldset = self.tile, self.tl, self.fieldset
        
        # load parcel file from previous runs
        self.particle_class.setLastID(0)
        pset = ParticleSet(fieldset=self.fieldset, pclass=self.particle_class)
        for _tile in range(tl.N_tiles):
            ncfile = self.nc(step=self.step - 1, tile=_tile)
            if os.path.isfile(ncfile):
                #particle_class = _get_particle_class(self.pclass)
                particle_class = self.particle_class
                #particle_class.setLastID(0)
                _pset = ParticleSet.from_particlefile(
                    fieldset,
                    particle_class,
                    ncfile,
                    restart=True,
                    restarttime = np.datetime64(self.starttime),
                )
                df = pd.read_csv(self.csv(step=self.step - 1, tile=_tile), index_col=0)
                df_not_in_tile = df.loc[df["tile"] != tile]
                if df_not_in_tile.size > 0:
                    boolind = np.array(
                        [
                            i in df_not_in_tile["id"].values
                            for i in _pset.collection.data["id"]
                        ]
                    )
                    _pset.remove_booleanvector(boolind)
                if _pset.size > 0:
                    pset.add(_pset)

        # could add particle here based on density
        ncfile = self.nc(step=0)
        if seed and os.path.isfile(ncfile):
            # load initial parcel position for this tile and compute parcel separation
            ds = xr.open_dataset(ncfile).isel(obs=0)
            lon_init, lat_init = ds.lon.values, ds.lat.values
            separation_init = [nsmallest(2, haversine(lon, lat, lon_init, lat_init))[1] 
                               for lon, lat in zip(lon_init, lat_init)
                              ]
            lon_last, lat_last = pset.collection.data["lon"], pset.collection.data["lat"]
            seed_positions = [(lon, lat) for lon, lat, s_init in 
                              zip(lon_init, lat_init, separation_init) 
                              if ((lon_last.size==0)
                                  or min(haversine(lon, lat, lon_last, lat_last))>s_init
                                 )
                             ]
            if seed_positions:
                self.particle_class.setLastID(0)
                lon, lat = list(zip(*seed_positions))
                _pset = ParticleSet(
                    fieldset=fieldset,
                    pclass=self.particle_class,
                    lon=lon,
                    lat=lat,
                )
                # update index to prevent any overlap
                _pset.collection.data["id"] = _pset.collection.data["id"] + 1 + self.id_max
                self.id_max = max(_pset.collection.data["id"])
                pset.add(_pset)
        
        if pset.size>0:
            self.empty=False
        
        self.pset = pset
        
    def execute(self, advection=None):

        if self.empty:
            return
        
        pset = self.pset
                
        if advection == "euler":
            adv = AdvectionEE
        else:
            adv = AdvectionRK4
        if self.pclass=="extended":
            kernel = Extended_Sample + pset.Kernel(adv)
        else:
            kernel = adv
        
        # if pset.size>0:
        if pset is not None:
            if pset.size > 0:
                file_out = pset.ParticleFile(self.nc(), 
                                             outputdt=self.dt_out,
                                            )
                pset.execute(
                    pyfunc=kernel,
                    endtime=self.endtime,
                    dt=self.dt_step,
                    recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                    output_file=file_out,
                    postIterationCallbacks=[self.store_to_csv],
                    callbackdt=self.dt_window,
                )
                # could add garbage collection in postIterationCallbacks, see: https://github.com/OceanParcels/parcels/issues/607

                # force converting temporary npy directory to netcdf file
                file_out.export()

    def nc(self, step=None, tile=None):
        if step is None:
            step = self.step
        if tile is None:
            tile = self.tile
        tile_dir = self._tile_run_dirs[tile]
        file = "floats_{:03d}_{:03d}.nc".format(step, tile)
        return os.path.join(tile_dir, file)

    def csv(self, step=None, tile=None):
        if step is None:
            step = self.step
        if tile is None:
            tile = self.tile
        tile_dir = self._tile_run_dirs[tile]
        file = "floats_assigned_{:03d}_{:03d}.csv".format(step, tile)
        return os.path.join(tile_dir, file)

    def store_to_csv(self):
        if self.pset is not None:
            # sort floats per tiles
            float_tiles = self.tl.assign(lon=self["lon"], lat=self["lat"])
            # store to csv
            float_tiles = float_tiles.to_frame(name="tile")
            float_tiles["id"] = self["id"]
            float_tiles = float_tiles.drop_duplicates(subset=["id"])
            float_tiles.to_csv(self.csv())

def _get_particle_class(pclass):
    # if pclass=='llc':
    #    return LLCParticle
    if pclass == "jit":
        return JITParticle
    elif pclass == "scipy":
        # not working at all at the moment
        return ScipyParticle
    elif pclass == "extended":
        return Particle_extended

class Particle_extended(JITParticle):
    zonal_velocity = Variable('zonal_velocity', dtype=np.float32)
    meridional_velocity = Variable('meridional_velocity', dtype=np.float32)
    #
    sea_level = Variable('sea_level', dtype=np.float32)
    temperature = Variable('temperature', dtype=np.float32)
    salinity = Variable('salinity', dtype=np.float32)
    
def Extended_Sample(particle, fieldset, time):
    particle.zonal_velocity, particle.meridional_velocity = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    #particle.zonal_velocity = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    #particle.meridional_velocity = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    #
    particle.sea_level = fieldset.sea_level[time, particle.depth, particle.lat, particle.lon]
    particle.temperature = fieldset.temperature[time, particle.depth, particle.lat, particle.lon]
    particle.salinity = fieldset.salinity[time, particle.depth, particle.lat, particle.lon]

# Make sure to remove the floats that start on land
def DeleteParticle(particle, fieldset, time):
    particle.delete()

def RemoveOnLand(particle, fieldset, time):
    u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    # not working below
    # water_depth = fieldset.waterdepth[particle.depth, particle.lat, particle.lon]
    # if math.fabs(particle.depth) < 500:
    if math.fabs(u) < 1e-12 and math.fabs(v) < 1e-12:
        particle.delete()
        
def step_window(
    tile,
    step,
    starttime,
    endtime,
    dt_window,
    dt_step,
    dt_out,
    tl,
    ds_tile=None,
    init_dij=10,
    parcels_remove_on_land=True,
    pclass="jit",
    id_max=None,
    seed=False,
    verbose=0,
):
    """timestep parcels within one tile (tile) and one time window (step)
    
    Parameters:
    -----------
        tile:
        step:
        starttime:
        endtime:
        dt_window:
        tl:
        run_dir:
        ds_tile:
        init_dij:
        parcels_remove_on_land: boolean, optional
        pclass: str, optional
        verbose: int, optional
        
    """

    # https://docs.dask.org/en/latest/scheduling.html
    # reset dask cluster locally
    dask.config.set(scheduler="threads")

    # directory where tile data is stored
    tile_dir = tl.run_dirs[tile]
    ds = ds_tile
    if ds is None:
        # "load" data either via pointer or via file reading
        # ds = D[tile] #.compute() # compute necessary?
        #
        llc = os.path.join(tile_dir, "llc.nc")
        ds = xr.open_dataset(llc, chunks={"time": 1})

    # init run object
    prun = run(tile, tl, ds, 
               step,
               starttime, endtime, dt_window,
               dt_step, dt_out,
               id_max=id_max,
               verbose=verbose,
               pclass=pclass,
              )
    #if debug==1:
    #    return prun

    # load drifter positions
    if step == 0:
        prun.init_particles(ds, init_dij)
    else:
        prun.init_particles_restart(seed)
    #if debug==2:
    #    return prun

    # if parcels_remove_on_land and prun.pset.size>0:
    if parcels_remove_on_land and prun.pset is not None:
        if prun.pset.size > 0:
            prun.pset.execute(
                RemoveOnLand,
                dt=0,
                recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
            )
    # 2.88 s for 10x10 tiles and dij=10

    # perform the parcels simulation
    prun.execute()
    
    # extract useful information
    if prun.empty:
        parcel_number=0
    else:
        parcel_number = prun.pset.collection.data["id"].size
    id_max = prun.id_max
    
    return parcel_number, id_max

def name_log_file(run_dir):
    now = datetime.now().strftime("%Y%m%d_%H%M")
    return os.path.join(run_dir, 'log_'+now+'.yaml')

def store_log(log_file, step, data):
    """ Store log data in log file
    Parameters:
    -----------
        log_file: str
        step: int
        data: dict
    """
    if os.path.isfile(log_file):
        with open(log_file,'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)
        cur_yaml.update({step: data})
    else:
        cur_yaml = {step: data}
    with open(log_file,'w') as yamlfile:
        yaml.safe_dump(cur_yaml, yamlfile)

def browse_log_files(run_dir):
    log_files = sorted(glob(os.path.join(run_dir, 'log_*yaml')))
    log = {}
    if log_files:
        for f in log_files:
            with open(f,'r') as yamlfile:
                log.update(yaml.safe_load(yamlfile))
    return log

def load_logs(root_dir, run_name):
    """ Browse run and load basic information (global float number, etc ...)
    Parameters
    
    
    """
    dirs = create_dir_tree(root_dir, run_name, 
                           overwrite=False, verbose=False
                          )
    tl = tiler(tile_dir=dirs["tiling"])
    log = browse_log_files(dirs['run'])
    steps = list(log)
    # massage data
    da0 = xr.DataArray([d['global_parcel_number'] for step, d in log.items()], 
                   dims="step", name="global_parcel_number"
                  ).assign_coords(step=steps)
    #
    _D = []
    for step, d in log.items():
        _D.append(xr.DataArray(list(d['max_ids'].values()), 
                               dims="tile", 
                               name='id_max'),
                 )
    da1 = xr.concat(_D, 'step').assign_coords(step=steps)
    #
    _D = []
    #_S = []
    for step, d in log.items():
        #if 'local_numbers' in d:
        _D.append(xr.DataArray(list(d['local_numbers'].values()), 
                               dims="tile",
                               name='local_numbers'),
                 )
        #_S.append(step)
    #da2 = xr.concat(_D, 'step').assign_coords(step=_S)
    da2 = xr.concat(_D, 'step').assign_coords(step=steps)
    #
    ds = xr.merge([da0, da1, da2])
    return ds, dirs


# ------------------------------ netcdf floats relative ---------------------------------------


def load_nc(
    run_dir,
    step_tile="*",
    index="trajectory",
    persist=False,
):
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
        ds = xr.open_dataset(file)
        # fix trajectory type
        ds["trajectory"] = ds["trajectory"].astype("int32")
        #
        return ds.to_dataframe().set_index(index)

    # find list of tile directories
    tile_run_dirs = glob(run_dir + "/data_*")

    # find list of netcdf float files
    float_files = []
    for _dir in tile_run_dirs:
        float_files.extend(sorted(glob(_dir + "/floats_" + step_tile + ".nc")))

    # read the netcdf files and store in a dask dataframe
    lazy_dataframes = [delayed(xr2df)(f) for f in float_files]
    _df = lazy_dataframes[0].compute()
    df = dd.from_delayed(lazy_dataframes, meta=_df).repartition(partition_size="100MB")
    if persist:
        df = df.persist()
    return df


# ------------------------------ parquet relative ---------------------------------------


def store_parquet(
    parquet_dir,
    df,
    partition_size="100MB",
    index=None,
    overwrite=False,
    engine="auto",
    compression="ZSTD",
    name=None,
):
    """store data under parquet format

    Parameters
    ----------
        parquet_dir: str
            path to directory where parquet files will be stored
        df: dask.dataframe
            to be stored
        partition_size: str, optional
            size of each partition that will be enforced
            Default is '100MB' which is dask recommended size
        index: str, optional
            which index to set before storing the dataframe
        name: str, optional
            name of the parquet archive on disk
        overwrite: bool, optional
            can overwrite or not an existing archive
        engine: str
            engine to store the parquet format
        compression: str
            type of compression to use when storing in parquet format
    """

    # check if right value for index
    columns_names = df.columns.tolist() + [df.index.name]
    if index is None:
        print('No reindexing')
    elif index not in columns_names:
        print("Index must be in ", columns_names)
        return

    if name is None:
        if index is not None:
            name = index
        else:
            print('index or name needs to be provided')
            return
    parquet_path = os.path.join(parquet_dir, name)

    # check wether an archive already exists
    if os.path.isdir(parquet_path):
        if overwrite:
            print("deleting existing archive: {}".format(parquet_path))
            shutil.rmtree(parquet_path)
        else:
            print("Archive already existing: {}".format(parquet_path))
            return

    # create archive path
    os.system("mkdir -p %s" % parquet_path)
    print("create new archive: {}".format(parquet_path))

    # change index of dataframe
    if index is not None and df.index.name != index:
        df = df.reset_index()
        df = df.set_index(index).persist()

    # repartition such that each partition is 100MB big
    if partition_size is not None:
        df = df.repartition(partition_size=partition_size)
    
    df.to_parquet(parquet_path, engine=engine, compression=compression)
    
    return parquet_path


def load_parquet(
    run_dir,
    index="trajectory",
    persist=False,
):
    """load data into a dask dataframe

    Parameters
    ----------
        run_dir: str, path to the simulation (containing the drifters directory)
        index: str, to set the path and load a dataframe with the right index
    """
    parquet_path = os.path.join(run_dir, "drifters", index)

    # test if parquet
    if os.path.isdir(parquet_path):
        return dd.read_parquet(parquet_path, engine="fastparquet")
    else:
        print("load_parquet error: directory not found ", parquet_path)
        return None


# ------------------------------ post-processing ------------------------------------


class parcels_output(object):
    
    def __init__(self, 
                 run_dir, 
                 parquets=None,
                 persist=False,
                ):
        """ Load parcels distributed run
        """
        self.run_dir = run_dir
        # explore tree and load relevant data
        tile_dir = os.path.join(run_dir, 'tiling/')
        if os.path.isdir(tile_dir):
            self.tiler = tiler(tile_dir=tile_dir)
        parquet_dir = os.path.join(run_dir, 'parquets/')
        if os.path.isdir(parquet_dir):
            parquet_paths = glob(parquet_dir+'*')
            self.parquets = {p.split('/')[-1]: p for p in parquet_paths}
        else:
            print('Not parquet files found, may need to produce them, see parcel_distributed.ipynb')
        self.df = {}
        if parquets is not None:
            for p in parquets:
                assert p in self.parquets, '{} parquet file does not exist'.format(p)
                _df = dd.read_parquet(self.parquets[p], engine="fastparquet")
                if persist:
                    _df = _df.persist()
                self.df[p] = _df
        #
        self.diagnostic_dir = os.path.join(self.run_dir,'diagnostics')

    def __getitem__(self, item):
        return self.df[item]
        
    ### store/load diagnostics
    def store_diagnostic(self, name, data, 
                         overwrite=False,
                         directory='diagnostics/',
                         **kwargs
                        ):
        """ Write diagnostics to disk
        
        Parameters
        ----------
        name: str
            Name of a diagnostics to store on disk
        data: dd.DataFrame (other should be implemented)
            Data to be stored
        overwrite: boolean, optional
            Overwrite an existing diagnostic. Default is False
        directory: str, optional
            Directory where diagnostics will be stored (absolute or relative to output directory).
            Default is 'diagnostics/'
        **kwargs:
            Any keyword arguments that will be passed to the file writer
        """
        # create diagnostics dir if not present
        _dir = _check_diagnostic_directory(directory, 
                                           self.run_dir, 
                                           create=True,
                                          )
        #
        if isinstance(data, dd.DataFrame):
            # store to parquet
            store_parquet(_dir, 
                          data,
                          overwrite=overwrite,
                          name=name,
                          **kwargs
                         )
        if isinstance(data, xr.Dataset):
            # store to zarr format
            zarr_path = os.path.join(_dir,name+'.zarr')
            write_kwargs = dict(kwargs)
            if overwrite:
                write_kwargs.update({'mode': 'w'})
            data = _move_singletons_as_attrs(data)
            data = _reset_chunk_encoding(data)
            data.to_zarr(zarr_path, **write_kwargs)            
            print('{} diagnostics stored in {}'.format(name, zarr_path))

    def load_diagnostic(self, name, 
                        directory='diagnostics/',
                        persist=False,
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
        _dir = _check_diagnostic_directory(directory, self.run_dir)
        data_path = os.path.join(_dir, name)
        # find the diagnostic file
        if os.path.isdir(data_path):
            if '.' not in name:
                # parquet archive
                data = dd.read_parquet(data_path, engine="auto")
            elif '.zarr' in name:
                data = xr.open_zarr(data_path)
                # zarr archive
            if persist:
                data = data.persist()
            return data  
        else:
            print('{} does not exist'.parquet_path)
  
                
def _check_diagnostic_directory(directory, dirname, 
                                create=False):
    """ Check existence of a directory and create it if necessary
    """
    # create diagnostics dir if not present
    if os.path.isdir(directory):
        # directory is an absolute path
        _dir = directory
    elif os.path.isdir(os.path.join(dirname, directory)):
        # directory is relative
        _dir = os.path.join(dirname, directory)
    else:
        if create:
            # need to create the directory
            _dir = os.path.join(dirname, directory)
            os.mkdir(_dir)
            print('Create new diagnostic directory {}'.format(_dir))
        else:
            raise OSError('Directory does not exist')
    return _dir                
                
def _move_singletons_as_attrs(ds, ignore=[]):
    """ change singleton variables and coords to attrs
    This seems to be required for zarr archiving
    """
    for c,co in ds.coords.items():
        if co.size==1 and ( len(co.dims)==1 and co.dims[0] not in ignore or len(co.dims)==0 ):
            ds = ds.drop_vars(c).assign_attrs({c: ds[c].values})
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

# ------------------------------ h3 relative ---------------------------------------


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
    df["hex_id"] = df.apply(
        get_hex, axis=1, args=(resolution,), meta="string"
    )  # use 'category' instead?
    return df


def add_lonlat(df, reset_index=False):
    if reset_index:
        df = df.reset_index()
    df["lat"] = df["hex_id"].apply(lambda x: h3.h3_to_geo(x)[0])
    df["lon"] = df["hex_id"].apply(lambda x: h3.h3_to_geo(x)[1] % 360)
    return df


def id_to_bdy(hex_id):
    hex_boundary = h3.h3_to_geo_boundary(hex_id)  # array of arrays of [lat, lng]
    hex_boundary = hex_boundary + [hex_boundary[0]]
    return [[h[1], h[0]] for h in hex_boundary]


def plot_h3_simple(
    df,
    metric_col,
    x="lon",
    y="lat",
    marker="o",
    alpha=1,
    figsize=(16, 12),
    colormap="viridis",
):
    df.plot.scatter(
        x=x,
        y=y,
        c=metric_col,
        title=metric_col,
        edgecolors="none",
        colormap=colormap,
        marker=marker,
        alpha=alpha,
        figsize=figsize,
    )
    plt.xticks([], [])
    plt.yticks([], [])
