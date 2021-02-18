import os, shutil
from os.path import join
import gc
import os.path as path
import pickle
from itertools import product
from tqdm import tqdm
from glob import glob

import numpy as np
import xarray as xr
import pandas as pd

import geopandas
from shapely.geometry import Polygon, Point
import pyproj
from datetime import timedelta, datetime

from h3 import h3

import dask
import dask.dataframe as dd
from dask.delayed import delayed

# from matplotlib import pyplot as plt

from xmitgcm.llcreader import llcmodel as llc

from parcels import FieldSet, ParticleSet, ParticleFile, plotTrajectoriesFile, Variable
from parcels import JITParticle, ScipyParticle
from parcels import ErrorCode, NestedField, AdvectionEE, AdvectionRK4

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
            Fraction of overlap
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
        with open(join(tile_dir, "various.p"), "rb") as f:
            various = pickle.load(f)
        self.global_domain_size = various["global_domain_size"]
        self.N_tiles = various["N_tiles"]

        # regenerate projections
        self.CRS = list(map(pyproj.CRS, various["crs_strings"]))

        # rebuild slices (tiles, boundaries)
        D = {}
        for t in ["tiles", "boundaries"]:
            D[t] = [
                (
                    slice(
                        various["slices"]["i_start_" + t].loc[i],
                        various["slices"]["i_end_" + t].loc[i],
                    ),
                    slice(
                        various["slices"]["j_start_" + t].loc[i],
                        various["slices"]["j_end_" + t].loc[i],
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
            df = pd.read_csv(join(tile_dir, key + "_bdata.csv"))
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
            [df_tiles.add_suffix("_tiles"), df_boundaries.add_suffix("_boundaries")],
            axis=1,
        )
        # add crs strings
        # df = df.assign(crs=self.crs_strings)
        # header
        # header = ['{}={}'.format(v) for v in [self.]]
        # df.to_csv(join(tile_dir, 'slices_crs_scalars.csv'))
        various = {
            "slices": df,
            "crs_strings": self.crs_strings,
            "global_domain_size": self.global_domain_size,
            "N_tiles": self.N_tiles,
        }
        with open(join(tile_dir, "various.p"), "wb") as f:
            pickle.dump(various, f)

        # boundary data
        for key in ["tiles", "boundaries"]:
            df = pd.concat(
                [
                    polygon_to_dataframe(gdf.geometry[0], suffix="%03d" % i)
                    for i, gdf in enumerate(self.G[key])
                ],
                axis=1,
            )
            df.to_csv(join(tile_dir, key + "_bdata.csv"))

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
    if not path.isdir(tile_dir):
        if create:
            # need to create the directory
            os.mkdir(tile_dir)
            print("Create new diagnostic directory {}".format(tile_dir))
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


def tile_store_llc(ds, time_slice, tl, tile_data_dirs, persist=False, netcdf=False):

    # tslice = slice(t, t+dt_windows*24, None)
    # ds_tsubset = ds.isel(time=time_slice)
    # extract time slice for dt_windows
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
    ds_extra = ds_tsubset.isel(i=slice(0, tl.N_extra), i_g=slice(0, tl.N_extra))
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
    # rechunk={'time': 2}

    ds_tiles = []
    for tile, ds_tile in enumerate(tqdm(D)):
        # i_g -> i, j->j_g and shift
        # ds_tile = fuse_dimensions(ds_tile)
        #
        if netcdf:
            nc_file = join(tile_data_dirs[tile], "llc.nc")
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
        tile_data_dirs,
        ds,
        step,
        starttime,
        endtime,
        dt_windows,
        pclass="jit",
        netcdf=False,
    ):

        self.tile = tile
        self.tl = tl
        self.run_dir = tile_data_dirs[tile]
        self._tile_dirs = tile_data_dirs
        self.pclass = pclass
        self.step = step
        self.starttime = starttime
        self.endtime = endtime
        self.dt_windows = dt_windows

        # load fieldset
        self.init_field_set(ds, netcdf)
        # step_window would call such object

        self.particle_class = _get_particle_class(pclass)

    def __getitem__(self, item):
        if item in ["lon", "lat", "id"]:
            return self.pset.particle_data[item]
        elif item == "size":
            return self.pset.size

    def init_field_set(self, ds, netcdf):

        self._fieldset_netcdf = netcdf

        variables = {
            "U": "SSU",
            "V": "SSV",
            "waterdepth": "Depth",
        }
        dims = {
            "U": {"lon": "XC", "lat": "YC", "time": "time"},
            "V": {"lon": "XC", "lat": "YC", "time": "time"},
            "waterdepth": {"lon": "XC", "lat": "YC"},
        }
        if netcdf == False:
            fieldset = FieldSet.from_xarray_dataset(
                ds,
                variables=variables,
                dimensions=dims,
                interp_method="cgrid_velocity",
                allow_time_extrapolation=True,
            )
        else:
            fieldset = FieldSet.from_netcdf(
                netcdf,
                variables=variables,
                dimensions=dims,
                interp_method="cgrid_velocity",
            )
        self.fieldset = fieldset

    def init_particles_t0(self, ds, dij):
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
        xv, yv = x.XC.values, x.YC.values
        # use tile to select points within the tile (most time conssuming operation)
        in_tile = tl.assign(lon=xv, lat=yv, tiles=[tile])
        xv, yv = xv[in_tile[in_tile == tile].index], yv[in_tile[in_tile == tile].index]
        #
        pset = None
        if xv.size > 0:
            self.particle_class.setLastID(0)
            pset = ParticleSet(
                fieldset=fieldset,
                pclass=self.particle_class,
                lon=xv.flatten(),
                lat=yv.flatten(),  # ** add index such as
                # pid_orig = np.arange(xv.flatten().size)+(tile*100000),
            )

        if pset is not None:
            if pset.size > 0:
                pset.particle_data["id"] = pset.particle_data["id"] + int(tile * 1e6)
        self.pset = pset
        
        if 3000002.0 in pset.particle_data["id"]:
            index = np.argmin(np.abs(np.array(pset.particle_data["id"]) - 3000002.0))
            print(
                "init_particle_t0:",
                tile,
                pset.particle_data["id"][index],
                pset.particle_data["lon"][index],
                pset.particle_data["lat"][index],
            )
        del pset

    def init_particles_restart(self, step):
        """reload data from previous runs"""
        tile, tl, fieldset = self.tile, self.tl, self.fieldset

        # load parcel file from previous runs

        self.particle_class.setLastID(0)
        pset = ParticleSet(fieldset=self.fieldset, pclass=self.particle_class)
        for _tile in range(tl.N_tiles):
            ncfile = self.nc(step - 1, _tile)
            if os.path.isfile(ncfile):
                particle_class = _get_particle_class(self.pclass)
                particle_class.setLastID(0)
                _pset = ParticleSet.from_particlefile(
                    fieldset,
                    pclass=particle_class,
                    filename=ncfile,
                    restarttime=self.starttime,
                )

                df = pd.read_csv(self.csv(step - 1, tile=_tile), index_col=0)
                df_not_in_tile = df.loc[df["tile"] != tile]
                if df_not_in_tile.size > 0:
                    boolind = np.array(
                        [
                            _pset.particle_data["id"][i] in df_not_in_tile["id"].values
                            for i in range(_pset.size)
                        ]
                    )
                    # _pset.remove_indices(list(df_not_in_tile.index))
                    _pset.remove_booleanvector(boolind)
                    del boolind
                if _pset.size > 0:
                    pset.add(_pset)
                del df
                del df_not_in_tile
                del _pset

        # could add particle here based on density
        
                
        self.pset = pset
        if 3000002.0 in pset.particle_data["id"]:
            index = np.argmin(np.abs(np.array(pset.particle_data["id"]) - 3000002.0))
            print(
                "init_particle_restart:",
                tile,
                pset.particle_data["id"][index],
                pset.particle_data["lon"][index],
                pset.particle_data["lat"][index],
            )
        del pset

    # def execute(self, T, step,
    def execute(
        self,
        endtime,
        step,
        dt_step=timedelta(hours=1.0),
        dt_out=timedelta(hours=1.0),
        advection="euler",
    ):

        if advection == "euler":
            adv = AdvectionEE
        else:
            adv = AdvectionRK4

        pset = self.pset

        # if pset.size>0:
        if pset is not None:
            if pset.size > 0:
                file_out = pset.ParticleFile(self.nc(step), outputdt=dt_out)
                pset.execute(
                    adv,
                    endtime=endtime,
                    dt=dt_step,
                    recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                    output_file=file_out,
                    postIterationCallbacks=[self.SaveCsv],
                    callbackdt=self.dt_windows,
                )

                # force converting temporary npy directory to netcdf file
                file_out.export()

    def nc(self, step, tile=None):
        if tile is None:
            tile = self.tile
        tile_dir = self._tile_dirs[tile]
        file = "floats_{:03d}_{:03d}.nc".format(step, tile)
        return join(tile_dir, file)

    def csv(self, step, tile=None):
        if tile is None:
            tile = self.tile
        tile_dir = self._tile_dirs[tile]
        file = "floats_assigned_{:03d}_{:03d}.csv".format(step, tile)
        return join(tile_dir, file)

    def SaveCsv(self):
        if self.pset is not None:
            # sort floats per tiles
            float_tiles = self.tl.assign(lon=self["lon"], lat=self["lat"])
            # store to csv
            float_tiles = float_tiles.to_frame(name="tile")
            float_tiles["id"] = self["id"]
            float_tiles = float_tiles.drop_duplicates(subset=["id"])
            float_tiles.to_csv(self.csv(self.step))
            del float_tiles
        if 3000002.0 in self.pset.particle_data["id"]:
            index = np.argmin(
                np.abs(np.array(self.pset.particle_data["id"]) - 3000002.0)
            )
            print(
                "SaveCsv:",
                self.tile,
                self.pset.particle_data["id"][index],
                self.pset.particle_data["lon"][index],
                self.pset.particle_data["lat"][index],
            )


def _get_particle_class(pclass):
    # if pclass=='llc':
    #    return LLCParticle
    if pclass == "jit":
        return JITParticle
    elif pclass == "scipy":
        # not working at all at the moment
        return ScipyParticle


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
    dt_windows,
    tl,
    run_dir,
    ds_tile=None,
    init_dij=10,
    parcels_remove_on_land=True,
    pclass="jit",
):
    """timestep parcels within one tile (tile) and one time window (step)
    
    Parameters:
    -----------
        tile:
        step:
        starttime:
        endtime:
        dt_windows:
        tl:
        run_dir:
        ds_tile:
        init_dij:
        parcels_remove_on_land: boolean, optional
        pclass: str, optional
        
    """

    # https://docs.dask.org/en/latest/scheduling.html
    # reset dask cluster locally
    dask.config.set(scheduler="threads")
    # from dask.distributed import Client
    # client = Client()

    # directory where tile data is stored
    tile_data_dirs = [
        join(run_dir, "data_{:03d}".format(t)) for t in range(tl.N_tiles)
    ]
    tile_dir = tile_data_dirs[tile]
    ds = ds_tile
    if ds is None:
        # "load" data either via pointer or via file reading
        # ds = D[tile] #.compute() # compute necessary?
        #
        llc = join(tile_dir, "llc.nc")
        ds = xr.open_dataset(llc, chunks={"time": 1})

    # init run object
    prun = run(tile, tl, tile_data_dirs, ds, step, starttime, endtime, dt_windows)

    # load drifter positions
    if step == 0:
        prun.init_particles_t0(ds, init_dij)
    else:
        prun.init_particles_restart(step)

    # if 3000002.0 in prun.pset.particle_data['id']:
    if tile == 23 or tile == 3:
        index = np.argmin(np.abs(np.array(prun.pset.particle_data["id"]) - 3000002.0))
        print(
            "step_window before land:tile,id,lon,lat,depth=",
            tile,
            prun.pset.particle_data["id"][index],
            prun.pset.particle_data["lon"][index],
            prun.pset.particle_data["lat"][index],
            prun.pset.particle_data["depth"][index],
        )
        print("step_window before land:", ds["XC"].values.min(), ds["XC"].values.max())
        print("step_window before land:", ds["YC"].values.min(), ds["YC"].values.max())
        print("step_window before land:size=", prun.pset.particle_data["id"].size)
    # ** to do: make sure we are not loosing particles

    # if parcels_remove_on_land and prun.pset.size>0:
    if parcels_remove_on_land and prun.pset is not None:
        if prun.pset.size > 0:
            prun.pset.execute(
                RemoveOnLand,
                dt=0,
                recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
            )
    # 2.88 s for 10x10 tiles and dij=10

    if tile == 23 or tile == 3:
        index = np.argmin(np.abs(np.array(prun.pset.particle_data["id"]) - 3000002.0))
        print(
            "step_window after land:",
            tile,
            prun.pset.particle_data["id"][index],
            prun.pset.particle_data["lon"][index],
            prun.pset.particle_data["lat"][index],
        )
        print("step_window after land:", prun.pset.particle_data["id"].size)

    # perform the parcels simulation
    # ** try AdvectionRK4 instead of AdvectionEE
    # prun.execute(dt_windows, step)
    prun.execute(endtime, step)
    # if 3000002.0 in prun.pset.particle_data['id']:
    if tile == 23 or tile == 3:
        index = np.argmin(np.abs(np.array(prun.pset.particle_data["id"]) - 3000002.0))
        print(
            "step_window after execute:",
            tile,
            prun.pset.particle_data["id"][index],
            prun.pset.particle_data["lon"][index],
            prun.pset.particle_data["lat"][index],
        )
        print("step_window after execute:", prun.pset.particle_data["id"].size)
    # prun.execute(dt_windows, step, advection='RK4')

    # assign to tiles and store
    # if prun.pset is not None:
    #    # sort floats per tiles
    #    float_tiles = tl.assign(lon=prun['lon'], lat=prun['lat'])
    #    # store to csv
    #    float_tiles = float_tiles.to_frame(name='tile')
    #    float_tiles['id'] = prun['id']
    #    float_tiles = float_tiles.drop_duplicates(subset=['id'])
    #    float_tiles.to_csv(prun.csv(step))
    #    del float_tiles
    del ds
    del prun
    # gc.collect()
    return


# ------------------------------ netcdf floats relative ---------------------------------------


def load_cdf(
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
    # tile_dir = join(run_dir,'tiling/')
    # tl = tiler(tile_dir=tile_dir)
    # tile_data_dirs = [join(run_dir,'data_{:03d}'.format(t))
    #                  for t in range(tl.N_tiles)
    #                 ]
    tile_data_dirs = glob(run_dir + "/data_*")

    # find list of netcdf float files
    float_files = []
    for _dir in tile_data_dirs:
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
    run_dir,
    df,
    partition_size="100MB",
    index=None,
    overwrite=False,
    sub_dir=None,
    engine="auto",
    compression="ZSTD",
    name=None,
):
    """store data under parquet format

    Parameters
    ----------
        run_dir: str
            path to the simulation
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
        sub_dir: str, optional
            default is 'parquets'
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
    if sub_dir is None:
        sub_dir = 'parquets'
    parquet_path = join(run_dir, sub_dir, name)

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
    parquet_path = join(run_dir, "drifters", index)

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
        tile_dir = join(run_dir, 'tiling/')
        if os.path.isdir(tile_dir):
            self.tiler = tiler(tile_dir=tile_dir)
        parquet_dir = join(run_dir, 'parquets/')
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
        self.diagnostic_dir = join(self.run_dir,'diagnostics')
                
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
            store_parquet(self.run_dir, 
                          data,
                            overwrite=overwrite,
                            name=name,
                          sub_dir='diagnostics',
                          **kwargs
                         )
        if isinstance(data, xr.Dataset):
            # store to zarr format
            zarr_path = join(_dir,name+'.zarr')
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
        data_path = join(_dir, name)
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
    elif os.path.isdir(join(dirname, directory)):
        # directory is relative
        _dir = join(dirname, directory)
    else:
        if create:
            # need to create the directory
            _dir = join(dirname, directory)
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
