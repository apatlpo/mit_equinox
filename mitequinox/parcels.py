
import os
import os.path as path
import pickle
from itertools import product

import numpy as np
import xarray as xr
import pandas as pd
import geopandas
from shapely.geometry import Polygon, Point
import pyproj

#from matplotlib import pyplot as plt

from xmitgcm.llcreader import llcmodel as llc


# ------------------------------- llc tiling -------------------------------------

_mates = [['maskW', 'maskS'],
          ['TAUX', 'TAUY'], 
          ['SSU', 'SSV'],
          ['dxC', 'dyC'], 
          ['dxG', 'dyG'], 
          ['hFacW','hFacS'], 
          ['rAw', 'rAs'],
         ]

crs_wgs84 = pyproj.CRS('EPSG:4326')
def default_projection(lon, lat):
    # https://proj.org/operations/projections/laea.html
    return "+proj=laea +lat_0={:d} +lon_0={:d} +units=m".format(int(lat), int(lon))


class tiler(object):
    
    def __init__(self,
                 ds=None,
                 factor=(4,4),
                 overlap=(500,500),
                 tile_dir=None,
                 name='tiling',
                 N_extra=1000,
                ):
        if ds is not None:
            self._build(ds, factor, overlap, N_extra)
            self.name=name
        elif tile_dir is not None:
            self._load(tile_dir)
        else:
            assert False, 'Either ds or tile_dir are required.'
        self.crs_wgs84 =  crs_wgs84

    def _build(self, ds, factor, overlap, N_extra, projection=None):
        ''' Generate tiling
        
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
        '''
    
        if 'time' in ds.dims:
            ds = ds.isel(time=0)
            
        # pair vector variables
        for m in _mates:
            if m[0] in ds:
                ds[m[0]].attrs['mate'] = m[1]
                ds[m[1]].attrs['mate'] = m[0]
                
        # concatenate face data
        ds = llc.faces_dataset_to_latlon(ds)

        # store initial grid size
        global_domain_size = ds.i.size, ds.j.size
    
        # add N_extra points along longitude to allow wrapping around dateline
        ds_extra = ds.isel(i=slice(0, N_extra), i_g=slice(0, N_extra))
        for dim in ['i', 'i_g']:
            ds_extra[dim] = ds_extra[dim] + ds[dim][-1] + 1
        ds = xr.merge([xr.concat([ds[v], ds_extra[v]], ds[v].dims[-1]) for v in ds])
                
        # start tiling
        tiles_1d, boundaries_1d = {}, {}
        tiles_1d['i'], boundaries_1d['i'] = tile_domain(global_domain_size[0], 
                                                        factor[0], 
                                                        overlap[0], 
                                                        wrap=True
                                                       )
        tiles_1d['j'], boundaries_1d['j'] = tile_domain(global_domain_size[1], 
                                                        factor[0], 
                                                        overlap[0]
                                                       )
        tiles = list(product(tiles_1d['i'], tiles_1d['j']))
        boundaries = list(product(boundaries_1d['i'], boundaries_1d['j']))
        N_tiles = len(tiles)

        data = {d: [ds.isel(i=s[0], j=s[1], i_g=s[0], j_g=s[1]) for s in slices] 
                for d, slices in zip(['tiles', 'boundaries'], [tiles, boundaries])}
        # tile data used for parcel advections
        # only boundary data will be used for reassignments

        # we need tile centers for geographic projections (dateline issues)
        centers = [ds.reset_coords()[['XC', 'YC']]
                   .isel(i=[int((t[0].start+t[0].stop)/2)],
                         j=[int((t[1].start+t[1].stop)/2)]
                        )
                   .squeeze() for t in tiles
                  ]
        centers = [{'lon': float(c.XC), 'lat': float(c.YC),
                    'i': int(c.i), 'j': int(c.j)
                   } for c in centers
                  ]
        
        # generate list of projections
        if not projection:
            projection = default_projection
        crs_strings = [projection(c['lon'], c['lat']) for c in centers] # for storage purposes
        CRS = list(map(pyproj.CRS, crs_strings))
        
        # build geometrical boundaries geopandas and shapely objects from boundaries
        S, G = {}, {}
        for key, _D in data.items():
            
            S[key], G[key] = [], []

            for d, crs in zip(_D, CRS):

                lon = _get_boundary(d['XC'])
                lat = _get_boundary(d['YC'])

                # generate shapely object
                _polygon = Polygon(list(zip(lon,lat)))
                # geopandas dataframe with projection immediately transformed to deal with dateline issues
                polygon_gdf = (geopandas.GeoDataFrame([1], 
                                                      geometry=[_polygon], 
                                                      crs=crs_wgs84
                                                     )
                               .to_crs(crs)
                              )
                # update shapely polygon on new projections
                polygon = polygon_gdf.geometry[0]
                S[key].append(polygon)
                G[key].append(polygon_gdf) 
        
        # store useful data
        V = ['global_domain_size', 'N_tiles', 
             'tiles', 'boundaries', 
             'CRS', 'crs_strings',
             'S', 'G',
            ]
        for v in V:
            setattr(self, v, eval(v))
            
    def _load(self, tile_dir):
        ''' Load tiler from tile_dir
        '''

        # load various dict
        with open(path.join(tile_dir, 'various.p'), 'rb') as f:
            various = pickle.load(f)
        self.global_domain_size = various['global_domain_size']
        self.N_tiles = various['N_tiles']

        # regenerate projections
        self.CRS = list(map(pyproj.CRS, various['crs_strings']))

        # rebuild slices (tiles, boundaries)
        D = {}
        for t in ['tiles', 'boundaries']:
            D[t] = [(slice(various['slices']['i_start_'+t].loc[i],
                           various['slices']['i_end_'+t].loc[i],
                          ),
                     slice(various['slices']['j_start_'+t].loc[i],
                           various['slices']['j_end_'+t].loc[i],
                          ),
                    ) for i in range(self.N_tiles)
                   ]
        self.tiles = D['tiles']
        self.boundaries = D['boundaries']

        # rebuild S and G
        S, G = {}, {} 
        for key in ['tiles', 'boundaries']:
            S[key], G[key] = [], []
            df = pd.read_csv(path.join(tile_dir, key+'_bdata.csv'))
            for i, crs in enumerate(self.CRS):
                polygon = Polygon(list(zip(df['x{:03d}'.format(i)], df['y{:03d}'.format(i)])))
                polygon_gdf = geopandas.GeoDataFrame([1],
                                                     geometry=[polygon], 
                                                     crs=crs
                                                    )
                S[key].append(polygon)
                G[key].append(polygon_gdf)
        self.S = S
        self.G = G
            

    def store(self, tile_dir, create=True):
        ''' Store tile to tile_dir
        '''
        
        _check_tile_directory(tile_dir, create)
        
        # tiles and boundaries and scales
        df_tiles = slices_to_dataframe(self.tiles)
        df_boundaries = slices_to_dataframe(self.boundaries)        
        df = pd.concat([df_tiles.add_suffix('_tiles'), 
                        df_boundaries.add_suffix('_boundaries')
                       ], axis=1)
        # add crs strings
        #df = df.assign(crs=self.crs_strings)
        # header
        #header = ['{}={}'.format(v) for v in [self.]]
        #df.to_csv(path.join(tile_dir, 'slices_crs_scalars.csv'))
        various = {'slices': df,
                   'crs_strings': self.crs_strings,
                   'global_domain_size': self.global_domain_size,
                   'N_tiles': self.N_tiles, 
                   }
        with open(path.join(tile_dir, 'various.p'), 'wb') as f:
            pickle.dump(various, f)
        
        # boundary data
        for key in ['tiles', 'boundaries']:
            df = pd.concat([polygon_to_dataframe(gdf.geometry[0], suffix='%03d'%i) 
                            for i, gdf in enumerate(self.G[key])], 
                            axis=1
                          )
            df.to_csv(path.join(tile_dir, key+'_bdata.csv'))

        print('Tiler stored in {}'.format(tile_dir))
        
    def assign(self, 
               lon=None, lat=None, 
               gs=None, 
               inner=True,
               tiles=None,
              ):
        ''' assign data to tiles
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
        '''
        if lon is not None and lat is not None:
            pts = geopandas.points_from_xy(lon, lat)
            gs = geopandas.GeoSeries(pts, crs=crs_wgs84)
        if inner:
            polygons = self.S['boundaries']
        else:
            polygons = self.S['tiles']

        if tiles is None:
            tiles = np.arange(self.N_tiles)
        elif isinstance(tiles, list):
            tiles = np.array(tiles)

        df = pd.DataFrame([gs.to_crs(self.CRS[t]).within(polygons[t]) for t in tiles]).T

        def _find_column(v):
            out = tiles[v]
            if out.size==0:
                return -1
            else:
                return out[0]
        tiles = df.apply(_find_column, axis=1)
        return tiles
    # assignment is slow, could we improve this?
    # https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    
    def tile(self, ds, tile=None, rechunk=True, persist=False):
        ''' Load zarr archive and tile

        Parameters
        ----------
        ds: xr.Dataset
            llc dataset that will be tiled
        tile: int, optional
            select one tile, returns a list of datasets otherwise (default)
        rechunk: boolean, optional
            set spatial chunks to -1
        '''
        if 'face' in ds.dims:
            # pair vector variables
            for m in _mates:
                if m[0] in ds:
                    ds[m[0]].attrs['mate'] = m[1]
                    ds[m[1]].attrs['mate'] = m[0]
            ds = llc.faces_dataset_to_latlon(ds).drop('face')
        if tile is None:
            tile=list(range(self.N_tiles))
        if isinstance(tile, list):
            return [self.tile(ds, tile=i, rechunk=rechunk, persist=persist) 
                    for i in tile
                   ]
        ds = ds.isel(i=self.tiles[tile][0],
                     j=self.tiles[tile][1],
                     i_g=self.tiles[tile][0],
                     j_g=self.tiles[tile][1],
                    )
        #
        _default_rechunk = {'i': -1, 'j': -1, 'i_g': -1, 'j_g': -1}
        if rechunk==True:
            ds = ds.chunk(_default_rechunk)
        elif isinstance(rechunk, dict):
            _default_rechunk.update(rechunk)
            ds = ds.chunk(_default_rechunk)
        #
        if persist:
            ds = ds.persist()
        return ds
    
        
def tile_domain(N, factor, overlap, wrap=False):
    ''' tile a 1D dimension into factor tiles with some overlap
    
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
    '''
    if wrap:
        _N = N+overlap
    else:
        _N = N
    n = np.ceil((_N+(factor-1)*overlap)/factor)
    slices = []
    boundaries = []
    for f in range(factor):
        slices.append(slice(int(f*(n-overlap)), int(min(f*(n-overlap)+n, _N))))
        # boundaries do need to be exact
        lower = 0
        if f>0 or wrap:
            lower = int(f*(n-overlap)+0.5*overlap)
        upper = N
        if f<factor-1 or wrap:
            upper = int(f*(n-overlap)+n-0.5*overlap)
        boundaries.append(slice(lower, upper))
    return slices, boundaries


def _get_boundary(da, i='i', j='j', stride=4):
    ''' return array along boundaries with optional striding
    '''
    return np.hstack([da.isel(**{j:  0}).values[:: stride], 
                      da.isel(**{i: -1}).values[:: stride], 
                      da.isel(**{j: -1}).values[::-stride], 
                      da.isel(**{i:  0}).values[::-stride],
                     ])

def slices_to_dataframe(slices):
    ''' transform list of slices into dataframe for storage
    '''
    i_start = [t[0].start for t in slices]
    i_stop = [t[0].stop for t in slices]
    j_start = [t[1].start for t in slices]
    j_stop = [t[1].stop for t in slices]
    return pd.DataFrame(map(list, zip(i_start, i_stop, j_start, j_stop)), 
                        columns=['i_start', 'i_end', 'j_start', 'j_end'])

def polygon_to_dataframe(polygon, suffix=''):
    _coords = polygon.boundary.coords
    return pd.DataFrame(_coords, columns=['x', 'y']).add_suffix(suffix)

def _check_tile_directory(tile_dir, create):
    """ Check existence of a directory and create it if necessary
    """
    # create diagnostics dir if not present
    if not path.isdir(tile_dir):
        if create:
            # need to create the directory
            os.mkdir(tile_dir)
            print('Create new diagnostic directory {}'.format(tile_dir))
        else:
            raise OSError('Directory does not exist')

def tile(tiler, ds, tile=None, rechunk=True):
    ''' Load zarr archive and tile

    Parameters
    ----------
    ds: xr.Dataset
        llc dataset that will be tiled
    tile: int, optional
        select one tile, returns a list of datasets otherwise (default)
    rechunk: boolean, optional
        set spatial chunks to -1
    '''
    if 'face' in ds.dims:
        # pair vector variables
        for m in _mates:
            if m[0] in ds:
                ds[m[0]].attrs['mate'] = m[1]
                ds[m[1]].attrs['mate'] = m[0]
        ds = llc.faces_dataset_to_latlon(ds).drop('face')
    ds = ds.isel(i=tiler.tiles[tile][0],
                 j=tiler.tiles[tile][1],
                 i_g=self.tiles[tile][0],
                 j_g=self.tiles[tile][1],
                )
    if rechunk:
        ds = ds.chunk({'i': -1, 'j': -1, 'i_g': -1, 'j_g': -1})
    return ds
    
def generate_randomly_located_data(lon=(0,180), 
                                   lat=(0,90), 
                                   N=1000):
    ''' Generate randomly located points
    '''
    # generate random points for testing
    points = [Point(_lon, _lat) for _lon,_lat in 
              zip(np.random.uniform(lon[0], lon[1],(N,)),
                  np.random.uniform(lat[0], lat[1],(N,)))
             ]
    return geopandas.GeoSeries(points, crs=crs_wgs84)    


# ------------------------------- parcels specific code ----------------------------

def fuse_dimensions(ds):
    coords = list(ds.coords)
    for c in ['i_g','j_g']:
        coords.remove(c) 
    ds = ds.reset_coords()
    ds = xr.merge([ds[v].rename({d: d[0] for d in ds[v].dims if d!='time'}) 
                   for v in ds
                  ])
    ds = ds.set_coords(coords)
    return ds

