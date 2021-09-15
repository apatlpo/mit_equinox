import numpy as np

import geopandas as gpd
from shapely.geometry import Polygon

from matplotlib import pyplot as plt
from cartopy import crs as ccrs
import cartopy.feature as cfeature

from .utils import load_swot_tracks

swot_tracks = load_swot_tracks()


def plot_site(bbox, figsize=(10,10), tracks=None, coast_resolution="50m"):
    """
    """

    if tracks is None:
        tracks = swot_tracks
    
    central_lon = (bbox[0]+bbox[1])*0.5
    central_lat = (bbox[2]+bbox[3])*0.5

    polygon = Polygon([(bbox[0], bbox[2]), 
                       (bbox[1], bbox[2]), 
                       (bbox[1], bbox[3]), 
                       (bbox[0], bbox[3]), 
                       (bbox[0], bbox[2]),
                      ])
    #poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=world.crs)
    gdf = tracks["swath"]
    gdf_clipped = gpd.clip(gdf, polygon)

    #crs = ccrs.Orthographic(central_lon, central_lat)
    crs = ccrs.AlbersEqualArea(central_lon, central_lat)

    crs_proj4 = crs.proj4_init

    fig, ax = plt.subplots(1, 1, 
                           subplot_kw={'projection': crs},
                           figsize=figsize,
                          )
    ax.set_extent(bbox)

    #_gdf = gdf.cx[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    _gdf = gdf_clipped
    gdf_crs = _gdf.to_crs(crs_proj4)
    ax.add_geometries(gdf_crs['geometry'],
                      crs=crs,
                      facecolor='grey', 
                      edgecolor='black',
                      alpha=0.5,
                     )

    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution=coast_resolution)

    return fig, ax





# compute mid point
mid = lambda a, b: dict(lon = (a["lon"]+b["lon"])*.5,
                        lat = (a["lat"]+b["lat"])*.5,
                       )

def follow_path(lon, lat, dl, speed):
    """ Build a trajectory with dl (nautical miles) along the lon, lat paths
    
    Parameters
    ----------
    lon, lat: lists
        List of reference positions
    dl: float
        Distance between drifter release
    speed: float
        Speed of the ship in knots (nautical miles per hour)
    Returns
    -------
    lon_i, lat_i: lists
        Lists of release positions
    time: list
        Release deployment time in hours
    """

    r = np.cos(np.pi/180.*np.mean(lat))
    time = [0.]
    for i in range(len(lon)-1):
        dlon = lon[i+1]-lon[i]
        dlat = lat[i+1]-lat[i]
        dl_segment = np.sqrt(dlon**2 * r**2 + dlat**2) # in deg
        ds = dl/60/dl_segment
        if i==0:
            lon_i, lat_i = [lon[0] - dlon * ds/2], [lat[0] - dlat * ds/2]
        s = 0
        while s<1:
            _lon = lon_i[-1] + dlon * ds
            _lat = lat_i[-1] + dlat * ds
            lon_i.append( _lon )
            lat_i.append( _lat )
            time.append( time[-1] + dl/speed )
            s += ds
            #flag = ( (_lon-lon[i]) * (_lon-lon[i+1]) < 0 ) | ( (_lat-lat[i]) * (_lat-lat[i+1]) < 0 )
    
    print("dl={}, speed={}, Number of drifter positions created = {}".format(dl, speed, len(time)))
    return lon_i, lat_i, time


