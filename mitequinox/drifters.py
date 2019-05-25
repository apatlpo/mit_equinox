
import pickle
import numpy as np
import pandas as pd
import geopandas



def load_pair(p, data_dir):
    d0, id0 = pickle.load(open(data_dir+'single/%09d.p'%p[0], 'rb'))
    d1, id1 = pickle.load(open(data_dir+'single/%09d.p'%p[1], 'rb'))
    # get rid of gaps and interpolate if necessary
    d0 = d0[~pd.isnull(d0.index)]
    d1 = d1[~pd.isnull(d1.index)]
    #
    d0, d1 = d0.align(d1, join='inner')
    # fill gaps, should keep track of this
    d0, d1 = compute_vector(d0, d1)
    # should go through vectors for interpolation
    #d0 = d0.resample('H').interpolate('linear')
    #d1 = d1.resample('H').interpolate('linear')
    d0 = d0.resample('H').asfreq()
    d1 = d1.resample('H').asfreq()
    d0, d1 = compute_lonlat(d0, d1)
    # converts to geopandas
    gd0 = to_gdataframe(d0)
    gd1 = to_gdataframe(d1)
    return gd0, gd1, p


#--------------------------------------  utils ------------------------------------------------


RADIUS_EARTH = 6378.0
deg2rad = np.pi / 180.
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
    arclen = 2 * np.arcsin(np.sqrt((np.sin((llat2 - llat1) / 2)) ** 2 +
                                   np.cos(llat1) * np.cos(llat2) * (np.sin((llon2 - llon1) / 2)) ** 2))
    return arclen * RADIUS_EARTH



# geodesy with vectors
# https://www.movable-type.co.uk/scripts/latlong-vectors.html
    
def compute_vector(*args, lon_key='LON', lat_key='LAT'):
    if len(args)==1:
        df = args[0]
        df['v0'] = np.cos(deg2rad*df[lat_key])*np.cos(deg2rad*df[lon_key])
        df['v1'] = np.cos(deg2rad*df[lat_key])*np.sin(deg2rad*df[lon_key])
        df['v2'] = np.sin(deg2rad*df[lat_key])
        return df
    else:
        return [compute_vector(df) for df in args]

def drop_vector(*args, v0='v0', v1='v1', v2='v2'):
    # should test for type in order to accomodate for xarray types
    if len(args)==1:
        return args[0].drop(columns=[v0,v1,v2])
    else:
        return [df.drop(columns=[v0,v1,v2]) for df in args]
    
def compute_lonlat(*args, dropv=True, v0='v0', v1='v1', v2='v2',
                   lon_key='LON', lat_key='LAT'):
    if len(args)==1:
        df = args[0]
        # renormalize vectors
        n = np.sqrt(df[v0]**2+df[v1]**2+df[v2]**2)
        df[v0], df[v1], df[v2] = df[v0]/n, df[v1]/n, df[v2]/n
        # estimate LON/LAT
        df[lon_key] = np.arctan2(df[v1],df[v0])/deg2rad
        df[lat_key] = np.arctan2(df[v2],np.sqrt(df[v0]**2+df[v1]**2))/deg2rad
        if dropv:
            df = drop_vector(df, v0=v0, v1=v1, v2=v2)
        return df
    else:
        return [compute_lonlat(df, dropv=dropv, v0=v0, v1=v1, v2=v2, 
                               lon_key=lon_key, lat_key=lat_key ) for df in args]
    

def to_gdataframe(*args):
    if len(args)==1:
        df = args[0]
        return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(
                    df.LON, df.LAT))
    else:
        return [geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(
                    df.LON, df.LAT)) for df in args]
