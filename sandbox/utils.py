import numpy as np

cos = lambda theta: np.cos(theta)
sin = lambda theta: np.sin(theta)

def spherical2cartesian(v, lon, lat):
    """ e_r, e_east, e_north to ex, ey, ez
    """
    M = [[cos(lat)*cos(lon), -sin(lon), -sin(lat)*cos(lon)],
         [cos(lat)*sin(lon) , cos(lon), -sin(lat)*sin(lon)],
         [sin(lat) , 0., cos(lat)]]
    vout = []
    for i in range(3):
        vout.append(lon*lat*0.) # enforces right coordinates
        for j in range(3):
            vout[i] = vout[i] + M[i][j]*v[j]
    return vout

def cartesian2spherical(v, lon, lat):
    """ ex, ey, ez to e_r, e_east, e_north
    """
    M = [[cos(lat)*cos(lon), -sin(lon), -sin(lat)*cos(lon)],
         [cos(lat)*sin(lon) , cos(lon), -sin(lat)*sin(lon)],
         [sin(lat) , 0., cos(lat)]]
    #for i in range(3): print(v[i]) # lon/lat, lon/lat, lat
    vout = []
    for i in range(3):
        vout.append(lon*lat*0.) # enforces right coordinates
        for j in range(3):
            vout[i] = vout[i] + M[j][i]*v[j]
    return vout
    