import threading

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo

from cmocean import cm

import numpy as np
import pandas as pd


# -------------------------------- various utils -------------------------------


def get_cmap_colors(Nc, cmap="plasma"):
    """load colors from a colormap to plot lines

    Parameters
    ----------
    Nc: int
        Number of colors to select
    cmap: str, optional
        Colormap to pick color from (default: 'plasma')
    """
    scalarMap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=Nc), cmap=cmap)
    return [scalarMap.to_rgba(i) for i in range(Nc)]


_default_cmaps = {
    "SSU": cm.balance,
    "SSV": cm.balance,
    "SSU_geo": cm.balance,
    "SSV_geo": cm.balance,
    "Eta": plt.get_cmap("RdGy_r"),
    "SST": cm.thermal,
    "SSS": cm.haline,
}


def _get_cmap(v, cmap):
    if cmap is None and v.name in _default_cmaps:
        return _default_cmaps[v.name]
    elif cmap is not None:
        return cmap
    else:
        return plt.get_cmap("magma")


# ------------------------------ plot ---------------------------------------

#
def plot_scalar(
    v,
    colorbar=False,
    title=None,
    vmin=None,
    vmax=None,
    savefig=None,
    offline=False,
    coast_resolution="110m",
    figsize=(10, 10),
    cmap=None,
):
    #
    if vmin is None:
        vmin = v.min()
    if vmax is None:
        vmax = v.max()
    #
    MPL_LOCK = threading.Lock()
    with MPL_LOCK:
        if offline:
            plt.switch_backend("agg")
        colmap = _get_cmap(v, cmap)
        #
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        try:
            im = v.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                vmin=vmin,
                vmax=vmax,
                x="XC",
                y="YC",
                add_colorbar=colorbar,
                cmap=colmap,
            )
            fig.colorbar(im)
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=2,
                color="k",
                alpha=0.5,
                linestyle="--",
            )
            gl.xlabels_top = False
            if coast_resolution is not None:
                ax.coastlines(resolution=coast_resolution, color="k")
        except:
            pass
        #
        if title is not None:
            ax.set_title(title)
        #
        if savefig is not None:
            fig.savefig(savefig, dpi=150)
            plt.close(fig)
        #
        # if not offline:
        #    plt.show()
        return fig, ax


#
def quick_llc_plot(data, axis_off=False, **kwargs):
    """quick plotter for llc4320 data"""
    face_to_axis = {
        0: (2, 0),
        1: (1, 0),
        2: (0, 0),
        3: (2, 1),
        4: (1, 1),
        5: (0, 1),
        7: (0, 2),
        8: (1, 2),
        9: (2, 2),
        10: (0, 3),
        11: (1, 3),
        12: (2, 3),
    }
    transpose = [7, 8, 9, 10, 11, 12]
    gridspec_kw = dict(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig, axes = plt.subplots(nrows=3, ncols=4, gridspec_kw=gridspec_kw)
    for face, (j, i) in face_to_axis.items():
        data_ax = data.sel(face=face)
        ax = axes[j, i]
        yincrease = True
        if face in transpose:
            data_ax = data_ax.transpose()
            yincrease = False
        data_ax.plot(ax=ax, yincrease=yincrease, **kwargs)
        if axis_off:
            ax.axis("off")
        ax.set_title("")
    return fig, axes


# ------------------------------ pretty ---------------------------------------


_region_params = {
    "atlantic": {
        "faces": [0, 1, 2, 6, 10, 11, 12],
        "extent": [-110, 25, -70, 70],
        "dticks": [10, 10],
        "projection": ccrs.Mollweide(),
    },
    "south-atlantic": {
        "faces": [1, 11, 0, 12],
        "extent": [-50, 20, -60, 5],
        "dticks": [10, 10],
        "projection": ccrs.LambertAzimuthalEqualArea(
            central_longitude=-15.0, central_latitude=-30
        ),
    },
    "global": {
        "faces": [i for i in range(13) if i != 6],
        "extent": "global",
        "dticks": [10, 10],
        "projection": ccrs.EckertIII(),
    },
    "global_pacific": {
        "faces": [i for i in range(13) if i != 6],
        "extent": "global",
        "dticks": [10, 10],
        "projection": ccrs.EckertIII(central_longitude=-180),
    },
}
#                  'south-atlantic':{'faces':[0,1,11,12],'extent':[-100,25,-70,5]},}


def plot_pretty(
    v,
    title=None,
    vmin=None,
    vmax=None,
    fig=None,
    ax=None,
    region="global",
    projection=None,
    extent=None,
    ignore_face=[],
    cmap=None,
    colorbar=False,
    colorbar_kwargs={},
    gridlines=True,
    land=True,
    coast_resolution="110m",
    offline=False,
    figsize=(15, 15),
    savefig=None,
    **kwargs,
):
    #
    if vmin is None:
        vmin = v.min().values
    if vmax is None:
        vmax = v.max().values
    #
    MPL_LOCK = threading.Lock()
    with MPL_LOCK:
        if offline:
            plt.switch_backend("agg")
        colmap = _get_cmap(v, cmap)
        #
        if "face" not in v.dims:
            v = v.expand_dims("face")
        #
        if isinstance(region, dict):
            params = region
        else:
            params = _region_params[region]
        _extent = params["extent"]
        _faces = (face for face in params["faces"] if face not in ignore_face)
        _projection = params["projection"]
        _dticks = params["dticks"]
        #
        if extent is not None:
            _extent = extent
        if projection is not None:
            _projection = projection
        #
        if fig is None and ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=_projection)
        # coastlines and land:
        # if coast_resolution is not None:
        #    ax.coastlines(resolution=coast_resolution, color='k')
        if land:
            if isinstance(land, dict):
                land_feature = cfeature.NaturalEarthFeature(*land['args'], 
                                                            **land['kwargs'],
                                                           )
                #land = {'args': ['physical', 'land', '10m'], 
                #        'kwargs': {edgecolor='face',
                #                   facecolor=cfeature.COLORS['land'],
                #                  }}
            else:
                land_feature = cfeature.LAND
            ax.add_feature(land_feature, zorder=2)
        if _extent == "global":
            # _extent = ax.set_extent()
            _extent = ax.get_extent()
        elif _extent is not None:
            ax.set_extent(_extent)
        for face in _faces:
            vplt = v.sel(face=face)
            if face in [6, 7, 8, 9]:
                eps = 0.2  # found empirically
                # this deals with dateline crossing areas
                im = vplt.where(
                    (vplt.XC > 0) & (vplt.XC < 180.0 - eps)
                ).plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    x="XC",
                    y="YC",
                    cmap=colmap,
                    add_colorbar=False,
                )
                im = vplt.where(
                    (vplt.XC < 0) & (vplt.XC > -180.0 + eps)
                ).plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    x="XC",
                    y="YC",
                    cmap=colmap,
                    add_colorbar=False,
                )
            else:
                im = vplt.plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    x="XC",
                    y="YC",
                    cmap=colmap,
                    add_colorbar=False,
                )
        if extent == "global":
            ax.set_extent("global")
        if colorbar:
            cbar = fig.colorbar(im, **colorbar_kwargs)
        else:
            cbar = None
        if gridlines and _extent is not None:
            # grid lines:
            xticks = np.arange(
                _extent[0],
                _extent[1] + _dticks[0],
                _dticks[1] * np.sign(_extent[1] - _extent[0]),
            )
            ax.set_xticks(xticks, crs=ccrs.PlateCarree())
            yticks = np.arange(
                _extent[2],
                _extent[3] + _dticks[1],
                _dticks[1] * np.sign(_extent[3] - _extent[2]),
            )
            ax.set_yticks(yticks, crs=ccrs.PlateCarree())
            gl = ax.grid()
        else:
            gl = ax.gridlines()  # draw_labels=True
        # ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
        # ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        # lon_formatter = LongitudeFormatter(zero_direction_label=True)
        # lat_formatter = LatitudeFormatter()
        # ax.xaxis.set_major_formatter(lon_formatter)
        # ax.yaxis.set_major_formatter(lat_formatter)
        # only with platecarre
        # if projection is 'PlateCarre':
        #    gl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='k',
        #                    alpha=0.5, linestyle='--')
        #    gl.xlabels_top = False
        #
        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 20, "fontweight": "bold"})
        #
        if savefig is not None:
            fig.savefig(savefig, dpi=150)
            plt.close(fig)
        #
        # if not offline:
        #    plt.show()
        return {"fig": fig, "ax": ax, "cbar": cbar}


# ------------------------------ scale_bar ---------------------------------------


def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(
            f"End is closer to start ({initial_distance}) than "
            f"given distance ({distance})."
        )

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys).base[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(
    ax,
    location,
    length,
    metres_per_unit=1000,
    unit_name="km",
    tol=0.01,
    angle=0,
    color="black",
    linewidth=3,
    text_offset=0.005,
    ha="center",
    va="bottom",
    plot_kwargs=None,
    text_kwargs=None,
    **kwargs,
):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates.
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {"linewidth": linewidth, "color": color, **plot_kwargs, **kwargs}
    text_kwargs = {
        "ha": ha,
        "va": va,
        "rotation": angle,
        "color": color,
        **text_kwargs,
        **kwargs,
    }

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad, tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(
        *text_location,
        f"{length} {unit_name}",
        rotation_mode="anchor",
        transform=ax.transAxes,
        **text_kwargs,
    )
