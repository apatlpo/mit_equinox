# HAL install notes

[hal wiki](https://gitlab.cnes.fr/hpc/wikiHPC/wikis/home)


### .bashrc

```
# for internet access, type proxyon
export proxy='http://login:password@proxy-surf.loc.cnes.fr:8050'
alias proxyon='export http_proxy="$proxy"; export https_proxy="$proxy"'
export no_proxy='cnes.fr,sis.cnes.fr,gitlab.cnes.fr'

module load conda # this does work, python/pip points toward wrong directions
conda activate equinox

umask 0003
```

### proxy

Needs to be setup in `.bashrc`

### .condarc

Create `.condarc` with:

```
ssl_verify: /etc/pki/tls/certs/ca-bundle.crt
channel_alias: https://artifactory.cnes.fr/artifactory/api/conda/
default_channels:
  - https://artifactory.cnes.fr/artifactory/api/conda/conda-main-remote
  - https://artifactory.cnes.fr/artifactory/api/conda/conda-r-remote
channels:
  - defaults
```

### .gitconfig

```
[http]
        proxy = http://login:password@proxy-surf.loc.cnes.fr:8050
[https]
        proxy = http://login:password@proxy-surf.loc.cnes.fr:8050
```

### python libraries

```
conda create -n equinox -c conda-forge python=3.8 dask dask-jobqueue \
            xarray zarr netcdf4 python-graphviz \
            tqdm \
            jupyterlab ipywidgets \
            cartopy geopandas descartes \
            scikit-learn seaborn \
            hvplot geoviews datashader nodejs \
            intake-xarray gcsfs \
            cmocean gsw \
            pytide pyinterp
conda activate equinox
conda install pip  # new, required for pip install git+https
# verify with `which pip` that there are not multiple versions of pip
#    consider using pip3 if it is the case
pip install git+https://github.com/xgcm/xgcm.git
pip install git+https://github.com/MITgcm/xmitgcm.git
pip install git+https://github.com/xgcm/xrft.git
pip install rechunker
conda install -c conda-forge parcels
conda install pywavelets
#
cd mit_equinox; pip install -e .
jupyter labextension install @jupyter-widgets/jupyterlab-manager \
                             @pyviz/jupyterlab_pyviz \
                             jupyter-leaflet
cp launch/jobqueue.yaml launch/distributed.yaml ~/.config/dask/
```


### cartopy

Your need to install shapefiles required to add coastlines.
To do that, get url of data first and download zip f:
```
run cartopy_download.py --dry-run gshhs
run cartopy_download.py --dry-run physical
...
```
and dowload files:
```
wget https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhs/oldversions/version2.2.0/GSHHS_shp_2.2.0.zip
wget http://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip
...
``

Unzip, move and rename in ipython with:

```
%run move_and_rename.py  ne_110m_coastline.zip
%run move_and_rename.py  ne_10m_coastline.zip
%run move_and_rename.py  ne_50m_coastline.zip
%run move_and_rename.py  ne_110m_ocean.zip
%run move_and_rename.py  ne_10m_ocean.zip
%run move_and_rename.py  ne_50m_ocean.zip
%run move_and_rename.py  ne_110m_land.zip
%run move_and_rename.py  ne_10m_land.zip
%run move_and_rename.py  ne_50m_land.zip
```

Check that it is working:
```
in ipython (qsub -I -X)
In [1]: %run features.py
```

Logs:

```
In [6]: run cartopy_download.py --dry-run gshhs
URL: https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhs/oldversions/version2.2.0/GSHHS_shp_2.2.0.zip

In [7]: run cartopy_download.py --dry-run physical
URL: http://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip
URL: http://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip
URL: http://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip
URL: http://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip
URL: http://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip
URL: http://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip
URL: http://naciscdn.org/naturalearth/110m/physical/ne_110m_ocean.zip
URL: http://naciscdn.org/naturalearth/50m/physical/ne_50m_ocean.zip
URL: http://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip
URL: http://naciscdn.org/naturalearth/110m/physical/ne_110m_rivers_lake_centerlines.zip
URL: http://naciscdn.org/naturalearth/50m/physical/ne_50m_rivers_lake_centerlines.zip
URL: http://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip
URL: http://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip
URL: http://naciscdn.org/naturalearth/50m/physical/ne_50m_lakes.zip
URL: http://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip
URL: http://naciscdn.org/naturalearth/110m/physical/ne_110m_geography_regions_polys.zip
URL: http://naciscdn.org/naturalearth/50m/physical/ne_50m_geography_regions_polys.zip
URL: http://naciscdn.org/naturalearth/10m/physical/ne_10m_geography_regions_polys.zip
URL: http://naciscdn.org/naturalearth/110m/physical/ne_110m_geography_regions_points.zip
URL: http://naciscdn.org/naturalearth/50m/physical/ne_50m_geography_regions_points.zip
URL: http://naciscdn.org/naturalearth/10m/physical/ne_10m_geography_regions_points.zip
URL: http://naciscdn.org/naturalearth/110m/physical/ne_110m_geography_marine_polys.zip
URL: http://naciscdn.org/naturalearth/50m/physical/ne_50m_geography_marine_polys.zip
URL: http://naciscdn.org/naturalearth/10m/physical/ne_10m_geography_marine_polys.zip
URL: http://naciscdn.org/naturalearth/110m/physical/ne_110m_glaciated_areas.zip
URL: http://naciscdn.org/naturalearth/50m/physical/ne_50m_glaciated_areas.zip
URL: http://naciscdn.org/naturalearth/10m/physical/ne_10m_glaciated_areas.zip
```
