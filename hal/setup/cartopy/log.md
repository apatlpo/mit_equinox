
Install shapefiles required to add coastlines:

Get url of data first and download zip f:
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

