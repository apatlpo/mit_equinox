# install of xarray/dask and other on HAL

In order to have internet access you need to set:
```
# ask aurelien
```

In order to pull and push from/to github, you need to run once:
```
# ask aurelien
```

For conda, you need to deactivate ssl in .condarc:
```
cat ~/.condarc
ssl_verify: false
```

This is what was installed:
```
module load conda
conda create -n equinox -c conda-forge python=3.6 
conda activate equinox
conda install -c conda-forge  dask xarray jupyterlab cartopy python-graphviz dask-jobqueue zarr netcdf4 spectrum seaborn
pip install --cert /etc/pki/tls/certs/ca-bundle.crt git+https://github.com/apatlpo/xmitgcm.git@angles
pip install --cert /etc/pki/tls/certs/ca-bundle.crt git+https://github.com/xgcm/xgcm.git
pip install --cert /etc/pki/tls/certs/ca-bundle.crt cmocean
git clone https://github.com/apatlpo/mit_equinox.git
cd mit_equinox
pip install -e .
```

You need to connect the equinox environment and jupyterhub:
```
ipython kernel install --user --name equinox
```
See help on HAL [wiki](https://gitlab.cnes.fr/inno/rt-nouvelles-technos-distrib/blob/master/doc/utilisation_hub.rst)

cartopy coastlines need to be downloaded and massaged, see `cartopy/log.md`

Optional (for testing): install latest xarray code and h5netcdf:
```
conda uninstall xarray --force
pip install --cert /etc/pki/tls/certs/ca-bundle.crt https://github.com/pydata/xarray/archive/master.zip 
pip install --cert /etc/pki/tls/certs/ca-bundle.crt h5netcdf 
```

[Holoviews](http://holoviews.org):
```
conda install -c conda-forge holoviews
```


---

This is not what was installed because conda-forge is not working for some reason
```
module load conda
conda create -n equinox -c conda-forge python=3.6 dask dask-jobqueue \
            xarray jupyterlab cartopy zarr python-graphviz spectrum 
conda activate equinox
#pip install git+https://github.com/xgcm/xmitgcm.git
pip install git+https://github.com/apatlpo/xmitgcm.git@angles
pip install git+https://github.com/xgcm/xgcm.git
#pip install git+https://github.com/rabernat/xrft.git
#pip install git+https://github.com/apatlpo/xscale.git
#pip install git+https://github.com/apatlpo/UTide.git
#pip install cmocean
cd mit_equinox; pip install -e .
cp datarmor/jobqueue.yaml datarmor/distributed.yaml ~/.config/dask/
```
