# Install useful libraries for equinox mit:

Download Miniconda3 (i.e. for python3) from the [conda website](https://conda.io/miniconda.html)
```
bash Miniconda3-latest-Linux-x86_64.sh
bash
conda update conda
conda create -n equinox -c conda-forge python=3.8 dask dask-jobqueue \
            xarray zarr netcdf4 python-graphviz \
            tqdm \
            jupyterlab ipywidgets \
            cartopy geopandas descartes \
            scikit-learn seaborn \
            hvplot geoviews datashader nodejs \
            intake-xarray gcsfs \
            cmocean gsw \
            xhistogram \
            pytide pyinterp
conda activate equinox
#conda install -c conda-forge xgcm xmitgcm
pip install git+https://github.com/xgcm/xgcm.git
#conda install -c conda-forge xgcm
pip install git+https://github.com/MITgcm/xmitgcm.git
pip install git+https://github.com/xgcm/xrft.git
pip install rechunker
conda install -c conda-forge parcels
pip install h3
conda install -c conda-forge zstandard  # maybe not necessary with following line:
conda install -c conda-forge fastparquet
conda install pywavelets
pip install git+git://github.com/psf/black
#
cd mit_equinox; pip install -e .
jupyter labextension install @jupyter-widgets/jupyterlab-manager \
                             @pyviz/jupyterlab_pyviz \
                             jupyter-leaflet
cp launch/jobqueue.yaml launch/distributed.yaml ~/.config/dask/
```

In order to add the environnement to kernels available to jupyter, you need to run:
```
python -m ipykernel install --user --name equinox --display-name "EQUINOX mit project env"
```

Uninstall library after `pip install -e .`:
- remove the egg file ( `print(distributed.__file__)` for example)
- from file `easy-install.pth`, remove the corresponding line (it should be a path to the source directory or of an egg file).

# General information about miniconda:

## Overview

Miniconda installers contain the conda package manager and Python.
Once miniconda is installed, you can use the conda command to install any other packages and create environments.

After downloading `Miniconda3-latest-Linux-x86_64.sh` or `Miniconda3-latest-MacOSX-x86_64.sh` you need to run it with: `bash Miniconda3-latest-MacOSX-x86_64.sh`

Miniconda must be used with bash. If you want to use it with csh, add in your .cshrc:
```
#
#----------------------------------------------------------------
# alias Miniconda
#----------------------------------------------------------------
#
source $home/.miniconda3/etc/profile.d/conda.csh
conda activate equinox
```

## Main commands:
What version, update conda
```
conda --version
conda update conda
```
Create new environment equinox
```
conda create --name equinox python
```
Switch to another environment (activate/deactivate) (or source_activate in csh)
```
conda activate equinox
```
To change your path from the current environment back to the root (or source_deactivate in csh)
```
conda deactivate
```
List all environments
```
conda info --envs
```
Delete an environment
```
conda remove --name equinox --all
```
View a list of packages and versions installed in an environmentSearch for a package
```
conda list
```
Check to see if a package is available for conda to install
```
conda search packagename
```
Install a new package
```
conda install packagename
```
Remove unused packages and caches:
```
conda clean --all
```
Remove conda
```
rm -rf /home/machine/username/miniconda3
```
where machine is the name of your computer and username is your username.


## Install a package from Anaconda.org

For packages that are not available using conda install, we can next look on Anaconda.org. Anaconda.org is a package management service for both public and private package repositories. Anaconda.org is a Continuum Analytics product, just like Anaconda and Miniconda.

In a browser, go to http://anaconda.org. We are looking for a package named “pestc4py”
There are more than a dozen copies of petsc4py available on Anaconda.org, first select your platform, then you can sort by number of downloads by clicking the “Downloads” heading.

Select the version that has the most downloads by clicking the package name. This brings you to the Anaconda.org detail page that shows the exact command to use to download it:

Check to see that the package downloaded
```
conda list
```

## Install a package with pip

For packages that are not available from conda or Anaconda.org, we can often install the package with pip (short for “pip installs packages”).
Exporting environment

```
conda env export > environment.yml on a machine
conda env create -f environment.yml -n $ENV_NAME on the new machine
```
