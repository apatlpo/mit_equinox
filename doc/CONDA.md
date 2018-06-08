# Install useful libraries for equinox mit:

Should we write dependencies in setup.py file in order to automate
the install of some libraries ?

Temporary install (waiting for dask 0.18.0 and xscale PR)
```
conda update conda
conda create -n equinox -c conda-forge python=3.6 dask xarray \
      jupyterlab cartopy utide
source activate equinox
conda uninstall dask --force
conda uninstall distributed --force
pip install git+https://github.com/dask/dask.git
pip install git+https://github.com/dask/distributed.git
pip install git+https://github.com/dask/dask-jobqueue.git
pip install git+https://github.com/xgcm/xmitgcm.git
pip install git+https://github.com/xgcm/xgcm.git
pip install git+https://github.com/apatlpo/xscale.git
cd mit_equinox; pip install -e .
cp datarmor/jobqueue.yaml ~/.config/dask/
pip install cmocean
```

Regular install:
Download Miniconda3 (i.e. for python3) from the [conda website](https://conda.io/miniconda.html)
```
bash Miniconda3-latest-Linux-x86_64.sh
bash
conda update conda
conda create -n equinox -c conda-forge python=3.6 dask xarray \
      jupyterlab cartopy utide
source activate equinox
pip install git+https://github.com/dask/dask-jobqueue.git
pip install git+https://github.com/xgcm/xmitgcm.git
pip install git+https://github.com/xgcm/xgcm.git
pip install git+https://github.com/rabernat/xrft.git
pip install git+https://github.com/serazing/xscale.git
cd mit_equinox; pip install -e .
pip install cmocean
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

Miniconda must be used with bash. If you want to use it with csh, add in your .cshrc (not ideal solution)
```
#
#----------------------------------------------------------------
# alias Miniconda
#----------------------------------------------------------------
#
setenv PATH ${PATH}: /home/machine/username/miniconda3/bin
alias source_activate 'setenv OLDPATH ${PATH};setenv PATH /home/machine/username/miniconda3/envs/\!*/bin:${PATH}'
alias source_deactivate 'setenv PATH $OLDPATH'
```
where machine is the name of your computer and username is your username.


## Main commands:
What version, update conda
```
conda --version
conda update conda
```
Create new environment myenv
```
conda create --name myenv python
```
Switch to another environment (activate/deactivate) (or source_activate in csh)
```
source activate myenv
```
To change your path from the current environment back to the root (or source_deactivate in csh)
```
source deactivate
```
List all environments
```
conda info --envs
```
Delete an environment
```
conda remove --name myenv --all
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


