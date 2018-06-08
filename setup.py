#!/usr/bin/env python

from distutils.core import setup

INSTALL_REQUIRES = ['xarray >= 0.10.6', ]

setup(name='mitequinox',
      description='LLC4320 numerical simulation analysis',
      url='https://github.com/apatlpo/mit_equinox',
      packages=['mitequinox'])

#      install_requires=INSTALL_REQUIRES,


