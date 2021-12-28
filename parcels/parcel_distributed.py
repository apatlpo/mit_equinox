import os, shutil

import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime
import geopandas

import dask
from dask.delayed import delayed

#%matplotlib inline
#from matplotlib import pyplot as plt

import mitequinox.utils as ut
import mitequinox.parcels as pa

step_window_delayed = delayed(pa.step_window)

import gc
import ctypes
def trim_memory() -> int:
     libc = ctypes.CDLL("libc.so.6")
     return libc.malloc_trim(0)


# ---- Run parameters

#root_dir = '/home1/scratch/aponte/parcels/'
root_dir = '/home/datawork-lops-osi/equinox/mit4320/parcels/'
#root_dir = '/home1/datawork/slgentil/parcels/'

# 5x5 tiles dij=100 T=365 5jobs x 5workers
#run_name = 'global_T365j_dt1j_dij50'
#run_name = 'debug'
#run_name = 'global_extra_T365j_dt1j_dij50_new'
run_name = 'global_extra_T365j_dt1j_dij50_new_batch'


# will overwrite existing simulation
overwrite = True

# simulation parameters

T = 360 # length of the total run [days]
#T = 20 # ** debug

dt_window = timedelta(days=1.)
dt_outputs = timedelta(hours=1.)
dt_step = timedelta(hours=1.)
dt_seed = 10 # in days
dt_reboot = timedelta(days=20.)

init_dij = 50 # initial position subsampling compared to llc grid

# number of dask jobs launched for parcels simulations
dask_jobs = 13


# ----

def load_llc():
    """ load llc data
    """
    ds = ut.load_data(V=['SSU', 'SSV', 'Eta', 'SST', 'SSS'])
    grd = ut.load_grd()[['XC', 'YC', 'XG', 'YG']]
    ds = xr.merge([ds, grd])
    return ds


def generate_tiling(overwrite):
    """ Generate overlapping geographical tiling for parcels distributing
    or simply return existing one
    """
    if overwrite:
        print("creates and store tiling")
        # create tiling
        grd = ut.load_grd()[['XC', 'YC', 'XG', 'YG']].reset_coords().persist()
        tl = pa.tiler(ds=grd, factor=(5, 10), overlap=(250, 250))
        # store tiler
        tl.store(dirs["tiling"])
    else:
        print("load existing tiling")
        tl = pa.tiler(tile_dir=dirs["tiling"])

    return tl


def format_info(step, t_start, t_end):
    print('-------------------------------------------')
    print('step={}  /  start={}  /  end={}'
          .format(step,
                  t_start.strftime("%Y-%m-%d:%H"),
                  t_end.strftime("%Y-%m-%d:%H"),
                 )
         )

def run(dirs, tl, restart, cluster, client):
    """
    restart = -1 # 0: no restart, -1: last index, precise index otherwise
    """

    # load dataset
    ds = load_llc()

    # set start and end times
    t_start = ut.np64toDate(ds['time'][0].values)
    t_end = t_start + int(T/dt_window.days)*dt_window
    print('Global start = {}  /  Global end = {}'
          .format(t_start.strftime("%Y-%m-%d:%H"),
                  t_end.strftime("%Y-%m-%d:%H"),
                 )
         )

    # get new log filename for this run
    log_file = pa.name_log_file(dirs['run'])

    if restart==-1:
        log = pa.browse_log_files(dirs['run'])
        if log:
            restart = max(list(log))+1
        else:
            restart = 0

    # clean up data for restart
    tl.clean_up(dirs["run"], restart)

    if restart==0:
        global_parcel_number = 0
        local_numbers = {tile: 0 for tile in range(tl.N_tiles)}
        max_ids = {tile: None for tile in range(tl.N_tiles)}
    else:
        #print(log, restart)
        _log = log[restart-1]
        global_parcel_number = _log['global_parcel_number']
        #local_numbers = _log['local_numbers'] # TMP !!!
        max_ids = _log['max_ids']

    # skips steps if restart
    step_t = list(enumerate(ut.dateRange(t_start, t_end, dt_window)))
    step_t = step_t[restart:]
    reboot = int(dt_reboot/dt_window)
    flag_out = False
    if reboot<=len(step_t):
        step_t = step_t[:reboot]
        flag_out = True

    for step, local_t_start in step_t:

        local_t_end = local_t_start+dt_window+dt_step

        # print step info
        format_info(step, local_t_start, local_t_end)

        # load, tile (and store) llc data
        ds_tiles = pa.tile_store_llc(ds,
                                     slice(local_t_start, local_t_end, None),
                                     tl,
                                     netcdf=False,
                                    )
        # should test how much ds_tiles represents once loaded in memory and compare it to
        # the 750GB observed while running parcels

        # seed with more particles
        seed = ( (local_t_start-t_start).days%dt_seed == 0 )

        global_parcel_number0 = global_parcel_number

        dsteps = [step_window_delayed(tile, step,
                                          local_t_start, local_t_end,
                                          dt_window, dt_step, dt_outputs,
                                          tl,
                                          ds_tile=ds_tiles[tile],
                                          init_dij=init_dij,
                                          parcels_remove_on_land=True,
                                          pclass="extended",
                                          id_max=max_ids[tile],
                                          seed=seed,
                                         )
                  for tile in range(tl.N_tiles)
                 ]
        try:
            dsteps_out = dask.compute(*dsteps)
        except:
            return False

        # try to manually clean up memory
        # https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
        client.run(gc.collect)
        client.run(trim_memory) # should not be done systematically

        # update number of global number of drifters and maximum ids for each tiles
        _local_numbers = list(zip(*dsteps_out))[0]
        global_parcel_number = sum(_local_numbers)
        max_ids = {tile: int(out[1]) for tile, out in enumerate(dsteps_out)}
        local_numbers = {tile: int(out) for tile, out in enumerate(_local_numbers)}
        log = dict(global_parcel_number=global_parcel_number,
                   local_numbers=local_numbers,
                   max_ids=max_ids,
                  )

        # store log
        pa.store_log(log_file, step, log)

        print('Total number of particles = {}  ({:+d})'
              .format(global_parcel_number,
                      global_parcel_number-global_parcel_number0,
                     )
             )

        return flag_out

def spin_up_cluster(jobs):
    print("start spinning up dask cluster, jobs={}".format(jobs))
    return ut.spin_up_cluster(type="distributed",
                    jobs=jobs,
                    processes=4,
                    fraction=0.9,
                    cores=4,
                    walltime='48:00:00',
                    )

def close_dask(cluster, client):
    print("starts closing dask cluster ...")
    try:
        cluster.close()
    except:
        print("cluster.close failed ")
    # manually kill pbs jobs
    manual_kill_jobs()
    # close client
    client.close()
    print("... done")

def manual_kill_jobs():
    """ manually kill dask pbs jobs
    """

    import subprocess, getpass
    #
    username = getpass.getuser()
    print(username)
    #
    bashCommand = 'qstat'
    output = subprocess.check_output(bashCommand, shell=True)
    #
    for line in output.splitlines():
        lined = line.decode('UTF-8')
        if username in lined and 'dask' in lined:
            print(lined)
            pid = lined.split('.')[0]
            bashCommand = 'qdel '+str(pid)
            boutput = subprocess.check_output(bashCommand, shell=True)


if __name__ == '__main__':

    # create run directory tree
    print("1 - create_dir_tree")
    dirs = pa.create_dir_tree(root_dir, run_name, overwrite=overwrite)

    # create tiling
    print("2 - generate_tiling")
    tl = generate_tiling(overwrite)

    # create run directories, erase them if need be
    print("3 - create_tile_run_tree")
    _overwrite=False
    if restart==0 and overwrite:
        _overwrite = True
    tl.create_tile_run_tree(dirs["run"], overwrite=_overwrite)

    flag = True
    reboot = 0
    while flag:

        print(" ------------- reboot {} ------------- ".format(reboot))

        # sping up cluster
        cluster, client = spin_up_cluster(dask_jobs)

        # run parcels simulation
        flag = run(dirs, tl, restart, cluster, client)

        # close dask
        close_dask()

        reboot+=1
