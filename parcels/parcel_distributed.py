import os, shutil
import logging
import traceback

import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime
import geopandas

import dask
from dask.delayed import delayed
from dask.distributed import performance_report

# from distributed.diagnostics import MemorySampler # requires latest dask ... wait for finishing this production run

# try to force flushing of memory
import gc, ctypes

import mitequinox.utils as ut
import mitequinox.parcels as pa

step_window_delayed = delayed(pa.step_window)

# ---- Run parameters

# root_dir = '/home1/scratch/aponte/parcels/'
root_dir = "/home/datawork-lops-osi/equinox/mit4320/parcels/"
# root_dir = '/home1/datawork/slgentil/parcels/'

# 5x5 tiles dij=100 T=365 5jobs x 5workers
# run_name = 'global_T365j_dt1j_dij50'
# run_name = 'debug'
# run_name = 'global_extra_T365j_dt1j_dij50_new'
# run_name = 'global_extra_T365j_dt1j_dij50_new_batch'
run_name = "global_T365j_dt1j_dij50"

# will overwrite existing simulation
# overwrite = True
overwrite = False

# simulation parameters

T = 360  # length of the total run [days]

dt_window = timedelta(days=1.0)
dt_outputs = timedelta(hours=1.0)
dt_step = timedelta(hours=1.0)
dt_seed = 10  # in days
dt_reboot = timedelta(days=30.0)

init_dij = 50  # initial position subsampling compared to llc grid

# number of dask jobs launched for parcels simulations
dask_jobs = 12
jobqueuekw = dict(processes=4, cores=4)

# following is not allowed on datarmor:
# dask_jobs = 12*4
# jobqueuekw=dict(processes=1, cores=1, memory="30GB",
#                resource_spec="select=1:ncpus=6:mem=30GB",
#               )

# dev !!
# dt_reboot = timedelta(days=2.)
# T = 6 # length of the total run [days]
# run_name = 'global_T6j_dt2j_dij50'
# dt_window = timedelta(days=2.)
# run_name = 'global_T6j_dt1j_dij50'
# dt_window = timedelta(days=1.)
# run_name = 'global_T6j_dt0p5j_dij50'
# dt_window = timedelta(hours=12.)
# jobqueuekw=dict(processes=4, cores=16)
# jobqueuekw["env_extra"] =  ['export MKL_NUM_THREADS=1', 'export NUMEXPR_NUM_THREADS=1',
#                            'export OMP_NUM_THREADS=4', 'export OPENBLAS_NUM_THREADS=1']
# run_name = 'global_T6j_dt0p5j_dij50_4OMP_NUM_THREADS'

# ----


def load_llc():
    """load llc data"""
    ds = ut.load_data(V=["SSU", "SSV", "Eta", "SST", "SSS"])
    grd = ut.load_grd()[["XC", "YC", "XG", "YG"]]
    ds = xr.merge([ds, grd])
    return ds


def generate_tiling(dirs, overwrite):
    """Generate overlapping geographical tiling for parcels distributing
    or simply return existing one
    """
    if overwrite:
        print("creates and store tiling")
        logging.info("creates and store tiling")
        # create tiling
        grd = ut.load_grd()[["XC", "YC", "XG", "YG"]].reset_coords().persist()
        tl = pa.tiler(ds=grd, factor=(5, 10), overlap=(250, 250))
        # store tiler
        tl.store(dirs["tiling"])
    else:
        print("load existing tiling")
        logging.info("load existing tiling")
        tl = pa.tiler(tile_dir=dirs["tiling"])

    return tl


def format_info(step, t_start, t_end):
    print("-------------------------------------------")
    str_fmt = "step={}  /  start={}  /  end={}".format(
        step,
        t_start.strftime("%Y-%m-%d %H:%M"),
        t_end.strftime("%Y-%m-%d %H:%M"),
    )
    print(str_fmt)
    logging.info(str_fmt)


def run(dirs, tl, cluster, client):
    """ """

    # load dataset
    ds = load_llc()

    # set start and end times
    t_start = ut.np64toDate(ds["time"][0].values)
    t_end = t_start + int(T * 86400 / dt_window.total_seconds()) * dt_window
    str_fmt = "Global start = {}  /  Global end = {}".format(
        t_start.strftime("%Y-%m-%d %H:%M"),
        t_end.strftime("%Y-%m-%d %H:%M"),
    )
    print(str_fmt)
    logging.info(str_fmt)

    # get new log filename for this run
    log_file = pa.name_log_file(dirs["run"])

    # if restart==-1:
    log = pa.browse_log_files(dirs["run"])
    if log:
        restart = max(list(log)) + 1
    else:
        restart = 0

    # clean up data for restart
    tl.clean_up(dirs["run"], restart)

    if restart == 0:
        global_parcel_number = 0
        local_numbers = {tile: 0 for tile in range(tl.N_tiles)}
        max_ids = {tile: None for tile in range(tl.N_tiles)}
    else:
        # print(log, restart)
        _log = log[restart - 1]
        global_parcel_number = _log["global_parcel_number"]
        # local_numbers = _log['local_numbers'] # TMP !!!
        max_ids = _log["max_ids"]

    # skips steps if restart
    step_t = list(enumerate(ut.dateRange(t_start, t_end, dt_window)))
    step_t = step_t[restart:]
    reboot = int(dt_reboot / dt_window)
    flag_out = False
    print("reboot={}, len(step_t)={}".format(reboot, len(step_t)))
    logging.info("reboot=%d, len(step_t)=%d", reboot, len(step_t))
    if reboot < len(step_t):
        step_t = step_t[:reboot]
        flag_out = True

    for step, local_t_start in step_t:

        local_t_end = local_t_start + dt_window + dt_step

        # print step info
        format_info(step, local_t_start, local_t_end)

        # load, tile (and store) llc data
        ds_tiles = pa.tile_store_llc(
            ds,
            slice(local_t_start, local_t_end, None),
            tl,
            netcdf=False,
        )

        # seed with more particles
        seed = (local_t_start - t_start).days % dt_seed == 0

        global_parcel_number0 = global_parcel_number

        dsteps = [
            step_window_delayed(
                tile,
                step,
                local_t_start,
                local_t_end,
                dt_window,
                dt_step,
                dt_outputs,
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
            # force even distribution amongst workers
            # dsteps_out = client.compute(dsteps, sync=True) # this is not working at the moment
            # http://distributed.dask.org/en/stable/locality.html#specify-workers-with-compute-persist
            # workers = list(client.scheduler_info()["workers"])[:len(dsteps)]
            # dsteps_out = dask.compute(*dsteps, workers={d: w for d, w in zip(dsteps,workers)})
            # dsteps_out = client.compute(*dsteps, workers={d: w for d, w in zip(dsteps,workers)})
        except:
            logging.exception("Got exception in step loop")
            return False

        # try to manually clean up memory
        # https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
        client.run(gc.collect)
        client.run(trim_memory)  # should not be done systematically

        # update number of global number of drifters and maximum ids for each tiles
        _local_numbers = list(zip(*dsteps_out))[0]
        global_parcel_number = sum(_local_numbers)
        max_ids = {tile: int(out[1]) for tile, out in enumerate(dsteps_out)}
        local_numbers = {tile: int(out) for tile, out in enumerate(_local_numbers)}
        log = dict(
            global_parcel_number=global_parcel_number,
            local_numbers=local_numbers,
            max_ids=max_ids,
        )

        # store log
        pa.store_log(log_file, step, log)

        str_fmt = "Total number of particles = {}  ({:+d})".format(
            global_parcel_number,
            global_parcel_number - global_parcel_number0,
        )
        print(str_fmt)
        logging.info(str_fmt)

    return flag_out


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def spin_up_cluster(jobs):
    print("start spinning up dask cluster, jobs={}".format(jobs))
    logging.info("start spinning up dask cluster, jobs={}".format(jobs))
    cluster, client = ut.spin_up_cluster(
        "distributed",
        jobs=jobs,
        fraction=0.9,
        walltime="06:00:00",
        **jobqueuekw,
    )
    logging.info("dashboard via ssh: " + ut.dashboard_ssh_forward(client))
    return cluster, client


def close_dask(cluster, client):
    print("starts closing dask cluster ...")
    logging.info("starts closing dask cluster ...")
    try:
        cluster.close()
    except:
        print("cluster.close failed ")
        logging.exception("cluster.close failed ...")
    # manually kill pbs jobs
    manual_kill_jobs()
    # close client
    client.close()
    print("... done")
    logging.info("... done")


def manual_kill_jobs():
    """manually kill dask pbs jobs"""

    import subprocess, getpass

    #
    username = getpass.getuser()
    #
    bashCommand = "qstat"
    output = subprocess.check_output(bashCommand, shell=True)
    #
    for line in output.splitlines():
        lined = line.decode("UTF-8")
        if username in lined and "dask" in lined:
            # print(lined)
            pid = lined.split(".")[0]
            bashCommand = "qdel " + str(pid)
            print(bashCommand)
            logging.info(" " + bashCommand)
            try:
                boutput = subprocess.check_output(bashCommand, shell=True)
            except subprocess.CalledProcessError as e:
                # print(e.output.decode())
                pass


def main():

    # create run directory tree
    print("1 - create_dir_tree")
    logging.info("1 - create_dir_tree")
    dirs = pa.create_dir_tree(root_dir, run_name, overwrite=overwrite)

    # create tiling
    print("2 - generate_tiling")
    logging.info("2 - generate_tiling")
    tl = generate_tiling(dirs, overwrite)

    # create run directories, erase them if need be
    print("3 - create_tile_run_tree")
    logging.info("3 - create_tile_run_tree")
    # _overwrite=False
    # if restart==0 and overwrite:
    #    _overwrite = True
    tl.create_tile_run_tree(dirs["run"], overwrite=overwrite)

    print("4 - start main loop")
    logging.info("4 - start main loop")

    flag = True
    reboot = 0
    while flag:

        print(" ------------- reboot {} ------------- ".format(reboot))
        logging.info("--- reboot %d", reboot)

        # sping up cluster
        cluster, client = spin_up_cluster(dask_jobs)
        # cluster, client = None, None
        print(cluster)

        # run parcels simulation
        # ms = MemorySampler()
        with performance_report(filename=f"dask-report-{reboot}.html"):
            flag = run(dirs, tl, cluster, client)
        # with performance_report(filename=f"dask-report-{reboot}.html"), ms.sample(f"reboot{reboot}")
        # MemorySampler requires latest dask ... wait for finishing this production run
        # https://distributed.dask.org/en/latest/diagnosing-performance.html#analysing-memory-usage-over-time
        # ms.to_pandas().to_netcdf ...

        # close dask
        close_dask(cluster, client)

        # if reboot>10:
        #    flag=False

        reboot += 1


def debug():
    from dask_jobqueue import PBSCluster

    cluster = PBSCluster(**jobqueuekw)
    print(cluster.job_script())


if __name__ == "__main__":

    # to std output
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # to file
    logging.basicConfig(
        filename="distributed.log",
        level=logging.INFO,
    )
    # level order is: DEBUG, INFO, WARNING, ERROR
    # encoding='utf-8', # available only in latests python versions

    # dask memory logging:
    # https://blog.dask.org/2021/03/11/dask_memory_usage
    # log dask summary?

    main()
    # debug()

    print("- all done")
    logging.info("- all done")
