import os, shutil
import logging
from time import time

import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime
import geopandas

import dask
from dask.delayed import delayed
from dask.distributed import performance_report, wait
from distributed.diagnostics import MemorySampler

# try to force flushing of memory
import gc, ctypes

import mitequinox.utils as ut
import mitequinox.parcels as pa

step_window_delayed = delayed(pa.step_window)

# ---- Run parameters

# root_dir = '/home1/scratch/aponte/parcels/'
root_dir = "/home/datawork-lops-osi/equinox/mit4320/parcels/"
#root_dir = '/home1/scratch/slgentil/parcels/'
#root_dir = '/home1/scratch/aponte/'

# 5x5 tiles dij=100 T=365 5jobs x 5workers
# run_name = "global_dij8_up3_r2s111_15j"
#run_name = "global_dij8_up3_r1s111_15j"
#run_name = "global_dij8_up3_r1s111_30j"

# will overwrite existing simulation
#overwrite = True
overwrite = False

### simulation parameters

## base case

# T = 360  # length of the total run [days]

# dt_window = timedelta(days=1.0)
# dt_outputs = timedelta(hours=1.0)
# dt_step = timedelta(hours=1.0)
# dt_seed = 10  # in days, base choice
# dt_seed = 0  # in days, base choice
# dt_reboot = timedelta(days=20.0)

# tile_size = dict(factor=(5, 10), overlap=(250, 250))
# init_dij = 50  # initial position subsampling compared to llc grid
# init_uplet = None # one number of parcels at each release location

# pclass = "extended" # TODO / uplet debug : do not interpolate all fields
# but pclass = "jit" may be broken, maybe not

# number of dask jobs launched for parcels simulations
# dask_jobs = 12 # 13?
# jobqueuekw = dict(processes=4, cores=4)

# following is not allowed on datarmor:
# dask_jobs = 12*4
# jobqueuekw=dict(processes=1, cores=1, memory="30GB",
#                resource_spec="select=1:ncpus=6:mem=30GB",
#               )

## uplet case

t_start = ut.np64toDate(np.datetime64('2012-02-01'))
run_name = "global_dij8_up3_r2s111_30j_201202"
#run_name = "global_dij8_up3_r4s111_30j_201202"
#run_name = "global_dij8_up3_r05s111_30j_201202"
#
#t_start = ut.np64toDate(np.datetime64('2012-05-01'))
#run_name = "global_dij8_up3_r2s111_30j_201205"
#
#t_start = ut.np64toDate(np.datetime64('2012-08-01'))
#run_name = "global_dij8_up3_r2s111_30j_201208"
#
#t_start = ut.np64toDate(np.datetime64('2011-12-01'))
#run_name = "global_dij8_up3_r2s111_30j_201112"

#T = 15  # length of the total run [days]
#T = 2  # length of the total run [days]
T = 30  # length of the total run [days]
dt_window = timedelta(days=1)
dt_outputs = timedelta(hours=1.0)
dt_step = timedelta(hours=1.0)
dt_seed = 0  # in days
dt_reboot = timedelta(days=3)

# smooth llc data
dt_smooth = None
dt_smooth = "12H"
#dt_smooth = "24H"
if dt_smooth is not None:
    run_name = run_name+"_smooth"+dt_smooth

tile_size = dict(factor=(6, 10), overlap=(150, 150)) # reduce size of tiles and decrease overlap

init_dij = 8  # initial position subsampling compared to llc grid
#init_uplet = (3, 4./111.) # initial number of parcels at each release location
init_uplet = (3, 2./111.) # initial number of parcels at each release location
#init_uplet = (3, 1./111.) # initial number of parcels at each release location
#init_uplet = (3, 0.5/111.) # initial number of parcels at each release location

pclass = "extended"
# pclass = "jit"  # uplet debug

dask_jobs = 8
jobqueuekw = dict(processes=2, cores=2) # uplet debug


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
    grd = ut.load_grd(V=["XC", "YC", "XG", "YG", "Depth"])
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
        tl = pa.tiler(ds=grd, **tile_size)
        # store tiler
        tl.store(dirs["tiling"])
    else:
        print("load existing tiling")
        logging.info("load existing tiling")
        tl = pa.tiler(tile_dir=dirs["tiling"])

    return tl


def format_info(step, t_start, t_end, timer_delta):
    print("-------------------------------------------")
    str_fmt = "step={}  /  start={}  /  end={} - elapse={:.1f} min".format(
        step,
        t_start.strftime("%Y-%m-%d %H:%M"),
        t_end.strftime("%Y-%m-%d %H:%M"),
        timer_delta/60.,
    )
    logging.info(str_fmt)


def run(dirs, tl, cluster, client):
    """ """

    # load dataset
    ds = load_llc()

    # set start and end times
    # t_start = ut.np64toDate(ds["time"][0].values)
    #t_start = ut.np64toDate(np.datetime64('2012-02-01'))
    #t_start = ut.np64toDate(np.datetime64('2012-05-01'))
    #t_start = ut.np64toDate(np.datetime64('2012-08-01'))
    #t_start = ut.np64toDate(np.datetime64('2011-12-01'))
    t_end = t_start + int(T * 86400 / dt_window.total_seconds()) * dt_window
    str_fmt = "Global start = {}  /  Global end = {}".format(
        t_start.strftime("%Y-%m-%d %H:%M"),
        t_end.strftime("%Y-%m-%d %H:%M"),
    )
    print(str_fmt)
    logging.info(str_fmt)

    # get new log filename for this run
    log_file = pa.name_log_file(dirs["run"])

    log = pa.browse_log_files(dirs["run"])
    if log:
        restart = max(list(log)) + 1
    else:
        restart = 0
    logging.info('restart='+str(restart))
    # clean up data for restart
    tl.clean_up(dirs["run"], restart)

    if restart == 0:
        global_parcel_number = 0
        local_numbers = {tile: 0 for tile in range(tl.N_tiles)}
        max_ids = {tile: None for tile in range(tl.N_tiles)}
    else:
        _log = log[restart - 1]
        global_parcel_number = _log["global_parcel_number"]
        # local_numbers = _log['local_numbers'] # TMP !!!
        max_ids = _log["max_ids"]

    # skips steps if restart
    step_t = list(enumerate(ut.dateRange(t_start, t_end, dt_window)))
    step_t = step_t[restart:]
    reboot = int(dt_reboot / dt_window)
    flag_out = False
    logging.info("reboot=%d, len(step_t)=%d", reboot, len(step_t))
    if reboot < len(step_t):
        step_t = step_t[:reboot]
        flag_out = True

    #timer_start = time()
    for step, local_t_start in step_t:

        local_t_end = local_t_start + dt_window + dt_step

        # print step info
        timer_start = time()
        #format_info(step, local_t_start, local_t_end, timer_stop-timer_start)
        format_info(step, local_t_start, local_t_end, 0)

        # load, tile (and store) llc data
        ds_tiles = pa.tile_store_llc(
            ds,
            slice(local_t_start, local_t_end, None),
            tl,
            persist=False,
            dt_smooth=dt_smooth,
        )
        _ = wait(ds_tiles)
        logging.debug("llc data loaded")
        # logging.debug("llc data persisted")

        # seed with more particles
        if dt_seed>0:
            seed = (local_t_start - t_start).days % dt_seed == 0
        else:
            seed = False

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
                init_uplet=init_uplet,
                parcels_remove_on_land=True,
                pclass=pclass,
                id_max=max_ids[tile],
                seed=seed,
            )
            for tile in range(tl.N_tiles)
        ]
        try:
            logging.debug("launching step_window distribution")
            #dsteps_out = dask.compute(*dsteps)
            dsteps_out = dask_compute_batch(dsteps, client) # TODO / uplet debug : launch in batches to use less workers
            # batch computation may not be functional yet

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
        # client.run(gc.collect)
        # client.run(trim_memory)  # should not be done systematically

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
        logging.info(str_fmt)
    
        # update timer
        timer_stop = time()
        format_info(step, local_t_start, local_t_end, timer_stop - timer_start)

    return flag_out


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def spin_up_cluster(jobs):
    logging.info("start spinning up dask cluster, jobs={}".format(jobs))
    cluster, client = ut.spin_up_cluster(
        "distributed",
        jobs=jobs,
        fraction=0.9,
        walltime="12:00:00",
        **jobqueuekw,
    )
    
    logging.info("dashboard via ssh: " + ut.dashboard_ssh_forward(client))
    return cluster, client


def close_dask(cluster, client):
    logging.info("starts closing dask cluster ...")
    try:
        
        client.close()
        logging.info("client closed ...")
        # manually kill pbs jobs
        manual_kill_jobs()
        logging.info("manually killed jobs ...")
        # cluster.close()
        # logging.info("cluster closed ...")
    except:
        logging.exception("cluster.close failed ...")
        # manually kill pbs jobs
        manual_kill_jobs()

    logging.info("... done")


def manual_kill_jobs():
    """manually kill dask pbs jobs"""

    import subprocess, getpass

    #
    username = getpass.getuser()
    #
    bashCommand = "qstat"
    try:
        output = subprocess.check_output(bashCommand, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    #
    for line in output.splitlines():
        lined = line.decode("UTF-8")
        if username in lined and "dask" in lined:
            pid = lined.split(".")[0]
            bashCommand = "qdel " + str(pid)
            logging.info(" " + bashCommand)
            try:
                boutput = subprocess.check_output(bashCommand, shell=True)
            except subprocess.CalledProcessError as e:
                # print(e.output.decode())
                pass

def dask_compute_batch(computations, client, batch_size=None):
    """ breaks down a list of computations into batches
    """
    # compute batch size according to number of workers
    if batch_size is None:
        # batch_size = len(client.scheduler_info()["workers"])
        batch_size = sum(list(client.nthreads().values()))
    # find batch indices
    total_range = range(len(computations))
    splits = max(1, np.ceil(len(total_range)/batch_size))
    batches = np.array_split(total_range, splits)
    # launch computations
    outputs = []
    for b in batches:
        logging.info("batches: " + str(b)+ " / "+str(total_range))
        out = dask.compute(*computations[slice(b[0], b[-1]+1)])
        outputs.append(out)
        
        # try to manually clean up memory
        # https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
        client.run(gc.collect)
        client.run(trim_memory)  # should not be done systematically
        
    return sum(outputs, ())

def main():

    # create run directory tree
    logging.info("1 - create_dir_tree")
    dirs = pa.create_dir_tree(root_dir, run_name, overwrite=overwrite)

    # create tiling
    logging.info("2 - generate_tiling")
    tl = generate_tiling(dirs, overwrite)

    # create run directories, erase them if need be
    logging.info("3 - create_tile_run_tree")
    tl.create_tile_run_tree(dirs["run"], overwrite=overwrite)

    logging.info("4 - start main loop")

    flag = True
    reboot = 0
    while flag:

        logging.info("--- reboot %d", reboot)

        # sping up cluster
        cluster, client = spin_up_cluster(dask_jobs)
        print(cluster)

        # run parcels simulation
        ms = MemorySampler()
        with performance_report(filename=f"dask-report-{reboot}.html"), ms.sample(f"reboot{reboot}"):
            flag = run(dirs, tl, cluster, client)
        # https://distributed.dask.org/en/latest/diagnosing-performance.html#analysing-memory-usage-over-time
        ms.to_pandas().to_csv(f"dask-memory-report-{reboot}.csv")

        # close dask
        close_dask(cluster, client)

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
        #level=logging.DEBUG,
    )
    # level order is: DEBUG, INFO, WARNING, ERROR
    # encoding='utf-8', # available only in latests python versions

    # dask memory logging:
    # https://blog.dask.org/2021/03/11/dask_memory_usage
    # log dask summary?

    main()
    # debug()

    logging.info("- all done")
