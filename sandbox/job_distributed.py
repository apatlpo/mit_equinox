import os, shutil
import logging
from time import time

import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime

import dask
from dask.delayed import delayed
from dask.distributed import performance_report, wait
from distributed.diagnostics import MemorySampler

# try to force flushing of memory
import gc, ctypes


# ---- Run parameters

root_dir = "/home/datawork-lops-osi/equinox/mit4320/parcels/"
run_name = "global_dij8_up3_r2s111_30j_201202"

# will overwrite existing results
#overwrite = True
overwrite = False

# dask parameters
dask_jobs = 8
jobqueuekw = dict(processes=2, cores=2) # uplet debug


# ---------------------------------- dask utils ---------------------------------- 

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



# ---------------------------------- core of the job to be done ---------------------------------- 

def run(v, cluster, client):
    """ """

    # load dataset
    # ds = ...

    # set start and end times
    #t_end = t_start + int(T * 86400 / dt_window.total_seconds()) * dt_window
    #str_fmt = "Global start = {}  /  Global end = {}".format(
    #    t_start.strftime("%Y-%m-%d %H:%M"),
    #    t_end.strftime("%Y-%m-%d %H:%M"),
    #)
    str_fmt = "some information in run"
    logging.info(str_fmt)

    # do work

    flat_out = True
    return flag_out


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

    # create run directory tree
    logging.info("1 - step 1")
    # ...
    
    # create tiling
    logging.info("2 - step 2")
    # ...

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
            flag = run(None, cluster, client)
        # https://distributed.dask.org/en/latest/diagnosing-performance.html#analysing-memory-usage-over-time
        ms.to_pandas().to_csv(f"dask-memory-report-{reboot}.csv")

        # close dask
        close_dask(cluster, client)

        reboot += 1

    logging.info("- all done")
