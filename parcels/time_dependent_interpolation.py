import os
from glob import glob
import logging

import numpy as np
import xarray as xr
import pandas as pd

import dask.dataframe as dd
from dask import delayed
import dask
from dask.distributed import wait, performance_report
# from distributed.diagnostics import MemorySampler # requires latest dask ... wait for finishing this production run

from tqdm import tqdm

import mitequinox.utils as ut
import mitequinox.plot as pl
import mitequinox.parcels as pa

import pyinterp


# ---- Run parameters

# Eulerian data to interpolate
v = "Eta"

# Drifter data
root_dir = "/home/datawork-lops-osi/equinox/mit4320/parcels/"
run_name = "global_T365j_dt1j_dij50"

# output
output_dir = "/home1/scratch/aponte/interp"

# will overwrite existing simulation
# overwrite = True
overwrite = False

# number of dask jobs launched for parcels simulations
dask_jobs = 12
jobqueuekw = dict(processes=1, cores=1)

# derived variables
nprocesses = dask_jobs*jobqueuekw["processes"]
#log_dir = os.path.join(output_dir, "logs")
_fmt = "%Y%m%d%H%M"

def spin_up_cluster(jobs):
    #print("start spinning up dask cluster, jobs={}".format(jobs))
    logging.info("start spinning up dask cluster, jobs={}".format(jobs))
    cluster, client = ut.spin_up_cluster(
        "distributed",
        jobs=jobs,
        fraction=0.9,
        walltime="36:00:00",
        **jobqueuekw,
    )
    logging.info("dashboard via ssh: " + ut.dashboard_ssh_forward(client))
    return cluster, client


def close_dask(cluster, client):
    #print("starts closing dask cluster ...")
    logging.info("starts closing dask cluster ...")
    try:
        cluster.close()
    except:
        #print("cluster.close failed ")
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
            #print(bashCommand)
            logging.info(" " + bashCommand)
            try:
                boutput = subprocess.check_output(bashCommand, shell=True)
            except subprocess.CalledProcessError as e:
                # print(e.output.decode())
                pass
            
def check_directory(
    directory,
    create=True,
):
    """Check existence of a directory and create it if necessary

    Parameters
    ----------
    directory: str
        Path of directory
    create: boolean
        Create directory if not there, default is True
    """
    # create diagnostics dir if not present
    if not os.path.isdir(directory):
        if create:
            # need to create the directory
            os.mkdir(directory)
            print("Create new diagnostic directory {}".format(directory))
        else:
            raise OSError("Directory does not exist")
    return directory

def get_last_processed_date():
    parquets = sorted(glob(os.path.join(output_dir, "*.parquet")))
    if parquets:
        return max([pd.datetime(p.split("/")[-1].split("_")[-1], _fmt) for p in parquets])
    
def write_data(df, t_start, t_end):
    """ tells whether log file exists
    """
    parquet = t_start.strftime(_fmt)+"_"+t_end.strftime(_fmt)+".parquet"
    df.to_parquet(os.path.join(output_dir, parquet))
    return parquet

def interp_low(da, df, **kwargs):
    """ low level part that interpolate the field
    
    https://pangeo-pyinterp.readthedocs.io/en/latest/generated/pyinterp.RTree.inverse_distance_weighting.html#pyinterp.RTree.inverse_distance_weighting
    """
    mesh = pyinterp.RTree()
    mesh.packing(np.vstack((da.XC.values.flatten(), 
                            da.YC.values.flatten())
                          ).T,
                 da.values.flatten(),
    )

    dkwargs = dict(within=True, radius=None, k=8, num_threads=0)
    # num_threads=0 uses all cpus, set to 1 if you don"t want this
    dkwargs.update(**kwargs)
    idw_eta, neighbors = mesh.inverse_distance_weighting(
        np.vstack((df["lon"], df["lat"])).T,
        **dkwargs
    )
    df_out = df["trajectory"].to_frame()
    df_out.loc[:, da.name+"_interpolated"] = idw_eta[:]
    return df_out

def interp(da, df, **kwargs):
    with dask.config.set(schedulers="threads"):
        if "XC" not in da.coords:
            grd = ut.load_grd(V=["XC", "YC"])
            da = da.assign_coords(XC=grd["XC"], YC=grd["YC"])
        return interp_low(da, df, **kwargs)

def split_process(s, df, ds, client):
    """ wrapping in methods helps with garbage collection ...
    """
    df_split = df.loc[str(s[0]):str(s[-1])].repartition(npartitions=nprocesses).persist()
    _ = wait(df_split)
    da = ds[v].sel(time=list(s)).chunk(dict(time=1, face=-1, i=-1, j=-1))
    
    values = [delayed(interp)(da.sel(time=t), df_split.loc[str(t)]) for t in s]
    futures = client.compute(values)
    results = client.gather(futures)
    
    df_interp = (dd.concat([dd.from_pandas(r, npartitions=1) for r in results])
             .repartition(npartitions=1)
             .persist()
            )
    return df_interp

def main():

    # create directories if not existent:
    check_directory(output_dir)
    #check_directory(log_dir)
    
    # sping up cluster
    #print("1 - spin up cluster")
    logging.info("1 - spin up cluster")
    cluster, client = spin_up_cluster(dask_jobs)
    #print(cluster)
    
    # create run directory tree
    logging.info("2 - load data")
    
    # Eulerian data
    ds = ut.load_data_zarr(v)
    grd = ut.load_grd(V=["XC", "YC"])

    # Lagrangian data
    parcels_index = "time"
    df = pa.parcels_output(root_dir+run_name, parquets=[parcels_index])[parcels_index]
    df = pa.degs2ms(df)
    
    # load timeline
    t = df.index.unique().compute().dropna().to_numpy()
    # note: there are NaN in the time series
    logging.info(f"{len(t)} iterations to perform")

    # get last processed date
    date_max = get_last_processed_date()

    #nprocesses = 2
    #nprocesses = len(client.scheduler_info()["workers"])
    splits = np.array_split(t, len(t)//(nprocesses-1))
    if date_max:
        splits = [s for s in splits if pd.to_datetime(s[0])>date_max]
    
    # run parcels simulation
    # ms = MemorySampler()
    #with performance_report(filename=f"dask-report-{reboot}.html"):
    #    flag = run(dirs, tl, cluster, client)
    # with performance_report(filename=f"dask-report-{reboot}.html"), ms.sample(f"reboot{reboot}")
    # MemorySampler requires latest dask ... wait for finishing this production run
    # https://distributed.dask.org/en/latest/diagnosing-performance.html#analysing-memory-usage-over-time
    # ms.to_pandas().to_netcdf ...
    
    #main loop
    try:
        
        D = []
        n=0
        for s in splits:
            if n==0:
                t_start=pd.to_datetime(s[0])
            
            _start = pd.to_datetime(s[0]).strftime(_fmt)
            _end = pd.to_datetime(s[-1]).strftime(_fmt)
            logging.info(f"step {_start} {_end}: start")
            
            # implement restart process
            #D.append(float(ds[v].sel(time=list(s)).isel(face=1).mean()))
            D.append(split_process(s, df, ds, client))

            logging.info(f"step {_start} {_end}: end")            
            # store if need be
            n+=1
            if n%10==0:
                df_interp = dd.concat(D).repartition(partition_size="100MB")
                parquet = write_data(df_interp, t_start, pd.to_datetime(s[-1]))
                logging.info(f"parquet stored: {parquet}")
                D=[]
                n=0
                
    except:
        logging.exception("main loop failed ...")
    
    # below may have to be moved out
    
    # concatenate into a full dataframe prior to storage
    # read individual archives if necesary
    #df_interp = dd.concat(D).repartition(partition_size="100MB").persist()
    #df_interp.head()
    #df_final = dd.merge(df, df_interp, on=["time", "trajectory"], how="inner").persist()
    
    # close dask
    close_dask(cluster, client)


if __name__ == "__main__":

    # to std output
    # logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
    # to file
    logging.basicConfig(
        filename="distributed.log",
        level=logging.INFO,
    )
    # level order is: DEBUG, INFO, WARNING, ERROR
    # encoding="utf-8", # available only in latests python versions

    # dask memory logging:
    # https://blog.dask.org/2021/03/11/dask_memory_usage
    # log dask summary?

    main()

    print("- all done")
    logging.info("- all done")
