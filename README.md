# equinox mit
Contains code that process mitgcm output for the EQUINOx project

- sandbox/ : first trials with xmitgcm, basic stuff

- doc/ : conda and git info

---
## install

All scripts require python librairies that may be installed with conda according to the following instructions [here](https://github.com/apatlpo/mit_equinox/blob/master/doc/CONDA.md)

---
## run on datarmor:

After having installed all libraries, and cloned this repository, go into `mit_equinox/datarmor`.

### method 1 :

For 8 nodes for example:
```
./launch-dask.sh 8
./launch-jlab.sh wait
```

Follow instructions that pop up from there

Once you are done computing, kill the relevant jobs.

Clean up after computations: `./clean.sh`

### method 2:

```
./launch-jlab.sh
```

Follow instructions that pop up from there.

The spin up of dask relies on dask-jobqueue:
```
... example ...
```

Kill jobs once done with computations. 
`python kill.py` may be used.

Clean up after computations: `./clean.sh`


