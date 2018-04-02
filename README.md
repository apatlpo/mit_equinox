# equinox mit
Contains code that process mitgcm output for the EQUINOx project

- sandbox/ : first trials with xmitgcm, basic stuff

- doc/ : conda and git info

---
## install

All scripts require python librairies that may be installed with conda according to the following instructions [here](https://github.com/apatlpo/mit_equinox/blob/master/doc/CONDA.md)

---
## run on datarmor:

After having installed all libraries, and cloned this repository, go into `mit_equinox/datarmor`
and run:
```
./launch-openq.sh
```
Follow instructions that pop up from there.

---
## misc

Simple SSH port forward from slyne:
```
ssh -N -L 8888:localhost:8888 slyne
```

