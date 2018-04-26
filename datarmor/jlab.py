#!/usr/bin/env python
import os, sys
import socket
import subprocess
import json
from glob import glob

if __name__ == '__main__':

    jlab_port = '8877'
    dash_port = '8787'

    notebook_dir = os.environ['HOME']
    user = os.environ['USER']

    host = socket.gethostname() # where jlab will be running
    hostname = 'datarmor1-10g'

    if True:
        while not glob('*.nodefile'):
            pass
        nodefile = glob('*.nodefile')[-1]
        with open(nodefile) as f:
            head = f.readline()
        bhost = head.split('.')[0]

    if False:
        scheduler = os.environ['DATAWORK']+'/dask/scheduler.json'
        print(scheduler)
        while not os.path.isfile(scheduler):
            pass
        sdata = json.load(open(scheduler))
        a = sdata['address'].replace('tcp','http')
        a = a[:a.rindex(':')+1]+str(sdata['services']['bokeh'])
        print(a)
        
    cmd = ['jupyter', 'lab', '--ip', host, 
           '--no-browser', '--port', jlab_port, 
           '--notebook-dir', notebook_dir]
    print(' '.join(cmd))
    proc = subprocess.Popen(cmd)

    print(f'ssh -N -L {jlab_port}:{host}:{jlab_port} '
          f'-L {dash_port}:{bhost}:8787 {user}@{hostname}')
    print('Then open the following URLs:')
    print(f'\tJupyter lab: http://localhost:{jlab_port}')
    print(f'\tDask dashboard: http://localhost:{dash_port}', flush=True)

