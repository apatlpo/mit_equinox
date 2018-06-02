#!/usr/bin/env python
import os, sys
import socket
import subprocess
import json
from glob import glob

if __name__ == '__main__':

    assert len(sys.argv)==2
    dashinfo = sys.argv[1]

    jlab_port = '8877'
    dash_port = '8787'

    notebook_dir = os.environ['HOME']
    user = os.environ['USER']

    host = socket.gethostname() # where jlab will be running
    hostname = 'datarmor1-10g'

    if 'wait' in dashinfo:
        print('wait in dashinfo')

    if 'wait' in dashinfo:
        while not glob('*.nodefile'):
            pass
        nodefile = glob('*.nodefile')[-1]
    elif dashinfo is not '0':
        nodefile = dashinfo

    if dashinfo is not '0':
        dash = True
        with open(nodefile) as f:
            head = f.readline()
        bhost = head.split('.')[0]
    else:
        dash = False

    cmd = ['jupyter', 'lab', '--ip', host, 
           '--no-browser', '--port', jlab_port, 
           '--notebook-dir', notebook_dir]
    print(' '.join(cmd))
    proc = subprocess.Popen(cmd)

    if dash:
        print(f'ssh -N -L {jlab_port}:{host}:{jlab_port} '
              f'-L {dash_port}:{bhost}:8787 {user}@{hostname}')
    else:
        print(f'ssh -N -L {jlab_port}:{host}:{jlab_port} '
              f' {user}@{hostname}')
    print('(Change the first port number if it is already used)')
    print('Then open the following URLs:')
    print(f'\tJupyter lab: http://localhost:{jlab_port}')
    if dash:
        print(f'\tDask dashboard: http://localhost:{dash_port}', flush=True)

