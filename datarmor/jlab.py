#!/usr/bin/env python
import os, sys
import socket
import subprocess

if __name__ == '__main__':

    jlab_port = '8877'
    dash_port = '8787'

    notebook_dir = os.environ['HOME']
    user = os.environ['USER']

    host = socket.gethostname() # where jlab will be running
    hostname = 'datarmor1-10g'

    cmd = ['jupyter', 'lab', '--ip', host, 
           '--no-browser', '--port', jlab_port, 
           '--notebook-dir', notebook_dir]
    print(' '.join(cmd))
    proc = subprocess.Popen(cmd)

    print(f'ssh -N -L {jlab_port}:{host}:{jlab_port} '
          f'-L {dash_port}:{host}:8787 {user}@{hostname}')
    print('Then open the following URLs:')
    print(f'\tJupyter lab: http://localhost:{jlab_port}')
    print(f'\tDask dashboard: http://localhost:{dash_port}', flush=True)

