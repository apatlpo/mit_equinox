#!/usr/bin/env python
import os, sys
import socket
import subprocess
import json
from glob import glob
from time import sleep

if __name__ == '__main__':

    assert len(sys.argv)==3
    dashinfo = sys.argv[1]
    portdigit = sys.argv[2]

    jlab_port = '887%s' %portdigit
    dash_port = '878%s' %portdigit

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
    elif dashinfo!='0':
        nodefile = dashinfo

    if dashinfo!='0':
        dash = True
        with open(nodefile) as f:
            head = f.readline()
        bhost = head.split('.')[0]
    else:
        dash = False
        bhost = host

    cmd = ['jupyter', 'lab', '--ip', host, 
           '--no-browser', '--port', jlab_port, 
           '--notebook-dir', notebook_dir ]
    print(' '.join(cmd))
    #cmd = ['jupyter lab --ipp '+host+' --no-browser --port '+jlab_port+' --notebook-dir '+notebook_dir+' > "output.txt" ']
    
    #with open("jlab.output", ) 
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)

    print(f'ssh -N -L {jlab_port}:{host}:{jlab_port} '
              f'-L {dash_port}:{bhost}:8787 {user}@{hostname}')
    print('(Change the first port number if it is already used)')
    print('Then open the following URLs:')
    print(f'\tJupyter lab: http://localhost:{jlab_port}')
    print(f'\tDask dashboard: http://localhost:{dash_port}', flush=True)

    sleep(10)

    # this is not working at the moment
    for stdout_line in iter(popen.stdout.readline, ""):
        #yield stdout_line 
        print(stdout_line)
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


