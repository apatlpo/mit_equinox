import subprocess, getpass
#
username = getpass.getuser()
print(username)
#
bashCommand = 'qstat'
output = subprocess.check_output(bashCommand, shell=True)
#
for line in output.splitlines():
    lined = line.decode('UTF-8')
    if username in lined and 'dask' in lined:
        print(lined)
        pid = lined.split('.')[0]
        bashCommand = 'qdel '+str(pid)
        boutput = subprocess.check_output(bashCommand, shell=True)

