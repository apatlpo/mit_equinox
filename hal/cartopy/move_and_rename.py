
import sys, shutil
import cartopy
import zipfile
from glob import glob

pref = 'ne_'

# unzip and rename
zfile = sys.argv[1]
outdir = cartopy.config['data_dir']+'/shapefiles/natural_earth/physical/'

zip = zipfile.ZipFile(zfile)
zip.extractall(outdir)
zip.close()

# for old cartopy version, you need to get rid ot ne_ prefix 
files = glob(outdir+zfile.strip('zip')+'*')
for f in files:
    print(f)
    #shutil.move(f,f.replace(pref,''))

print(outdir)




