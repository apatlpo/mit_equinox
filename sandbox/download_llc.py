import wget
import os

V = ['oceTAUX', 'oceTAUY', 'KPPhbl']

I = range(10368, 1495008+1,144)

i=10368
v = V[0]

for v in V:
    for i in I:
        file = '%s.%.10d.data.shrunk'%(v,i)
        url = 'https://data.nas.nasa.gov/ecco/download_data.php?file=/' \
                +'eccodata/llc_4320/compressed/%.10d/'%i \
                +file
        if not os.path.isfile('./'+file):
            wget.download(url, './'+file)
        