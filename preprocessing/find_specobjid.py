import h5py 
import numpy as np




with h5py.File('data/sdss_data.h5') as f:
    for key in f.keys():
        print(key)
        
    filenames = f['filenames_all'][:]
    filenames = [name[0].decode('utf-8') for name in filenames]
    
    query = "733-56575-0753.fits"
    
    matches = matches = [i for i, s in enumerate(filenames) if query in s]
    
    print(matches)
    print(f['specobjid_all'][matches])
