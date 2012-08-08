
def readFVCOM(file, varList, noisy=False):
    """
    Read in the FVCOM results file and spit out numpy arrays for
    each of the variables.
    """

    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError('Failed to load the NetCDF4 library')

    rootgrp = Dataset(file, 'r')

    if noisy:
        print "File format: " + rootgrp.file_format

    FVCOM = {}
    for key, var in rootgrp.variables.items():
        if noisy:
            print 'Found ' + key,

        if key in varList:
            if noisy:
                print '(extracted)'
            FVCOM[key] = rootgrp.variables[key][:]
        else:
            if noisy:
                print

    return FVCOM

def getSurfaceElevation(Z, idx):
    """
    Extract the surface elevation from Z at index ind. If ind is multiple
    values, extract and return the surface elevations at all those locations.

    Z is usually extracted from the dict created when using readFVCOM() on a
    NetCDF file.

    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError('NumPy not found')

    nt, nx = np.shape(Z)

    surfaceElevation = np.empty([nt,np.shape(idx)[0]])
    for cnt, i in enumerate(idx):
        if not np.isnan(i):
            surfaceElevation[:,cnt] = Z[:,i]

    return surfaceElevation
