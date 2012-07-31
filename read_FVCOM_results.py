
def readFVCOM(file, varList, noisy=False):
    """
    Read in the FVCOM results file and spit out numpy arrays for
    each of the variables.
    """

    from netCDF4 import Dataset, MFDataset

    rootgrp = Dataset(file, 'r')
    mfdata = MFDataset(file)

    if noisy:
        print "File format: " + rootgrp.file_format

    FVCOM = {}
    for key, var in rootgrp.variables.items():
        if noisy:
            print 'Found ' + key,

        if key in varList:
            if noisy:
                print '(extracted)'
            FVCOM[key] = mfdata.variables[key][:]
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

    import numpy as np

    nt, nx = np.shape(Z)

    surfaceElevation = np.empty([nt,np.shape(idx)[0]])
    for cnt, i in enumerate(idx):
        surfaceElevation[:,cnt] = Z[:,i]

    return surfaceElevation
