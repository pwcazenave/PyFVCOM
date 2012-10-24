
def readFVCOM(file, varList, clipTime=False, noisy=False):
    """
    Read in the FVCOM results file and spit out numpy arrays for
    each of the variables specified in the varList list.

    Optionally specify a timeRange which will extract only the range of times
    of interest. Specify as indices and not Modified Julian Date or Gregorian
    date. Give a start and end index as a list (e.g. [0, 200]).

    """

    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError('Failed to load the NetCDF4 library')

    rootgrp = Dataset(file, 'r')

    if noisy:
        print "File format: " + rootgrp.file_format

        if clipTime is not False:
            print 'Clipping time from index {:.0f} to {:.0f}'.format(clipTime[0], clipTime[1])

    FVCOM = {}
    for key, var in rootgrp.variables.items():
        if noisy:
            print 'Found ' + key,

        if key in varList:
            if noisy:
                print '(extracted)'

            # Default to any variable not having a time dimension.
            hasTime = False

            if clipTime is not False:
                # Check the current variable dimensions to see if it has a time
                # dimension.

                for dim in rootgrp.variables[key].dimensions:
                    if str(dim) == 'time':
                        hasTime = True

            if hasTime:
                # Since time is an unlimited dimension, it will be listed
                # first. That means if we specify a range for a
                # multidimensional variable, only the first variable will be
                # clipped, the others will be output in their entirety.
                FVCOM[key] = rootgrp.variables[key][clipTime[0]:clipTime[1]]
            else:
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
