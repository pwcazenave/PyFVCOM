
def readFVCOM(file, varList, clipDims=False, noisy=False):
    """
    Read in the FVCOM results file and spit out numpy arrays for
    each of the variables specified in the varList list.

    Optionally specify a dict which keys whose names match the dimensions in
    the NetCDF file and whose values are strings specifying alternative ranges
    or lists of indices. For example, to extract the first hundred time steps,
    supply clipDims as:

        clipDims = {'time':'0:99'}

    To extract the first, 400th and 10,000th values of any array with nodes:

        clipDims = {'node':['0, 399, 9999']}

    To improve performance, sort the nodes in the dict otherwise lookups from
    the NetCDF file will be slow.

    Any dimension not given in clipDims will be extracted in full.

    """

    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError('Failed to load the NetCDF4 library')

    rootgrp = Dataset(file, 'r')

    # Create a dict of the dimension names and their current sizes
    dims = {}
    for key, var in rootgrp.dimensions.iteritems():
        # Make the dimensions ranges so we can use them to extract all the
        # values.
        dims[key] = '0:' + str(len(var)-1)

    # Compare the dimensions in the NetCDF file with those provided. If we've
    # been given a dict of dimensions which differs from those in the NetCDF
    # file, then use those.
    if clipDims is not False:
        commonKeys = set(dims).intersection(clipDims.keys())
        for k in commonKeys:
            dims[k] = clipDims[k]

    if noisy:
        print "File format: " + rootgrp.file_format

    FVCOM = {}
    for key, var in rootgrp.variables.iteritems():
        if noisy:
            print 'Found ' + key,

        if key in varList:
            vDims = rootgrp.variables[key].dimensions

            toExtract = []
            [toExtract.append(dims[d]) for d in vDims]

            # I know, I know, eval() is evil.
            getData = 'rootgrp.variables[\'{}\']{}'.format(key,str(toExtract).replace('\'', ''))
            FVCOM[key] = eval(getData)

            if noisy:
                if len(str(toExtract)) < 60:
                    print '(extracted {})'.format(str(toExtract).replace('\'', ''))
                else:
                    print '(extracted given nodes)'

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
