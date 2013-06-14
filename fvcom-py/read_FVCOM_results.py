
def readFVCOM(file, varList, clipDims=False, noisy=False):
    """
    Read in the FVCOM results file and spit out numpy arrays for each of the
    variables specified in the varList list.

    Optionally specify a dict with keys whose names match the dimension names
    in the NetCDF file and whose values are strings specifying alternative
    ranges or lists of indices. For example, to extract the first hundred time
    steps, supply clipDims as:

        clipDims = {'time':'0:100'}

    To extract the first, 400th and 10,000th values of any array with nodes:

        clipDims = {'node':['0, 3999, 9999']}

    Any dimension not given in clipDims will be extracted in full.

    Parameters
    ----------

    file : str
        Full path to an FVCOM NetCDF output file.
    varList : list
        List of variable names to be extracted.
    clipDims : dict, optional
        Dict whose keys are dimensions and whose values are a string of either
        a range (e.g. {'time':'0:100'}) or a list of individual indices (e.g.
        {'time':'[0, 1, 80, 100]'}).
    noisy : bool
        Set to True to enable verbose output.

    Returns
    -------

    FVCOM : dict
        Dict of data extracted from the NetCDF file. Keys are those given in
        varList and the data are stored as ndarrays.

    """

    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError('Failed to load the NetCDF4 library')

    import sys

    rootgrp = Dataset(file, 'r')

    # Create a dict of the dimension names and their current sizes
    dims = {}
    for key, var in rootgrp.dimensions.iteritems():
        # Make the dimensions ranges so we can use them to extract all the
        # values.
        dims[key] = '0:' + str(len(var))

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
            sys.stdout.flush()

        if key in varList:
            vDims = rootgrp.variables[key].dimensions

            toExtract = [dims[d] for d in vDims]

            # I know, I know, eval() is evil.
            getData = 'rootgrp.variables[\'{}\']{}'.format(key,str(toExtract).replace('\'', ''))
            FVCOM[key] = eval(getData)

            if noisy:
                if len(str(toExtract)) < 60:
                    print '(extracted {})'.format(str(toExtract).replace('\'', ''))
                else:
                    print '(extracted given indices)'

        elif noisy:
                print

    return FVCOM


def elems2nodes(elems, tri, nvert, noisy=False):
    """
    Calculate a nodal value based on the average value for the elements
    of which it a part. This necessarily involves an average, so the
    conversion from nodes2elems and elems2nodes is not necessarily
    reversible.

    Parameters
    ----------

    elems : ndarray
        Array of unstructured grid element values to move to the element
        nodes.
    tri : ndarray
        Array of shape (nelem, 3) comprising the list of connectivity
        for each element.
    nvert : int
        Number of nodes (vertices) in the unstructured grid.

    Returns
    -------

    nodes : ndarray
        Array of values at the grid nodes.

    """

    try:
        import numpy as np
    except ImportError:
        raise ImportError('NumPy not found')

    count = np.zeros(nvert, dtype=int)

    # Deal with 1D and 2D element arrays separately
    if np.ndim(elems) == 1:
        nodes = np.zeros(nvert)
        for i, indices in enumerate(tri):
            n0, n1, n2 = indices
            nodes[n0] = nodes[n0] + elems[i]
            nodes[n1] = nodes[n1] + elems[i]
            nodes[n2] = nodes[n2] + elems[i]
            count[n0] = count[n0] + 1
            count[n1] = count[n1] + 1
            count[n2] = count[n2] + 1

    elif np.ndim(elems) == 2:
        nodes = np.zeros((np.shape(elems)[0], nvert))
        for i, indices in enumerate(tri):
            n0, n1, n2 = indices
            nodes[:, n0] = nodes[:, n0] + elems[:, i]
            nodes[:, n1] = nodes[:, n1] + elems[:, i]
            nodes[:, n2] = nodes[:, n2] + elems[:, i]
            count[n0] = count[n0] + 1
            count[n1] = count[n1] + 1
            count[n2] = count[n2] + 1
    else:
        raise 'Too many dimensions (maximum of two)'

    # Now calculate the average for each node based on the number of
    # elements of which it is a part.
    nodes = nodes / count

    return nodes


def nodes2elems(nodes, tri, noisy=False):
    """
    Calculate a element centre value based on the average value for the
    nodes from which it is formed. This necessarily involves an average,
    so the conversion from nodes2elems and elems2nodes is not
    necessarily reversible.

    Parameters
    ----------

    nodes : ndarray
        Array of unstructured grid node values to move to the element
        centres.
    tri : ndarray
        Array of shape (nelem, 3) comprising the list of connectivity
        for each element.

    Returns
    -------

    elems : ndarray
        Array of values at the grid nodes.

    """

    nvert = np.shape(tri)[0]

    if np.ndim(elems) == 1:
        elems = np.zeros(nvert)
        for i, indices in enumerate(tri):
            elems[i] = np.mean(nodes[indices])

    elif np.ndim(elems) == 2:
        elems = np.zeros((np.shape(nodes)[0], nvert))
        for i, indices in enumerate(tri):
            elems[:, i] = np.mean(nodes[:, indices])
    else:
        raise 'Too many dimensions (maximum of two)'


def getSurfaceElevation(Z, idx):
    """
    Extract the surface elevation from Z at index ind. If ind is multiple
    values, extract and return the surface elevations at all those locations.

    Z is usually extracted from the dict created when using readFVCOM() on a
    NetCDF file.

    Parameters
    ----------

    Z : ndarray
        Unstructured array of surface elevations with time.
    idx : list
        List of indices from which to extract time series of surface
        elevations.

    Returns
    -------

    surfaceElevation : ndarray
        Time series of surface elevations at the indices supplied in
        idx.

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
