from __future__ import print_function

def readFVCOM(file, varList=None, clipDims=False, noisy=False, atts=False):
    """
    Read in the FVCOM results file and spit out numpy arrays for each of the
    variables specified in the varList list.

    Optionally specify a dict with keys whose names match the dimension names
    in the NetCDF file and whose values are strings specifying alternative
    ranges or lists of indices. For example, to extract the first hundred time
    steps, supply clipDims as:

        clipDims = {'time':'0:100'}

    To extract the first, 400th and 10,000th values of any array with nodes:

        clipDims = {'node':'[0, 3999, 9999]'}

    Any dimension not given in clipDims will be extracted in full.

    Specify atts=True to extract the variable attributes.

    Parameters
    ----------
    file : str, list
        If a string, the full path to an FVCOM NetCDF output file. If a list,
        a series of files to be loaded. Data will be concatenated into a single
        dict.
    varList : list, optional
        List of variable names to be extracted. If omitted, all variables are
        returned.
    clipDims : dict, optional
        Dict whose keys are dimensions and whose values are a string of either
        a range (e.g. {'time':'0:100'}) or a list of individual indices (e.g.
        {'time':'[0, 1, 80, 100]'}). Slicing is supported (::5 for every fifth
        value) but it is not possible to extract data from the end of the array
        with a negative index (e.g. 0:-4).
    noisy : bool, optional
        Set to True to enable verbose output.
    atts : bool, optional
        Set to True to enable output of the attributes (defaults to False).

    Returns
    -------
    FVCOM : dict
        Dict of data extracted from the NetCDF file. Keys are those given in
        varList and the data are stored as ndarrays.
    attributes : dict, optional
        If atts=True, returns the attributes as a dict for each
        variable in varList. The key 'dims' contains the array dimensions (each
        variable contains the names of its dimensions) as well as the shape of
        the dimensions defined in the NetCDF file.

    See Also
    --------
    readProbes : read in FVCOM ASCII probes output files.

    """

    try:
        from netCDF4 import Dataset, MFDataset
    except ImportError:
        raise ImportError('Failed to load the NetCDF4 library')

    import sys

    # If we have a list, assume it's lots of files and load them all.
    if isinstance(file, list):
        try:
            try:
                rootgrp = MFDataset(file, 'r')
            except IOError as e:
                raise IOError('Unable to open file {}. Aborting.'.format(file))
        except:
            # Try aggregating along a 'time' dimension (for POLCOMS, for example)
            try:
                rootgrp = MFDataset(file, 'r', aggdim='time')
            except IOError as e:
                raise IOError('Unable to open file {}. Aborting.'.format(file))

    else:
        rootgrp = Dataset(file, 'r')


    # Create a dict of the dimension names and their current sizes
    dims = {}
    for key, var in list(rootgrp.dimensions.items()):
        # Make the dimensions ranges so we can use them to extract all the
        # values.
        dims[key] = '0:' + str(len(var))

    # Compare the dimensions in the NetCDF file with those provided. If we've
    # been given a dict of dimensions which differs from those in the NetCDF
    # file, then use those.
    if clipDims is not False:
        commonKeys = set(dims).intersection(list(clipDims.keys()))
        for k in commonKeys:
            dims[k] = clipDims[k]

    if noisy:
        print("File format: {}".format(rootgrp.file_format))

    if not varList:
        varList = iter(list(rootgrp.variables.keys()))

    FVCOM = {}

    # Save the dimensions in the attributes dict.
    if atts:
        attributes = {}
        attributes['dims'] = dims

    for key, var in list(rootgrp.variables.items()):
        if noisy:
            print('Found ' + key, end=' ')
            sys.stdout.flush()

        if key in varList:
            vDims = rootgrp.variables[key].dimensions

            toExtract = [dims[d] for d in vDims]

            # I know, I know, eval() is evil.
            getData = 'rootgrp.variables[\'{}\']{}'.format(key,str(toExtract).replace('\'', ''))
            FVCOM[key] = eval(getData)

            # Add the units and dimensions for this variable to the list of
            # attributes.
            if atts:
                attributes[key] = {}
                try:
                    attributes[key]['units'] = rootgrp.variables[key].units
                except:
                    pass

                try:
                    attributes[key]['dims'] = rootgrp.variables[key].dimensions
                except:
                    pass

            if noisy:
                if len(str(toExtract)) < 60:
                    print('(extracted {})'.format(str(toExtract).replace('\'', '')))
                else:
                    print('(extracted given indices)')

        elif noisy:
                print()

    # Close the open file.
    rootgrp.close()

    if atts:
        return FVCOM, attributes
    else:
        return FVCOM


def ncread(file, vars=None, dims=False, noisy=False, atts=False):
    """
    Read in a netCDF file and return numpy arrays for each of the variables
    specified in the vars list.

    Optionally specify a dict with keys whose names match the dimension names
    in the NetCDF file and whose values are strings specifying alternative
    ranges or lists of indices. For example, to extract the first hundred time
    steps, supply clipDims as:

        dims = {'time':'0:100'}

    To extract the first, 400th and 10,000th values of any array with nodes:

        dims = {'node':'[0, 3999, 9999]'}

    Any dimension not given in dims will be extracted in full.

    Specify atts=True to extract attributes.

    Parameters
    ----------
    file : str, list
        If a string, the full path to an FVCOM NetCDF output file. If a list,
        a series of files to be loaded. Data will be concatenated into a single
        dict.
    vars : list, optional
        List of variable names to be extracted. If omitted, all variables are
        returned.
    dims : dict, optional
        Dict whose keys are dimensions and whose values are a string of either
        a range (e.g. {'time':'0:100'}) or a list of individual indices (e.g.
        {'time':'[0, 1, 80, 100]'}). Slicing is supported (::5 for every fifth
        value) but it is not possible to extract data from the end of the array
        with a negative index (e.g. 0:-4).
    noisy : bool, optional
        Set to True to enable verbose output.
    atts : bool, optional
        Set to True to enable output of the attributes (defaults to False).

    Returns
    -------
    nc : dict
        Dict of data extracted from the NetCDF file. Keys are those given in
        varList and the data are stored as ndarrays.
    attributes : dict, optional
        If True, returns the attributes as a dict for each variable in varList.
        The key 'dims' contains the array dimensions (each variable contains
        the names of its dimensions) as well as the shape of the dimensions
        defined in the NetCDF file.

    Notes
    -----
    This is actually a wrapper for the readFVCOM function, but since that
    function is actually not specific to FVCOM, it seemed sensible to have this
    generic function. Eventually I imagine this will be the underlying version
    and the readFVCOM function will call this one i.e. the roles will be
    swapped.

    """

    if atts:
        nc, attributes = readFVCOM(file, varList=vars, clipDims=dims, noisy=noisy, atts=atts)
        return nc, attributes
    else:
        nc = readFVCOM(file, varList=vars, clipDims=dims, noisy=noisy, atts=atts)
        return nc


def readProbes(files, noisy=False):
    """
    Read in FVCOM probes output files. Reads both 1 and 2D outputs. Currently
    only sensible to import a single station with this function since all data
    is output in a single array.

    Parameters
    ----------
    files : list
        List of file paths to load.
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    times : ndarray
        Modified Julian Day times for the extracted time series.
    values : ndarray
        Array of the extracted time series values.

    See Also
    --------
    readFVCOM : read in FVCOM netCDF output.

    TODO
    ----

    Add support to multiple sites with a single call. Perhaps returning a dict
    with the keys based on the file name is most sensible here?

    """

    try:
        import numpy as np
    except ImportError:
        raise ImportError('Unable to load NumPy.')

    if len(files) == 0:
        raise Exception('No files provided.')

    if not isinstance(files, list):
        files = [files]

    for i, file in enumerate(files):
        if noisy: print('Loading file {} of {}...'.format(i + 1, len(files)), end=' ')

        data = np.genfromtxt(file, skip_header=18)

        if i == 0:
            times = data[:, 0]
            values = data[:, 1:]
        else:
            times = np.hstack((times, data[:, 0]))
            values = np.vstack((values, data[:, 1:]))

        if noisy: print('done.')

    # It may be the case that the files have been supplied in a random order,
    # so sort the values by time here.
    sidx = np.argsort(times)
    times = times[sidx]
    values = values[sidx, ...] # support both 1 and 2D data

    return times, values


def elems2nodes(elems, tri, nvert):
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
        raise Exception('Too many dimensions (maximum of two)')

    # Now calculate the average for each node based on the number of
    # elements of which it is a part.
    nodes = nodes / count

    return nodes


def nodes2elems(nodes, tri):
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

    try:
        import numpy as np
    except ImportError:
        raise ImportError('NumPy not found')

    nvert = np.shape(tri)[0]

    if np.ndim(nodes) == 1:
        elems = nodes[tri].mean(axis=-1)
    elif np.ndim(nodes) == 2:
        elems = nodes[..., tri].mean(axis=-1)
    else:
        raise Exception('Too many dimensions (maximum of two)')

    return elems


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
