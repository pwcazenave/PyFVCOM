from __future__ import print_function

import sys
import inspect

import numpy as np

from warnings import warn
from datetime import datetime
from netCDF4 import Dataset, MFDataset


class ncwrite():
    """
    Save data in a dict to a netCDF file.

    Notes
    -----
    1. Unlimited dimension (None) can only be time and MUST be the 1st
       dimension in the variable dimensions list (or tuple).
    2. Variable dimensions HAVE to BE lists ['time']

    Parameters
    ----------
    data : dict
        Dict of dicts with keys 'dimension', 'variables' and
        'global_attributes'.
    file : str
        Path to output file name.

    Author(s)
    ---------
    Stephane Saux-Picart
    Pierre Cazenave

    Examples
    --------
    >>> lon = np.arange(-10, 10)
    >>> lat = np.arange(50, 60)
    >>> Times = ['2010-02-11 00:10:00.000000', '2010-02-21 00:10:00.000000']
    >>> p90 = np.sin(400).reshape(20, 10, 2)
    >>> data = {}
    >>> data['dimensions'] = {
    ...     'lat': np.size(lat),
    ...     'lon':np.size(lon),
    ...     'time':np.shape(timeStr)[1],
    ...     'DateStrLen':26
    ... }
    >>> data['variables'] = {
    ... 'latitude':{'data':lat,
    ...     'dimensions':['lat'],
    ...     'attributes':{'units':'degrees north'}
    ... },
    ... 'longitude':{
    ...     'data':lon,
    ...     'dimensions':['lon'],
    ...     'attributes':{'units':'degrees east'}
    ... },
    ... 'Times':{
    ...     'data':timeStr,
    ...     'dimensions':['time','DateStrLen'],
    ...     'attributes':{'units':'degrees east'},
    ...     'fill_value':-999.0,
    ...     'data_type':'c'
    ... },
    ... 'p90':{'data':data,
    ...     'dimensions':['lat','lon'],
    ...     'attributes':{'units':'mgC m-3'}}}
    ... data['global attributes'] = {
    ...     'description': 'P90 chlorophyll',
    ...     'source':'netCDF3 python',
    ...     'history':'Created {}'.format(time.ctime(time.time()))
    ... }
    >>> ncwrite(data, 'test.nc')

    """

    def __init__(self, input_dict, filename_out, Quiet=False):
        self.filename_out = filename_out
        self.input_dict = input_dict
        self.Quiet = Quiet
        self.createNCDF()

    def createNCDF(self):
        """
        Function to create and write the data to the specified netCDF file.

        """

        rootgrp = Dataset(self.filename_out, 'w', format='NETCDF3_CLASSIC', clobber=True)

        # Create dimensions.
        if 'dimensions' in self.input_dict:
            for k, v in self.input_dict['dimensions'].iteritems():
                rootgrp.createDimension(k, v)
        else:
            if not self.Quiet:
                print('No netCDF created:')
                print('  No dimension key found (!! has to be \"dimensions\"!!!)')
            return()

        # Create global attributes.
        if 'global attributes' in self.input_dict:
            for k, v in self.input_dict['global attributes'].iteritems():
                rootgrp.setncattr(k, v)
        else:
            if not self.Quiet:
                print('  No global attribute key found (!! has to be \"global attributes\"!!!)')

        # Create variables.
        for k, v in self.input_dict['variables'].iteritems():
            dims = self.input_dict['variables'][k]['dimensions']
            data = v['data']
            # Create correct data type if provided
            if 'data_type' in self.input_dict['variables'][k]:
                data_type = self.input_dict['variables'][k]['data_type']
            else:
                data_type = 'f4'
            # Check whether we've been given a fill value.
            if 'fill_value' in self.input_dict['variables'][k]:
                fill_value = self.input_dict['variables'][k]['fill_value']
            else:
                fill_value = None
            # Create ncdf variable
            if not self.Quiet:
                print('  Creating variable: {} {} {}'.format(k, data_type, dims))
            var = rootgrp.createVariable(k, data_type, dims, fill_value=fill_value)
            if len(dims) > np.ndim(data):
                # If number of dimensions given to netCDF is greater than the
                # number of dimension of the data, then  fill the netCDF
                # variable accordingly.
                if 'time' in dims:
                    # Check for presence of time dimension (which can be
                    # unlimited variable: defined by None).
                    try:
                        var[:] = data
                    except IndexError:
                        raise(IndexError(('Supplied data shape {} does not match the specified'
                        ' dimensions {}, for variable \'{}\'.'.format(data.shape, var.shape, k))))
                else:
                    if not self.Quiet:
                        print('Problem in the number of dimensions')
            else:
                try:
                    var[:] = data
                except IndexError:
                    raise(IndexError(('Supplied data shape {} does not match the specified'
                    ' dimensions {}, for variable \'{}\'.'.format(data.shape, var.shape, k))))

            # Create attributes for variables
            if 'attributes' in self.input_dict['variables'][k]:
                for ka, va in self.input_dict['variables'][k]['attributes'].iteritems():
                    var.setncattr(ka, va)

        rootgrp.close()


def ncread(file, vars=None, dims=False, noisy=False, atts=False):
    """
    Read in the FVCOM results file and spit out numpy arrays for each of the
    variables specified in the vars list.

    Optionally specify a dict with keys whose names match the dimension names
    in the NetCDF file and whose values are strings specifying alternative
    ranges or lists of indices. For example, to extract the first hundred time
    steps, supply dims as:

        dims = {'time':'0:100'}

    To extract the first, 400th and 10,000th values of any array with nodes:

        dims = {'node':'[0, 3999, 9999]'}

    Any dimension not given in dims will be extracted in full.

    Specify atts=True to extract the variable attributes.

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
    FVCOM : dict
        Dict of data extracted from the NetCDF file. Keys are those given in
        vars and the data are stored as ndarrays.
    attributes : dict, optional
        If atts=True, returns the attributes as a dict for each
        variable in vars. The key 'dims' contains the array dimensions (each
        variable contains the names of its dimensions) as well as the shape of
        the dimensions defined in the NetCDF file. The key 'global' contains
        the global attributes.

    See Also
    --------
    read_probes : read in FVCOM ASCII probes output files.

    """

    # If we have a list, assume it's lots of files and load them all.
    if isinstance(file, list):
        try:
            try:
                rootgrp = MFDataset(file, 'r')
            except IOError as msg:
                raise IOError('Unable to open file {} ({}). Aborting.'.format(file, msg))
        except:
            # Try aggregating along a 'time' dimension (for POLCOMS, for example)
            try:
                rootgrp = MFDataset(file, 'r', aggdim='time')
            except IOError as msg:
                raise IOError('Unable to open file {} ({}). Aborting.'.format(file, msg))

    else:
        rootgrp = Dataset(file, 'r')

    # Create a dict of the dimension names and their current sizes
    read_dims = {}
    for key, var in list(rootgrp.dimensions.items()):
        # Make the dimensions ranges so we can use them to extract all the
        # values.
        read_dims[key] = '0:' + str(len(var))

    # Compare the dimensions in the NetCDF file with those provided. If we've
    # been given a dict of dimensions which differs from those in the NetCDF
    # file, then use those.
    if dims:
        commonKeys = set(read_dims).intersection(list(dims.keys()))
        for k in commonKeys:
            read_dims[k] = dims[k]

    if noisy:
        print("File format: {}".format(rootgrp.file_format))

    if not vars:
        vars = iter(list(rootgrp.variables.keys()))

    FVCOM = {}

    # Save the dimensions in the attributes dict.
    if atts:
        attributes = {}
        attributes['dims'] = read_dims
        attributes['global'] = {}
        for g in rootgrp.ncattrs():
            attributes['global'][g] = getattr(rootgrp, g)

    for key, var in list(rootgrp.variables.items()):
        if noisy:
            print('Found ' + key, end=' ')
            sys.stdout.flush()

        if key in vars:
            vDims = rootgrp.variables[key].dimensions

            toExtract = [read_dims[d] for d in vDims]

            # If we have no dimensions, we must have only a single value, in
            # which case set the dimensions to empty and append the function to
            # extract the value.
            if not toExtract:
                toExtract = '.getValue()'

            # Thought I'd finally figured out how to replace the eval approach,
            # but I still can't get past the indexing needed to be able to
            # subset the data.
            # FVCOM[key] = rootgrp.variables.get(key)[0:-1]
            # I know, I know, eval() is evil.
            getData = 'rootgrp.variables[\'{}\']{}'.format(key, str(toExtract).replace('\'', ''))
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


def read_probes(files, noisy=False, locations=False):
    """
    Read in FVCOM probes output files. Reads both 1 and 2D outputs. Currently
    only sensible to import a single station with this function since all data
    is output in a single array.

    Parameters
    ----------
    files : list, tuple
        List of file paths to load.
    noisy : bool, optional
        Set to True to enable verbose output.
    locations : bool, optional
        Set to True to export position and depth data for the sites.

    Returns
    -------
    times : ndarray
        Modified Julian Day times for the extracted time series.
    values : ndarray
        Array of the extracted time series values.
    positions : ndarray, optional
        If locations has been set to True, return an array of the positions
        (lon, lat, depth) for each site.

    See Also
    --------
    ncread : read in FVCOM netCDF output.

    TODO
    ----

    Add support to multiple sites with a single call. Perhaps returning a dict
    with the keys based on the file name is most sensible here?

    """

    if len(files) == 0:
        raise Exception('No files provided.')

    if not (isinstance(files, list) or isinstance(files, tuple)):
        files = [files]

    for i, file in enumerate(files):
        if noisy:
            print('Loading file {} of {}...'.format(i + 1, len(files)), end=' ')

        # Get the header so we can extract the position data.
        with open(file, 'r') as f:
            # Latitude and longitude is stored at line 15 (14 in sPpython
            # counting). Eastings and northings are at 13 (12 in Python
            # indexing).
            lonlatz = [float(pos.strip()) for pos in filter(None, f.readlines()[14].split(' '))]

        data = np.genfromtxt(file, skip_header=18)

        if i == 0:
            times = data[:, 0]
            values = data[:, 1:]
            positions = lonlatz
        else:
            times = np.hstack((times, data[:, 0]))
            values = np.vstack((values, data[:, 1:]))
            positions = np.vstack((positions, lonlatz))

        if noisy:
            print('done.')

    # It may be the case that the files have been supplied in a random order,
    # so sort the values by time here.
    sidx = np.argsort(times)
    times = times[sidx]
    values = values[sidx, ...]  # support both 1 and 2D data

    if locations:
        return times, values, positions
    else:
        return times, values


def write_probes(file, mjd, timeseries, datatype, site, depth, sigma=(-1, -1), lonlat=(0, 0), xy=(0, 0), datestr=None):
    """
    Writes out an FVCOM-formatted time series at a specific location.

    Parameters
    ----------
    mjd : ndarray, list, tuple
        Date/time in Modified Julian Day
    timeseries : ndarray
        Data to write out (vector/array for 1D/2D). Shape should be
        [time, values], where values can be 1D or 2D.
    datatype : tuple, list, tuple
        List with the metadata. Give the long name (e.g. `Temperature') and the
        units (e.g. `Celsius').
    site : str
        Name of the output location.
    depth : float
        Depth at the time series location.
    sigma : ndarray, list, tupel, optional
        Start and end indices of the sigma layer of time series (if
        depth-resolved, -1 otherwise).
    lonlat : ndarray, list, optional
        Coordinates (spherical)
    xy : ndarray, list, optional
        Coordinates (cartesian)
    datestr : str, optional
        Date at which the model was run (contained in the main FVCOM netCDF
        output in the history global variable). If omitted, uses the current
        local date and time. Format is ISO 8601 (YYYY-MM-DDThh:mm:ss.mmmmmm)
        (e.g. 2005-05-25T12:09:56.553467).

    See Also
    --------
    read_probes : read in FVCOM probes output.
    ncread : read in FVCOM netCDF output.

    """

    if not datestr:
        datestr = datetime.now().isoformat()

    day = np.floor(mjd[0])
    usec = (mjd[0] - day) * 24.0 * 3600.0 * 1000.0 * 1000.0

    with open(file, 'w') as f:
        # Write the header.
        f.write('{} at {}\n'.format(datatype[0], site))
        f.write('{} ({})\n'.format(datatype[0], datatype[1]))
        f.write('\n')
        f.write(' !========MODEL START DATE==========\n')
        f.write(' !    Day #    :                 57419\n'.format(day))
        f.write(' ! MicroSecond #:           {}\n'.format(usec))
        f.write(' ! (Date Time={}Z)\n'.format(datestr))
        f.write(' !==========================\n')
        f.write(' \n')
        f.write('          K1            K2\n'.format())
        f.write('           {}             {}\n'.format(*sigma))
        f.write('      X(M)          Y(M)            DEPTH(M)\n')
        f.write('  {:.3f}    {:.3f}         {z:.3f}\n'.format(*xy, z=depth))
        f.write('      LON           LAT               DEPTH(M)\n')
        f.write('      {:.3f}         {:.3f}         {z:.3f}\n'.format(*lonlat, z=depth))
        f.write('\n')
        f.write(' DATA FOLLOWS:\n')
        f.write(' Time(days)    Data...\n')

        # Generate the line format based on the data we've got.
        if np.max(sigma) < 0 or np.min(sigma) - np.max(sigma) == 0:
            # 1D data, so simple time, value format.
            fmt = '{:.5f} {:.3f}\n'
        else:
            # 2D data, so build the string to match the vertical layers.
            fmt = '{:.5f} '
            for sig in range(np.shape(timeseries)[-1]):
                fmt += '{:.3f} '
            fmt = fmt.strip() + '\n'

        # Dump the data (this may be slow).
        for line in np.column_stack((mjd, timeseries)):
            f.write(fmt.format(*line))


def elems2nodes(elems, tri, nvert=None):
    """
    Calculate a nodal value based on the average value for the elements
    of which it a part. This necessarily involves an average, so the
    conversion from nodes2elems and elems2nodes is not reversible.

    Parameters
    ----------
    elems : ndarray
        Array of unstructured grid element values to move to the element
        nodes.
    tri : ndarray
        Array of shape (nelem, 3) comprising the list of connectivity
        for each element.
    nvert : int, optional
        Number of nodes (vertices) in the unstructured grid.

    Returns
    -------
    nodes : ndarray
        Array of values at the grid nodes.

    """

    if not nvert:
        nvert = np.max(tri) + 1
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

    elif np.ndim(elems) > 1:
        # Horrible hack alert to get the output array shape for multiple
        # dimensions.
        nodes = np.zeros((list(np.shape(elems)[:-1]) + [nvert]))
        for i, indices in enumerate(tri):
            n0, n1, n2 = indices
            nodes[..., n0] = nodes[..., n0] + elems[..., i]
            nodes[..., n1] = nodes[..., n1] + elems[..., i]
            nodes[..., n2] = nodes[..., n2] + elems[..., i]
            count[n0] = count[n0] + 1
            count[n1] = count[n1] + 1
            count[n2] = count[n2] + 1

    # Now calculate the average for each node based on the number of
    # elements of which it is a part.
    nodes /= count

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

    if np.ndim(nodes) == 1:
        elems = nodes[tri].mean(axis=-1)
    elif np.ndim(nodes) == 2:
        elems = nodes[..., tri].mean(axis=-1)
    else:
        raise Exception('Too many dimensions (maximum of two)')

    return elems


# For backwards compatibility.
def readFVCOM(file, varList=None, clipDims=False, noisy=False, atts=False):
    warn('{} is deprecated. Use ncread instead.'.format(inspect.stack()[0][3]))

    F = ncread(file, vars=varList, dims=clipDims, noisy=noisy, atts=atts)

    return F


def readProbes(*args, **kwargs):
    warn('{} is deprecated. Use read_probes instead.'.format(inspect.stack()[0][3]))
    return read_probes(*args, **kwargs)


def writeProbes(*args, **kwargs):
    warn('{} is deprecated. Use write_probes instead.'.format(inspect.stack()[0][3]))
    return write_probes(*args, **kwargs)
