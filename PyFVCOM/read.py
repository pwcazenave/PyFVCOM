# -*- coding: utf-8 -*-

from __future__ import print_function, division

import copy
import sys
from datetime import datetime, timedelta
from warnings import warn

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from netCDF4 import Dataset, MFDataset, num2date, date2num
from shapely.geometry import Polygon

from PyFVCOM.grid import Domain, control_volumes, get_area_heron
from PyFVCOM.grid import unstructured_grid_volume, elems2nodes, GridReaderNetCDF
from PyFVCOM.utilities.general import _passive_data_store

class _TimeReader(object):

    def __init__(self, dataset, dims=None, verbose=False):
        """
        Parse the time data from an FVCOM netCDF file Missing standard FVCOM time variables are automatically created.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            The netCDF file Dataset object.
        dims : dict, optional
            Dictionary of dimension names along which to subsample e.g. dims={'time': [0, 100]}. Dimensions are
            specified as list of indices. Time can also be specified as either a time string (e.g. '2000-01-25
            23:00:00.00000') or given as a datetime object. If omitted, all times are loaded. Negative indices are
            supported. To load from the 10th and last time can be written as 'time': [9, -1]).
        verbose : bool, optional
            Set to True to enable verbose output. Defaults to False.

        """

        _noisy = verbose
        self._dims = copy.deepcopy(dims)

        time_variables = ('time', 'Itime', 'Itime2', 'Times')
        got_time, missing_time = [], []
        for time in time_variables:
            # Since not all of the time_variables specified above are required, only try to load the data if they
            # exist. We'll raise an error if we don't find any of them though.
            if time in dataset.variables:
                setattr(self, time, dataset.variables[time][:])
                got_time.append(time)
                attributes = _passive_data_store()
                for attribute in dataset.variables[time].ncattrs():
                    setattr(attributes, attribute, getattr(dataset.variables[time], attribute))
                # setattr(self.atts, time, attributes)
            else:
                missing_time.append(time)

        if len(missing_time) == len(time_variables):
            warn('No time variables found in the netCDF.')
        else:
            # If our file has incomplete dimensions (i.e. no time), add that here.
            if not hasattr(dims, 'time'):

                _Times_shape = None
                _other_time_shape = None
                if 'Times' in got_time:
                    _Times_shape = np.shape(self.Times)
                _other_times = [i for i in got_time if i != 'Times']
                if _other_times:
                    if getattr(self, _other_times[0]).shape:
                        _other_time_shape = len(getattr(self, _other_times[0]))
                    else:
                        # We only have a single value, so len doesn't work.
                        _other_time_shape = 1

                if _noisy:
                    print('Added time dimension size since it is missing from the input netCDF file.')

            if 'Times' in got_time:
                # Check whether we've got missing values and try and make them from one of the others. This sometimes
                # happens if you stop a model part way through a run. We check for masked arrays at this point
                # because the netCDF library only returns masked arrays when we have NaNs in the results.
                if isinstance(dataset.variables['Times'][:], np.ma.core.MaskedArray):
                    bad_times = np.argwhere(np.any(dataset.variables['Times'][:].data == ([b''] * dataset.dimensions['DateStrLen'].size), axis=1)).ravel()
                    if np.any(bad_times):
                        if 'time' in got_time:
                            for bad_time in bad_times:
                                if self.time[bad_time]:
                                    self.Times[bad_time] = list(datetime.strftime(num2date(self.time[bad_time], units='days since 1858-11-17 00:00:00'), '%Y-%m-%dT%H:%M:%S.%f'))
                        elif 'Itime' in got_time and 'Itime2' in got_time:
                            for bad_time in bad_times:
                                if self.Itime[bad_time] and self.Itime2[bad_time]:
                                    self.Times[bad_time] = list(datetime.strftime(num2date(self.Itime[bad_time] + self.Itime2[bad_time] / 1000.0 / 60 / 60, units=getattr(dataset.variables['Itime'], 'units')), '%Y-%m-%dT%H:%M:%S.%f'))

                # Overwrite the existing Times array with a more sensibly shaped one.
                try:
                    self.Times = np.asarray([''.join(t.astype(str)).strip() for t in self.Times])
                except TypeError:
                    # We might have a masked array, so just use the raw data.
                    self.Times = np.asarray([''.join(t.astype(str)).strip() for t in self.Times.data])

            # Make whatever we got into datetime objects and use those to make everything else. Note: the `time'
            # variable is often the one with the lowest precision, so use the others preferentially over that.
            if 'Times' not in got_time:
                if 'time' in got_time:
                    _dates = num2date(self.time, units=getattr(dataset.variables['time'], 'units'))
                elif 'Itime' in got_time and 'Itime2' in got_time:
                    _dates = num2date(self.Itime + self.Itime2 / 1000.0 / 60 / 60 / 24, units=getattr(dataset.variables['Itime'], 'units'))
                else:
                    raise ValueError('Missing sufficient time information to make the relevant time data.')

                try:
                    self.Times = np.array([datetime.strftime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in _dates])
                except ValueError:
                    self.Times = np.array([datetime.strftime(d, '%Y/%m/%d %H:%M:%S.%f') for d in _dates])
                # Add the relevant attribute for the Times variable.
                attributes = _passive_data_store()
                setattr(attributes, 'time_zone', 'UTC')
                # setattr(self.atts, 'Times', attributes)

            if 'time' not in got_time:
                if 'Times' in got_time:
                    try:
                        _dates = np.array([datetime.strptime(''.join(t.astype(str)).strip(), '%Y-%m-%dT%H:%M:%S.%f') for t in self.Times])
                    except ValueError:
                        _dates = np.array([datetime.strptime(''.join(t.astype(str)).strip(), '%Y/%m/%d %H:%M:%S.%f') for t in self.Times])
                elif 'Itime' in got_time and 'Itime2' in got_time:
                    _dates = num2date(self.Itime + self.Itime2 / 1000.0 / 60 / 60 / 24, units=getattr(dataset.variables['Itime'], 'units'))
                else:
                    raise ValueError('Missing sufficient time information to make the relevant time data.')

                # We're making Modified Julian Days here to replicate FVCOM's 'time' variable.
                self.time = date2num(_dates, units='days since 1858-11-17 00:00:00')
                # Add the relevant attributes for the time variable.
                attributes = _passive_data_store()
                setattr(attributes, 'units', 'days since 1858-11-17 00:00:00')
                setattr(attributes, 'long_name', 'time')
                setattr(attributes, 'format', 'modified julian day (MJD)')
                setattr(attributes, 'time_zone', 'UTC')
                # setattr(self.atts, 'time', attributes)

            if 'Itime' not in got_time and 'Itime2' not in got_time:
                if 'Times' in got_time:
                    try:
                        _dates = np.array([datetime.strptime(''.join(t.astype(str)).strip(), '%Y-%m-%dT%H:%M:%S.%f') for t in self.Times])
                    except ValueError:
                        _dates = np.array([datetime.strptime(''.join(t.astype(str)).strip(), '%Y/%m/%d %H:%M:%S.%f') for t in self.Times])
                elif 'time' in got_time:
                    _dates = num2date(self.time, units=getattr(dataset.variables['time'], 'units'))
                else:
                    raise ValueError('Missing sufficient time information to make the relevant time data.')

                # We're making Modified Julian Days here to replicate FVCOM's 'time' variable.
                _datenum = date2num(_dates, units='days since 1858-11-17 00:00:00')
                self.Itime = np.floor(_datenum)
                self.Itime2 = (_datenum - np.floor(_datenum)) * 1000 * 60 * 60 * 24  # microseconds since midnight
                attributes = _passive_data_store()
                setattr(attributes, 'units', 'days since 1858-11-17 00:00:00')
                setattr(attributes, 'format', 'modified julian day (MJD)')
                setattr(attributes, 'time_zone', 'UTC')
                # setattr(self.atts, 'Itime', attributes)
                attributes = _passive_data_store()
                setattr(attributes, 'units', 'msec since 00:00:00')
                setattr(attributes, 'time_zone', 'UTC')
                # setattr(self.atts, 'Itime2', attributes)

            # Additional nice-to-have time representations.
            if 'Times' in got_time:
                try:
                    self.datetime = np.array([datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in self.Times])
                except ValueError:
                    self.datetime = np.array([datetime.strptime(d, '%Y/%m/%d %H:%M:%S.%f') for d in self.Times])
                attributes = _passive_data_store()
                setattr(attributes, 'long_name', 'Python datetime.datetime')
                # setattr(self.atts, 'datetime', attributes)
            else:
                self.datetime = _dates
            self.matlabtime = self.time + 678942.0  # to MATLAB-indexed times from Modified Julian Date.
            attributes = _passive_data_store()
            setattr(attributes, 'long_name', 'MATLAB datenum')
            # setattr(self.atts, 'matlabtime', attributes)

            # Clip everything to the time indices if we've been given them. Update the time dimension too.
            if 'time' in self._dims:
                if not isinstance(self._dims['time'], slice) and all([isinstance(i, (datetime, str)) for i in self._dims['time']]):
                    # Convert datetime dimensions to indices in the currently loaded data. Assume we've got a list
                    # and if that fails, we've probably got a single index, so convert it accordingly.
                    try:
                        self._dims['time'] = np.arange(*[self._time_to_index(i) for i in self._dims['time']])
                    except TypeError:
                        self._dims['time'] = np.arange(*[self._time_to_index(self._dims['time'])])  # make iterable
                for time in self:
                    setattr(self, time, getattr(self, time)[self._dims['time']])

    def __iter__(self):
        return (a for a in dir(self) if not a.startswith('_'))

    def _time_to_index(self, *args, **kwargs):
        """
        Find the time index for the given time string (%Y-%m-%d %H:%M:%S.%f) or datetime object.

        Parameters
        ----------
        target_time : str or datetime.datetime
            Time for which to find the time index. If given as a string, the time format must be "%Y-%m-%d %H:%M:%S.%f".
        tolerance : float, optional
            Seconds of tolerance to allow when finding the appropriate index. Use this flag to only return an index
            to within some tolerance. By default, the closest time is returned irrespective of how far in time it is
            from the data.

        Returns
        -------
        time_idx : int
            Index for the currently loaded data closest to the specified time.

        """

        time_idx = time_to_index(self.datetime, *args, **kwargs)

        return time_idx

class _AttributeReader(object):
    def __init__(self, dataset, variables=None, verbose=False):
        """
        Load the attributes for the variables in the dataset. Optionally limit the attributes to the variables in
        variables.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            The netCDF file dataset object.
        variables : list, optional
            List of variables to extract. If omitted, only the grid, grid metrics and time variables are extracted.
        verbose : bool, optional
            Set to True to enable verbose output. Defaults to False.

        """

        self._ds = dataset
        self._noisy = verbose

        grid = ['a1u', 'a2u', 'art1', 'art2',
                'h', 'h_center',
                'lat', 'latc', 'lon', 'lonc',
                'nbe', 'nbsn', 'nbve', 'ntsn', 'ntve', 'nv',
                'siglay', 'siglay_center', 'siglev', 'siglev_center',
                'x', 'y', 'xc', 'yc']
        time = ['time', 'Times', 'Itime', 'Itime2']

        all_variables = grid + time
        if variables is not None:
            all_variables = all_variables + variables

        for var in all_variables:
            if var in self._ds.variables:
                if self._noisy:
                    print(f'Getting attributes for {var}')
                self.get_attribute(var)

    def get_attribute(self, variable):
        """
        Get the attributes for the given variable and add them to the relevant object.

        Parameters
        ----------
        variable : str
            The variable from which to extract the attributes.

        """

        if not hasattr(self, variable):
            # Hmmm, don't like using _passive_data_store here...
            setattr(self, variable, _passive_data_store())

        for attribute in self._ds.variables[variable].ncattrs():
            setattr(getattr(self, variable), attribute, getattr(self._ds.variables[variable], attribute))

    def __iter__(self):
        return (a for a in dir(self) if not a.startswith('__') and not a.endswith('__'))


class _MakeDimensions(object):
    def __init__(self, dataset):
        """
        Calculate some dimensions from the given _GridReader object.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            The netCDF4 object from which to extract the dimensions we need.

        """

        for dim in dataset.dimensions:
            setattr(self, dim, dataset.dimensions[dim].size)

    def __iter__(self):
        return (a for a in dir(self) if not a.startswith('__') and not a.endswith('__'))


class FileReader(Domain):
    """ Load FVCOM model output.

    Class simplifies the preparation of FVCOM model output for analysis with PyFVCOM.

    Author(s)
    ---------
    Pierre Cazenave (Plymouth Marine Laboratory)

    Credits
    -------
    This code leverages ideas (and in some cases, code) from PySeidon (https://github.com/GrumpyNounours/PySeidon)
    and PyLag-tools (https://gitlab.em.pml.ac.uk/PyLag/PyLag-tools).

    """

    def __init__(self, fvcom, variables=[], dims={}, zone='30N', debug=False, verbose=False, subset_method='slice'):
        """
        Parameters
        ----------
        fvcom : str, pathlib.Path, netCDF4.Dataset
            Path to an FVCOM netCDF.
        variables : list-like, optional
            List of variables to extract. If omitted, no variables are extracted, which means you won't be able to
            add this object to another one which does have variables in it.
        dims : dict, optional
            Dictionary of dimension names along which to subsample e.g. dims={'time': [0, 100], 'nele': [0, 10, 100],
            'node': 100}.
            All netCDF variable dimensions are specified as list of indices. Time can also be specified as either a
            time string (e.g. '2000-01-25 23:00:00.00000') or given as a datetime object.
            Any combination of dimensions is possible; omitted dimensions are loaded in their entirety.
            Negative indices are supported. To load from the 10th and last time can be written as 'time': [9, -1]).
        zone : str, list-like, optional
            UTM zones (defaults to '30N') for conversion of UTM to spherical coordinates.
        verbose : bool, optional
            Set to True to enable verbose output. Defaults to False.
        debug : bool, optional
            Set to True to enable debug output. Defaults to False.
        subset_method : str, optional
            Define the subsetting method to use if `dims' has been given. Choices are 'memory' or 'slice'. With
            'memory', all the data are loaded from netCDF into memory and then sliced in memory; with 'slice',
            the data are sliced directly from the netCDF file. The former uses a lot more memory but may be faster,
            the latter uses much less memory but is more sensitive to the structure of the netCDF.

        Example
        -------

        # Load and plot surface currents as surface and quiver plots.
        >>> from PyFVCOM.read import FileReader
        >>> from PyFVCOM.plot import Plotter
        >>> from PyFVCOM.current import vector2scalar
        >>> fvcom = FileReader('casename_0001.nc', variables=['u', 'v'], dims={'siglay': [0]})
        >>> # Calculate speed and direction from the current vectors
        >>> fvcom.data.direction, fvcom.data.speed = vector2scalar(fvcom.data.u, fvcom.data.v)
        >>> plot = Plotter(fvcom)
        >>> plot.plot_field(fvcom.data.speed)
        >>> plot.plot_quiver(fvcom.data.u, fvcom.data.v, field=fvcom.data.speed, add_key=True, scale=5)

        Author(s)
        ---------
        Pierre Cazenave (Plymouth Marine Laboratory)
        Mike Bedington (Plymouth Marine Laboratory)

        """

        self._debug = debug
        self._noisy = verbose
        self._fvcom = fvcom
        self._zone = zone
        self._get_data_pattern = subset_method

        if not hasattr(self, '_bounding_box'):
            self._bounding_box = False
        # We may modify the dimensions, so make a deepcopy (copy isn't sufficient) so successive calls to FileReader
        # from MFileReader work properly.
        self._dims = copy.deepcopy(dims)

        # Silently convert a string variable input to an iterable list.
        if isinstance(variables, str):
            variables = [variables]
        self._variables = variables

        # Prepare this object with an empty object for the data which we may populate later on. It feels like this
        # should be created by a separate class (in the same style as _MakeDimensions et al.), but I haven't got
        # around to it yet.
        self.data = _passive_data_store()

        if isinstance(self._fvcom, Dataset):
            self.ds = self._fvcom
            # We use this as a string in some output messages, so get the file name from the object.
            self._fvcom = self.ds.filepath()
        else:
            self._fvcom = str(self._fvcom)  # in case it was a pathlib.Path.
            self.ds = Dataset(self._fvcom, 'r')

        self.dims = _MakeDimensions(self.ds)

        for dim in self._dims:
            dim_is_iterable = hasattr(self._dims[dim], '__iter__')
            dim_is_string = isinstance(self._dims[dim], str)  # for date ranges
            dim_is_slice = isinstance(self._dims[dim], slice)
            if not dim_is_iterable and not dim_is_slice and not dim_is_string:
                if self._noisy:
                    print('Making dimension {} iterable'.format(dim))

                self._dims[dim] = [self._dims[dim]]

        self._load_time()
        self._dims = self.time._dims  # grab the updated dimensions from the _TimeReader object.

        # Update the time dimension no we've read in the time data (in case we did so with a specificed dimension
        # range).
        self.dims.time = len(self.time.time)

        self._load_grid(fvcom)

        # Load the attributes of anything we've been asked to load.
        self.atts = _AttributeReader(self.ds, self._variables)

        if self._variables:
            self.load_data(self._variables)

    def __iter__(self):
        return (a for a in dir(self) if not a.startswith('_'))

    def __eq__(self, other):
        # For easy comparison of classes.
        return self.__dict__ == other.__dict__

    def __add__(self, fvcom, debug=False):
        """
        This special method means we can stack two FileReader objects in time through a simple addition (e.g. fvcom1
        += fvcom2)

        Parameters
        ----------
        fvcom : PyFVCOM.FileReader
            Previous time to which to add ourselves.

        Returns
        -------
        idem : PyFVCOM.FileReader
            Concatenated (in time) `PyFVCOM.FileReader' class.

        Notes
        -----
        - fvcom1 and fvcom2 have to cover the exact same spatial domain
        - last time step of fvcom1 must be <= to the first time step of fvcom2
        - if variables have not been loaded, then subsequent loads will load only the data from the first netCDF.
        Make sure you load all your variables before you merge objects.

        History
        -------

        This is a reimplementation of the FvcomClass.__add__ method of `PySeidon'. Modified by Pierre Cazenave for
        use in `PyFVCOM', mainly because PySeidon isn't compatible with python 3 (yet).

        """

        # Compare our current grid and time with the supplied one to make sure we're dealing with the same model
        # configuration. We also need to make sure we've got the same set of data (if any). We'll warn if we've got
        # no data loaded that we can't do subsequent data loads.
        node_compare = self.dims.nele == fvcom.dims.nele
        nele_compare = self.dims.node == fvcom.dims.node
        siglay_compare = self.dims.siglay == fvcom.dims.siglay
        siglev_compare = self.dims.siglev == fvcom.dims.siglev
        time_compare = self.time.datetime[-1] <= fvcom.time.datetime[0]
        old_data = [i for i in self.data]
        new_data = [i for i in fvcom.data]
        data_compare = new_data == old_data
        if not node_compare:
            raise ValueError('Horizontal node data are incompatible.')
        if not nele_compare:
            raise ValueError('Horizontal element data are incompatible.')
        if not siglay_compare:
            raise ValueError('Vertical sigma layers are incompatible.')
        if not siglev_compare:
            raise ValueError('Vertical sigma levels are incompatible.')
        if not time_compare:
            raise ValueError("Time periods are incompatible (`fvcom2' must be greater than or equal to `fvcom1')."
                             "`fvcom1' has end {} and `fvcom2' has start {}".format(self.time.datetime[-1],
                                                                                    fvcom.time.datetime[0]))
        if not data_compare:
            raise ValueError('Loaded data sets for each FileReader class must match.')
        if not (old_data == new_data) and (old_data or new_data):
            warn('Subsequent attempts to load data for this merged object will only load data from the first object. '
                 'Load data into each object before merging them.')

        # Copy ourselves to a new version for concatenation. self is the old so we get appended to by the new.
        idem = copy.copy(self)

        # Go through all the parts of the data with a time dependency and concatenate them. Leave the grid alone.
        for var in idem.data:
            if 'time' in idem.ds.variables[var].dimensions:
                setattr(idem.data, var, np.concatenate((getattr(idem.data, var), getattr(fvcom.data, var))))
        for time in idem.time:
            setattr(idem.time, time, np.concatenate((getattr(idem.time, time), getattr(fvcom.time, time))))

        # Remove duplicate times.
        time_indices = np.arange(len(idem.time.time))
        _, dupes = np.unique(idem.time.time, return_index=True)
        dupe_indices = np.setdiff1d(time_indices, dupes)
        for var in idem.data:
            # Only delete things with a time dimension.
            if 'time' in idem.ds.variables[var].dimensions:
                time_axis = idem.ds.variables[var].dimensions.index('time')
                setattr(idem.data, var, np.delete(getattr(idem.data, var), dupe_indices, axis=time_axis))
        for time in idem.time:
            try:
                time_axis = idem.ds.variables[time].dimensions.index('time')
                setattr(idem.time, time, np.delete(getattr(idem.time, time), dupe_indices, axis=time_axis))
            except KeyError:
                # This is hopefully one of the additional time variables which doesn't exist in the netCDF dataset.
                # Just delete the relevant indices by assuming that time is the first axis.
                setattr(idem.time, time, np.delete(getattr(idem.time, time), dupe_indices, axis=0))

        # Update dimensions accordingly.
        idem.dims.time = len(idem.time.time)

        return idem

    def _load_grid(self, fvcom):
        self.grid = GridReaderNetCDF(fvcom, dims=self._dims, zone=self._zone, debug=self._debug, verbose=self._noisy)
    
    def _load_time(self):
        self.time = _TimeReader(self.ds, dims=self._dims)

    def _update_dimensions(self, variables):
        # Update the dimensions based on variables we've been given. Construct a list of the unique dimensions in all
        # the given variables and use that to update self.dims.
        unique_dims = {}  # {dim_name: size}
        for var in variables:
            for dim in self.ds.variables[var].dimensions:
                if dim not in unique_dims:
                    dim_index = self.ds.variables[var].dimensions.index(dim)
                    unique_dims[dim] = getattr(self.data, var).shape[dim_index]
        for dim in unique_dims:
            if self._debug:
                if getattr(self.dims, dim) != unique_dims[dim]:
                    print('{} dimension, old/new: {}/{}.'.format(dim, getattr(self.dims, dim), unique_dims[dim]))
                else:
                    print('{} dimension size unchanged ({}).'.format(dim, getattr(self.dims, dim)))
            setattr(self.dims, dim, unique_dims[dim])

    def load_data(self, var, dims=None):
        """
        Load a given variable(s).

        Respect dimensions if supplied, otherwise falls back to those in self.FileReader (if any).

        Parameters
        ----------
        var : list-like, str
            Variable(s) to load.
        dims : dictionary, optional
            Supply specific dimensions to load. If omitted, uses the global dimensions supplied to FileReader (if any).

        """

        if dims is None:
            dims = copy.copy(self._dims)
        else:
            # Reload the grid and time data with the new dimensions, so everything matches.
            if self._debug:
                print('Updating existing data to match supplied dimensions when loading data')
            # Use the supplied dimensions as the new dimensions array.
            self._dims = dims
            self.time = _TimeReader(self.ds, dims=self._dims)
            self.dims.time = len(self.time)
            self.grid = _GridReaderNetCDF(self._fvcom, dims=self._dims, zone=self._zone, debug=self._debug,
                                          verbose=self._noisy)

        # Check if we've got iterable variables and make one if not.
        if not hasattr(var, '__iter__') or isinstance(var, str):
            var = [var]

        for v in var:
            if self._debug or self._noisy:
                print(f'Loading: {v}')

            if v not in self.ds.variables:
                raise NameError(f"Variable '{v}' not present in {self._fvcom}")

            var_dim = self.ds.variables[v].dimensions
            variable_shape = self.ds.variables[v].shape
            variable_indices = [slice(None) for _ in variable_shape]
            # Update indices for dimensions of the variable we've been asked to subset.
            for dimension in var_dim:
                if dimension in dims:
                    variable_index = var_dim.index(dimension)
                    if self._debug:
                        print(f'Extracting specific indices for {dimension}')
                    variable_indices[variable_index] = dims[dimension]

            # Add attributes for the variable we're loading.
            self.atts.get_attribute(v)

            if 'time' not in var_dim:
                # Should we error here or carry on having warned?
                warn(f'{v} does not contain a time dimension.')

            try:
                if self._get_data_pattern == 'slice':
                    if self._debug:
                        print('Slicing the data directly from netCDF')
                    setattr(self.data, v, self.ds.variables[v][variable_indices])
                elif self._get_data_pattern == 'memory':
                    if self._debug:
                        print('Loading all data in memory and then subsetting')
                    data_raw = self.ds.variables[v][:]

                    for i in range(data_raw.ndim):
                        if not isinstance(variable_indices[i], slice):
                            if self._debug:
                                print(f'Extracting indices {variable_indices[i]} for variable {v}')
                            data_raw = data_raw.take(variable_indices[i], axis=i)

                    setattr(self.data, v, data_raw)
                    del data_raw

            except MemoryError:
                raise MemoryError("Variable {} too large for RAM. Use `dims' to load subsets in space or time or "
                                  "`variables' to request only certain variables.".format(v))

        # Update the dimensions to match the data.
        self._update_dimensions(var)

    def closest_time(self, when):
        """ Find the index of the closest time to the supplied time (datetime object). """
        try:
            return np.argmin(np.abs(self.time.datetime - when))
        except AttributeError:
            self.load_time()
            return np.argmin(np.abs(self.time.datetime - when))

    def grid_volume(self, load_zeta=False):
        """
        Calculate the grid volume (optionally time varying) for the loaded grid.

        If the surface elevation data have been loaded (`zeta'), the volume varies with time, otherwise, the volume
        is for the mean water depth (`h'). To load the surface elevation with this call, set `load_zeta' to True.

        Parameters
        ----------
        load_zeta : bool, optional
            Set to True to load the surface elevation data. If omitted, any existing surface elevation data are used,
            otherwise it is ignored.

        Provides
        --------
        self.grid.depth_volume : np.ndarray
            Depth-resolved volume.
        self.grid.depth_integrated_volume : np.ndarray
            Depth-integrated volume.

        """

        if hasattr(self.data, 'zeta'):
            surface_elevation = self.data.zeta
        elif load_zeta:
            self.load_data(['zeta'])
        else:
            surface_elevation = np.zeros((self.dims.time, self.dims.node))

        self.grid.depth_volume, self.grid.depth_integrated_volume = unstructured_grid_volume(self.grid.art1, self.grid.h, surface_elevation, self.grid.siglev, depth_integrated=True)

    def _get_cv_volumes(self, poolsize=None):
        """
        Calculate the control area volumes in the model domain.

        Parameters
        ----------
        poolsize : int, optional
            Specify a number of processes to use when calculating the grid control volumes. Defaults to no parallelism.

        Provides
        --------
        self.grid.depth : np.ndarray
            Time varying water depth.
        self.grid.depth_volume : np.ndarray
            Depth-resolved volume.
        self.grid.depth_integrated_volume : np.ndarray
            The volume of the model domain over time.

        Todo
        ----
        This function duplicates some of self.grid_volume, so we should rationalise these two to avoid adding new
        variables if we can use existing ones instead.

        """

        if not hasattr(self.grid, 'art1'):
            self.grid.art1 = np.asarray(control_volumes(self.grid.x, self.grid.y, self.grid.triangles,
                                                        element_control=False, poolsize=poolsize))

        if not hasattr(self.data, 'zeta'):
            self.load_data(['zeta'])
        # Calculate depth-resolved and depth-integrated volume.
        self.grid_volume()
        self.grid.depth = self.data.zeta + self.grid.h

    def total_volume_var(self, var, poolsize=None):
        """
        Integrate a given variable in space returning a time series of the integrated values.

        Parameters
        ----------
        var : str
            The name of the variable to load. Must be a depth-resolved array.
        poolsize : int, optional
            Specify a number of processes to use when calculating the grid control volumes. Defaults to no parallelism.

        Provides
        --------
        {var}_total : np.ndarray
            Adds a new array which is a time series of the integrated value of the variable at each model time.

        """

        self._get_cv_volumes(poolsize=poolsize)

        if not hasattr(self.data, var):
            self.load_data([var])

        if len(getattr(self.data, var).shape) != 3:
            raise ValueError('The requested variable ({}) is not depth-resolved.'.format(var))

        # Do as a loop because it nukes the memory otherwise
        int_vol = np.zeros(self.dims.time)

        for i in range(len(self.grid.x)):
            int_vol += np.sum(getattr(self.data, var)[..., i] *
                              self.grid.depth_volume[..., i], axis=1)

        setattr(self.data, '{}_total'.format(var), int_vol)

    def avg_volume_var(self, var):
        """
        TODO: Add docstring.

        :param var:
        :return:

        """
        if not hasattr(self, 'volume'):
            self._get_cv_volumes()

        if not hasattr(self.data, var):
            self.load_data([var])

        int_vol = 0
        for i in range(len(self.grid.x)):
            int_vol += np.average(getattr(self.data, var)[:, :, i],
                                  weights=self.grid.depth_volume[..., i], axis=1)
        setattr(self.data, '{}_average'.format(var), int_vol)

    def time_to_index(self, *args, **kwargs):
        """
        Find the time index for the given time string (%Y-%m-%d %H:%M:%S.%f) or datetime object.

        Parameters
        ----------
        target_time : str or datetime.datetime
            Time for which to find the time index. If given as a string, the time format must be "%Y-%m-%d %H:%M:%S.%f".
        tolerance : float, optional
            Seconds of tolerance to allow when finding the appropriate index. Use this flag to only return an index
            to within some tolerance. By default, the closest time is returned irrespective of how far in time it is
            from the data.

        Returns
        -------
        time_idx : int
            Index for the currently loaded data closest to the specified time.

        """

        time_idx = time_to_index(self.time.datetime, *args, **kwargs)

        return time_idx

    def time_average(self, variable, period, return_times=False):
        """
        Average the requested variable in time at the specified frequency. If the data for variable are not loaded,
        they will be before averaging. Averaging starts at the first midnight in the time series and ends at the last
        midnight in the time series (values outside those times are ignored).

        The result is added to self.data as an attribute named '{}_{}'.format(variable, period).

        Parameters
        ----------
        variable : str
            A variable to average in time.
        period : str
            The period over which to average. Select from `daily', `weekly', `monthly' or `yearly' (`annual' is a
            synonym). `Monthly' is actually 4-weekly rather than per calendar month.
        return_times : bool, optional
            Set to True to return the times of the averages as datetimes. Defaults to False.

        Returns
        -------
        times : np.ndarray
            If return_times is set to True, return an array of the datetimes which correspond to each average.

        """

        # TODO: the monthly and yearly averaging could be more robust by finding the unique complete months/years and
        # TODO: averaging those rather than assuming 4 weekly months and 365 day years.
        interval_seconds = np.unique([i.total_seconds() for i in np.diff(self.time.datetime)])
        if len(interval_seconds) > 1:
            raise ValueError('Cannot compute time average on irregularly sampled data.')
        seconds_per_day = 60 * 60 * 24
        if period == 'daily':
            step = int(seconds_per_day / interval_seconds)
        elif period == 'weekly':
            step = int(seconds_per_day * 7 / interval_seconds)
        elif period == 'monthly':
            step = int(seconds_per_day * 7 * 4 / interval_seconds)
        elif period == 'yearly' or period == 'annual':
            step = int(seconds_per_day * 365 / interval_seconds)
        else:
            raise ValueError('Unsupported period {}'.format(period))

        if not hasattr(self.data, variable):
            if self._noisy:
                print('Loading {} for time-averaging.'.format(variable))
            self.load_data(variable)

        # We're assuming time is the first dimension here since we're working with FVCOM data by and large. If that
        # changes, we'll have to find the index of the time dimension and then do the reshape with that dimension
        # first before reshaping back to the original order. You might be able to understand why I'm not bothering
        # with all that right now.

        # First, find the first midnight values as all periods average from midnight on the first day.
        midnights = [_.hour == 0 for _ in self.time.datetime]
        first_midnight = np.argwhere(midnights)[0][0]
        last_midnight = np.argwhere(midnights)[-1][0]

        if first_midnight == last_midnight:
            raise IndexError('Too few data to average at {} frequency.'.format(period))

        # For the averaging, reshape the time dimension into chunks which match the periods and then average along
        # that reshaped dimension. Getting the new shape is a bit fiddly. We should always have at least two
        # dimensions here (time, space) so this should always work.
        other_dimensions = [_ for _ in getattr(self.data, variable).shape[1:]]

        # Check that the maximum difference of the first day's data and the first averaged data is zero:
        # (averaged[0] - getattr(self.data, variable)[first_midnight:first_midnight + step].mean(axis=0)).max() == 0
        averaged = getattr(self.data, variable)[first_midnight:last_midnight, ...]
        averaged = np.mean(averaged.reshape([-1, step] + other_dimensions), axis=1)

        setattr(self.data, '{}_{}'.format(variable, period), averaged)

        if return_times:
            # For the arithmetic to be simple, we'll do this on `time'. This is possibly an issue as `time' is
            # sometimes not sufficiently precise to resolve the actual times accurately. It would be better to do
            # this on `datetime' instead, but then we have to fiddle around making things relative and it all gets a
            # bit tiresome.
            new_times = num2date(self.time.time[first_midnight:last_midnight].reshape(-1, step).mean(axis=1),
                                 units=self.atts.time.units)
            return new_times

    def add_river_flow(self, river_nc_file, river_nml_file):
        """
        TODO: docstring.

        """

        nml_dict = get_river_config(river_nml_file)
        river_node_raw = np.asarray(nml_dict['RIVER_GRID_LOCATION'], dtype=int) - 1
        self.river = _passive_data_store()

        river_nc = nc.Dataset(river_nc_file, 'r')
        time_raw = river_nc.variables['Times'][:]
        self.river.time_dt = [datetime.strptime(b''.join(this_time).decode('utf-8').rstrip(), '%Y/%m/%d %H:%M:%S') for this_time in time_raw]

        ref_date = datetime(1900,1,1)
        mod_time_sec = [(this_dt - ref_date).total_seconds() for this_dt in self.time.datetime]
        self.river.river_time_sec = [(this_dt - ref_date).total_seconds() for this_dt in self.river.time_dt]

        if 'node' in self._dims:
            self.river.river_nodes = np.argwhere(np.isin(self._dims['node'], river_node_raw))
            rivers_in_grid = np.isin(river_node_raw, self._dims['node'])
        else:
            self.river.river_nodes = river_node_raw    
            rivers_in_grid = np.ones(river_node_raw.shape, dtype=bool)

        river_flux_raw = river_nc.variables['river_flux'][:,rivers_in_grid]
        self.river.river_fluxes = np.asarray([np.interp(mod_time_sec, self.river.river_time_sec, this_col) for this_col in river_flux_raw.T]).T
        self.river.total_flux = np.sum(self.river.river_fluxes, axis=1)

        river_temp_raw = river_nc.variables['river_temp'][:,rivers_in_grid]
        self.river.river_temp = np.asarray([np.interp(mod_time_sec, self.river.river_time_sec, this_col) for this_col in river_temp_raw.T]).T

        river_salt_raw = river_nc.variables['river_salt'][:,rivers_in_grid]
        self.river.river_salt = np.asarray([np.interp(mod_time_sec, self.river.river_time_sec, this_col) for this_col in river_salt_raw.T]).T


def MFileReader(fvcom, noisy=False, *args, **kwargs):
    """
    Wrapper around FileReader for loading multiple files at once.

    Parameters
    ----------
    fvcom : list-like, str
        List of files to load.
    noisy : bool, optional
        Set to True to write out the name of each file being loaded.

    Additional arguments are passed to `PyFVCOM.read.FileReader'.

    Returns
    -------
    FVCOM : PyFVCOM.read.FileReader
        Concatenated data from the files in `fvcom'.

    """

    if isinstance(fvcom, str):
        if noisy:
            print('Loading {}'.format(fvcom))
        FVCOM = FileReader(fvcom, *args, **kwargs)
    else:
        for file in fvcom:
            if noisy:
                print('Loading {}'.format(file))
            if file == fvcom[0]:
                FVCOM = FileReader(file, *args, **kwargs)
            else:
                FVCOM += FileReader(file, *args, **kwargs)

    return FVCOM


class FileReaderFromDict(FileReader):
    """
    Convert an ncread dictionary into a (sparse) FileReader object. This does a passable job of impersonating a full
    FileReader object if you've loaded data with ncread.

    """

    def __init__(self, fvcom):
        """
        Will initialise a FileReader object from an ncread dictionary. Some attempt is made to fill in missing
        information (dimensions mainly).

        Parameters
        ----------
        fvcom : dict
            Output of ncread.

        """

        # Prepare this object with all the objects we'll need later on (data, dims, time, grid, atts).
        self._prep()

        grid_names = ('lon', 'lat', 'lonc', 'latc', 'nv',
                      'h', 'h_center',
                      'nbe', 'ntsn', 'nbsn', 'ntve', 'nbve',
                      'art1', 'art2', 'a1u', 'a2u',
                      'siglay', 'siglev')
        time_names = ('time', 'Times', 'datetime', 'Itime', 'Itime2')

        for key in fvcom:
            if key in grid_names:
                setattr(self.grid, key, fvcom[key])
            elif key in time_names:
                setattr(self.time, key, fvcom[key])
            else:  # assume data.
                setattr(self.data, key, fvcom[key])
        # Make some dimensions
        self.dims.three = 3
        self.dims.four = 4
        self.dims.maxnode = 11
        self.dims.maxelem = 9
        # This is a little repetitive (each dimension can be set multiple times), but it has simplicity to its
        # advantage.
        for obj in self.data:
            if obj in ('ua', 'va'):
                try:
                    self.dims.time, self.dims.nele = getattr(self.data, obj).shape
                except ValueError:
                    # Assume we've got a single position.
                    self.dims.time = getattr(self.data, obj).shape[0]
                    self.dims.nele = 1
            elif obj in ('temp', 'salinity'):
                try:
                    self.dims.time, self.dims.siglay, self.dims.node = getattr(self.data, obj).shape
                except ValueError:
                    # Assume we've got a single position
                    self.dims.time, self.dims.siglay = getattr(self.data, obj).shape[:2]
                    self.dims.node = 1
                self.dims.siglev = self.dims.siglay + 1
            elif obj in ['zeta']:
                try:
                    self.dims.time, self.dims.node = getattr(self.data, obj).shape
                except ValueError:
                    # Assume we've got a single position
                    self.dims.time = getattr(self.data, obj).shape[0]
                    self.dims.node = 1
            elif obj in ('Times'):
                self.dims.time, self.dims.DateStrLen = getattr(self.time, obj).shape
            elif obj in ('time', 'Itime', 'Itime2', 'datetime'):
                self.dims.time = getattr(self.time, obj).shape


class SubDomainReader(FileReader):
    """
    Create a sub-domain based on either a supplied polygon or an interactively created one.

    """
    def __init__(self, *args, **kwargs):
        # Automatically inherits the FileReader docstring.
        self._bounding_box = True
        super().__init__(*args, **kwargs)

    def _make_subset_dimensions(self):
        """
        If the 'wesn' keyword has been included in the supplied dimensions, interactively select a region if the
        value of 'wesn' is not a shapely Polygon. If it is a shapely Polygon, use that for the subsetting.

        This mimics the function of PyFVCOM.read.FileReader._make_subset_dimensions but with greater flexibility in
        terms of the region to subset.

        """

        if 'wesn' in self._dims:
            if isinstance(self._dims['wesn'], Polygon):
                bounding_poly = np.asarray(self._dims['wesn'].exterior.coords)
            # Drop the 'wesn' dimension now as it's not necessary for anything else.
            self._dims.pop('wesn')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(self.grid.lon, self.grid.lat, c='lightgray')
            plt.show()

            keep_picking = True
            while keep_picking:
                n_pts = int(input('How many polygon points? '))
                bounding_poly = np.full((n_pts, 2), np.nan)
                poly_lin = []
                for point in range(n_pts):
                    bounding_poly[point, :] = plt.ginput(1)[0]
                    poly_lin.append(ax.plot(np.hstack([bounding_poly[:, 0], bounding_poly[0, 0]]),
                                       np.hstack([bounding_poly[:, 1], bounding_poly[0,1]]),
                                       c='r', linewidth=2)[0])
                    fig.canvas.draw()

                happy = input('Is that polygon OK? Y/N: ')
                if happy.lower() == 'y':
                    keep_picking = False

                for this_l in poly_lin:
                    this_l.remove()

            plt.close()

        poly_path = mpath.Path(bounding_poly)

        # Shouldn't need the np.asarray here, I think, but leaving it in as I'm not 100% sure.
        self._dims['node'] = np.squeeze(np.argwhere(np.asarray(poly_path.contains_points(np.asarray([self.grid.lon, self.grid.lat]).T))))
        self._dims['nele'] = np.squeeze(np.argwhere(np.all(np.isin(self.grid.triangles, self._dims['node']), axis=1)))
        self._get_data_pattern == 'memory'

    def _find_open_faces(self):
        """
        TODO: docstring.

        """
        vol_cells_ext = np.hstack([self._dims['nele'], -1])  # closed boundaries are given a -1 in the nbe matrix
        open_sides = np.where(~np.isin(self.grid.nbe, vol_cells_ext))
        open_side_cells = open_sides[0]

        open_side_rows = self.grid.triangles[open_side_cells, :]
        open_side_nodes = []
        row_choose = np.asarray([0, 1, 2])
        for this_row, this_not in zip(open_side_rows, open_sides[1]):
            this_row_choose = row_choose[~np.isin(row_choose, this_not)]
            open_side_nodes.append(this_row[this_row_choose])
        open_side_nodes = np.asarray(open_side_nodes)

        open_side_dict = {}

        for this_cell in open_side_cells:
            this_cell_all_nodes = self.grid.triangles[this_cell, :]
            this_cell_nodes = this_cell_all_nodes[np.isin(this_cell_all_nodes, open_side_nodes)]

            vector_pll = [self.grid.x[this_cell_nodes[0]] - self.grid.x[this_cell_nodes[1]],
                          self.grid.y[this_cell_nodes[0]] - self.grid.y[this_cell_nodes[1]]]
            vector_nml = np.asarray([vector_pll[1], -vector_pll[0]]) / np.sqrt(vector_pll[0]**2 + vector_pll[1]**2)
            epsilon = 0.0001
            mid_point = np.asarray([self.grid.x[this_cell_nodes[0]] + 0.5 * (self.grid.x[this_cell_nodes[1]] - self.grid.x[this_cell_nodes[0]]),
                                    self.grid.y[this_cell_nodes[0]] + 0.5 * (self.grid.y[this_cell_nodes[1]] - self.grid.y[this_cell_nodes[0]])])

            cell_path = mpath.Path(np.asarray([self.grid.x[this_cell_all_nodes], self.grid.y[this_cell_all_nodes]]).T)

            if cell_path.contains_point(mid_point + epsilon * vector_nml):  # want outward pointing normal
                vector_nml = -1 * vector_nml

            side_length = np.sqrt((self.grid.x[this_cell_nodes[0]] - self.grid.x[this_cell_nodes[1]])**2 +
                                  (self.grid.y[this_cell_nodes[0]] - self.grid.y[this_cell_nodes[1]])**2)

            open_side_dict[this_cell] = [vector_nml, this_cell_nodes, side_length]

        self.open_side_dict = open_side_dict

    def _generate_open_vol_flux(self, noisy=False):
        """
        TODO: docstring.

        """

        # Get a bunch of stuff if not already calculated

        if not hasattr(self, 'open_side_dict'):
            if noisy:
                print('Open faces not identified yet, running _find_open_faces()')
            self._find_open_faces()
        open_face_cells = np.asarray(list(self.open_side_dict.keys()))
        open_face_vel = {}  # currently unused

        if not hasattr(self.grid, 'depth'):
            if noisy:
                print('Time varying depth not preloaded, fetching')
            self._get_cv_volumes()

        if not hasattr(self.data, 'u'):
            if noisy:
                print('U data not preloaded, fetching')
            self.load_data(['u'])
            u_openface = self.data.u[..., open_face_cells]
            delattr(self.data, 'u')
        else:
            u_openface = self.data.u[..., open_face_cells]

        if not hasattr(self.data, 'v'):
            if noisy:
                print('V data not preloaded, fetching')
            self.load_data(['v'])
            v_openface = self.data.v[..., open_face_cells]
            delattr(self.data, 'v')
        else:
            v_openface = self.data.v[..., open_face_cells]

        # Loop through each open boundary cell, get the normal component of the velocity, 
        # calculate the (time-varying) area of the open face at each sigma layer, then add this to the flux dictionary
        if noisy:
            print('{} open boundary cells'.format(len(open_face_cells)))

        open_side_flux_dict = {}
        for this_open_cell, this_open_data in self.open_side_dict.items():
            if noisy:
                print('Adding flux for open cell {}'.format(this_open_cell))
            this_cell_ind = open_face_cells == this_open_cell
            this_cell_vel = [np.squeeze(u_openface[..., this_cell_ind]), np.squeeze(v_openface[..., this_cell_ind])]

            this_normal_vec = this_open_data[0]
            this_dot = np.squeeze(np.asarray(this_cell_vel[0] * this_normal_vec[0] + this_cell_vel[1] * this_normal_vec[1]))

            this_cell_nodes = this_open_data[1]
            this_cell_deps = self.grid.depth[:, this_cell_nodes]
            this_cell_siglev = self.grid.siglev[:, this_cell_nodes]

            this_cell_deps_siglev = -np.tile(this_cell_siglev, [this_cell_deps.shape[0], 1, 1]) * np.transpose(np.tile(this_cell_deps, [this_cell_siglev.shape[0], 1, 1]), (1, 0, 2))
            this_cell_deps_siglev_abs = np.tile(self.data.zeta[:, np.newaxis, this_cell_nodes], [1, this_cell_deps_siglev.shape[1], 1]) - this_cell_deps_siglev

            this_cell_dz = this_cell_deps_siglev[:, :-1, :] - this_cell_deps_siglev[:, 1:, :]

            this_node1_xyz = [np.tile(self.grid.x[this_cell_nodes[0]], this_cell_deps_siglev.shape[:2]),
                              np.tile(self.grid.y[this_cell_nodes[0]], this_cell_deps_siglev.shape[:2]),
                              this_cell_deps_siglev_abs[:, :, 1]]
            this_node2_xyz = [np.tile(self.grid.x[this_cell_nodes[1]], this_cell_deps_siglev.shape[:2]),
                              np.tile(self.grid.y[this_cell_nodes[1]], this_cell_deps_siglev.shape[:2]),
                              this_cell_deps_siglev_abs[:, :, 1]]

            this_cell_cross = np.sqrt((this_node1_xyz[0] - this_node2_xyz[0])**2 +
                                      (this_node1_xyz[1] - this_node2_xyz[1])**2 +
                                      (this_node1_xyz[2] - this_node2_xyz[2])**2)
            this_cell_hyps = np.sqrt((this_node1_xyz[0][:, 1:] - this_node2_xyz[0][:, :-1])**2 +
                                     (this_node1_xyz[1][:, 1:] - this_node2_xyz[1][:, :-1])**2 +
                                     (this_node1_xyz[2][:, 1:] - this_node2_xyz[2][:, :-1])**2)

            area_tri_1 = get_area_heron(np.abs(this_cell_dz[:, :, 0]), this_cell_cross[:, :-1], this_cell_hyps)
            area_tri_2 = get_area_heron(np.abs(this_cell_dz[:, :, 1]), this_cell_cross[:, 1:], this_cell_hyps)

            this_area = area_tri_1 + area_tri_2
            this_vol_flux = this_area * this_dot

            open_side_flux_dict[this_open_cell] = [this_dot, this_area, this_vol_flux]

        self.open_side_flux = open_side_flux_dict

    def add_evap_precip(self):
        self.load_data(['precip', 'evap'])

    def add_river_data(self):
        nml_dict = get_river_config(river_nml_file)
        river_node_raw = np.asarray(nml_dict['RIVER_GRID_LOCATION'], dtype=int) - 1

        # Get only rivers which feature in the subdomain
        rivers_in_grid = np.isin(river_node_raw, self._dims['node'])

        river_nc = nc.Dataset(river_nc_file, 'r')
        time_raw = river_nc.variables['Times'][:]
        time_dt = [datetime.strptime(b''.join(this_time).decode('utf-8'), '%Y-%m-%d %H:%M:%S') for this_time in time_raw]

        ref_date = datetime(1900,1,1)
        mod_time_sec = [(this_dt - ref_date).total_seconds() for this_dt in self.time.datetime]
        river_time_sec = [(this_dt - ref_date).total_seconds() for this_dt in time_dt]

        self.river.river_nodes = np.argwhere(np.isin(self._dims['node'], river_node_raw))

        river_flux_raw = river_nc.variables['river_flux'][:, rivers_in_grid]
        self.river.river_fluxes = np.asarray([np.interp(mod_time_sec, river_time_sec, this_col) for this_col in river_flux_raw.T]).T
        self.river.total_flux = np.sum(self.river.river_fluxes, axis=1)

        river_temp_raw = river_nc.variables['river_temp'][:, rivers_in_grid]
        self.river.river_temp = np.asarray([np.interp(mod_time_sec, river_time_sec, this_col) for this_col in river_temp_raw.T]).T

        river_salt_raw = river_nc.variables['river_salt'][:, rivers_in_grid]
        self.river.river_salt = np.asarray([np.interp(mod_time_sec, river_time_sec, this_col) for this_col in river_salt_raw.T]).T

    def aopen_integral(self, var):
        """
        TODO: docstring.
        TODO: finish.

        """
        var_to_int = getattr(self.data, var)
        if len(var_to_int) == len(self.grid.xc):
            var_to_int = elems2nodes(var, self.grid.triangles)
        
        

        return var_to_int

    def volume_integral(self, var):
        """
        TODO: docstring.
        TODO: finish.

        """
        var_to_int = getattr(self.data, var)
        if len(var_to_int) == len(self.grid.xc):
            var_to_int = elems2nodes(var, self.grid.triangles)

        if not hasattr(self, 'volume'):
            self.get_cv_volumes()

        if not hasattr(self.data, var):
            self._get_variable(var)

        setattr(self, var + '_total', np.sum(np.sum(getattr(self.data, var) * self.volume, axis=2), axis=1))

    def surface_integral(self, var):
        """
        TODO: docstring.
        TODO: finish.

        """
        pass


def time_to_index(times, target_time, tolerance=False):
    """
    Find the time index for the given time string (%Y-%m-%d %H:%M:%S.%f) or datetime object.

    Parameters
    ----------
    times : list
        List of datetime objects from which to find the closest `target_time'.
    target_time : str or datetime.datetime
        Time for which to find the time index from `times'. If given as a string, the time format must be "%Y-%m-%d
        %H:%M:%S.%f".
    tolerance : float, optional
        Seconds of tolerance to allow when finding the appropriate index. Use this flag to only return an index
        to within some tolerance. By default, the closest time is returned irrespective of how far in time it is
        from the data.

    Returns
    -------
    time_idx : int
        Index for the currently loaded data closest to the specified time.

    """

    if not isinstance(target_time, datetime):
        try:
            target_time = datetime.strptime(target_time, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            # Try again in case we've not been given fractional seconds, just to be nice.
            target_time = datetime.strptime(target_time, '%Y-%m-%d %H:%M:%S')

    time_diff = np.abs(times - target_time)
    if not tolerance:
        time_idx = np.argmin(time_diff)
    else:
        if np.min(time_diff) <= timedelta(seconds=tolerance):
            time_idx = np.argmin(time_diff)
        else:
            time_idx = None

    return time_idx


class ncwrite(object):
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
    quiet : bool
        Set to True for verbose output. Defaults to False.
    format : str
        Pick the netCDF file format. Defaults to 'NETCDF3_CLASSIC'. See
        `Dataset` for more types.

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

    def __init__(self, input_dict, filename_out, quiet=False, format='NETCDF3_CLASSIC'):
        self.filename_out = filename_out
        self.input_dict = input_dict
        self.quiet = quiet
        self.format = format
        self.createNCDF()

    def createNCDF(self):
        """
        Function to create and write the data to the specified netCDF file.

        """

        rootgrp = Dataset(self.filename_out, 'w', format=self.format, clobber=True)

        # Create dimensions.
        if 'dimensions' in self.input_dict:
            for k, v in self.input_dict['dimensions'].items():
                rootgrp.createDimension(k, v)
        else:
            if not self.quiet:
                print('No netCDF created:')
                print('  No dimension key found (!! has to be \"dimensions\"!!!)')
            return()

        # Create global attributes.
        if 'global attributes' in self.input_dict:
            for k, v in self.input_dict['global attributes'].items():
                rootgrp.setncattr(k, v)
        else:
            if not self.quiet:
                print('  No global attribute key found (!! has to be \"global attributes\"!!!)')

        # Create variables.
        for k, v in self.input_dict['variables'].items():
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
            if not self.quiet:
                print('  Creating variable: {} {} {}'.format(k, data_type, dims))
            var = rootgrp.createVariable(k, data_type, dims, fill_value=fill_value)
            if len(dims) > np.ndim(data):
                # If number of dimensions given to netCDF is greater than the
                # number of dimension of the data, then fill the netCDF
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
                    if not self.quiet:
                        print('Problem in the number of dimensions')
            else:
                try:
                    var[:] = data
                except IndexError:
                    raise(IndexError(('Supplied data shape {} does not match the specified'
                    ' dimensions {}, for variable \'{}\'.'.format(data.shape, var.shape, k))))

            # Create attributes for variables
            if 'attributes' in self.input_dict['variables'][k]:
                for ka, va in self.input_dict['variables'][k]['attributes'].items():
                    var.setncattr(ka, va)

        rootgrp.close()


def ncread(file, vars=None, dims=False, noisy=False, atts=False, datetimes=False):
    """
    Read in the FVCOM results file and spit out numpy arrays for each of the
    variables specified in the vars list.

    Optionally specify a dict with keys whose names match the dimension names
    in the netCDF file and whose values are strings specifying alternative
    ranges or lists of indices. For example, to extract the first hundred time
    steps, supply dims as:

        dims = {'time':'0:100'}

    To extract the first, 400th and 10,000th values of any array with nodes:

        dims = {'node':'[0, 3999, 9999]'}

    Any dimension not given in dims will be extracted in full.

    Specify atts=True to extract the variable attributes. Set datetimes=True
    to convert the FVCOM Modified Julian Day values to python datetime objects.

    Parameters
    ----------
    file : str, list
        If a string, the full path to an FVCOM netCDF output file. If a list,
        a series of files to be loaded. Data will be concatenated into a single
        dict.
    vars : list, optional
        List of variable names to be extracted. If omitted, all variables are
        returned.
    dims : dict, optional
        Dict whose keys are dimensions and whose values are a string of either
        a range (e.g. {'time':'0:100'}) or a list of individual indices (e.g.
        {'time':'[0, 1, 80, 100]'}). Slicing is supported (::5 for every fifth
        value).
    noisy : bool, optional
        Set to True to enable verbose output.
    atts : bool, optional
        Set to True to enable output of the attributes (defaults to False).
    datetimes : bool, optional
        Set to True to convert FVCOM time to Python datetime objects (creates
        a new `datetime' key in the output dict. Only applies if `vars'
        includes either the `Times' or `time' variables. Note: if FVCOM has
        been run with single precision output, then the conversion of the
        `time' values to a datetime object suffers rounding errors. It's
        best to either run FVCOM in double precision or specify only the
        `Times' data in the `vars' list.

    Returns
    -------
    FVCOM : dict
        Dict of data extracted from the netCDF file. Keys are those given in
        vars and the data are stored as ndarrays. If `datetimes' is True,
        then this also includes a `datetime' key in which is the FVCOM
        time series converted to Python datetime objects.
    attributes : dict, optional
        If atts=True, returns the attributes as a dict for each
        variable in vars. The key `dims' contains the array dimensions (each
        variable contains the names of its dimensions) as well as the shape of
        the dimensions defined in the netCDF file. The key `global' contains
        the global attributes.

    See Also
    --------
    read_probes : read in FVCOM ASCII probes output files.

    """

    # Set to True when we've converted from Modified Julian Day so we don't
    # end up doing the conversion twice, once for `Times' and again for
    # `time' if both variables have been requested in `vars'.
    done_datetimes = False
    got_itime = False
    got_itime2 = False
    # Check whether we'll be able to fulfill the datetime request.
    if datetimes and vars and not list(set(vars) & set(('Times', 'time', 'Itime', 'Itime2'))):
        raise ValueError("Conversion to python datetimes has been requested "
                         "but no time variable (`Times', `time', `Itime' or `Itime2') has been "
                         "requested in vars.")

    # If we have a list, assume it's lots of files and load them all. Only use
    # MFDataset on lists of more than 1 file.
    if isinstance(file, list) and len(file) > 1:
        try:
            try:
                rootgrp = MFDataset(file, 'r')
            except IOError as msg:
                raise IOError('Unable to open file {} ({}). Aborting.'.format(file, msg))
        except:
            # Try aggregating along a 'time' dimension (for POLCOMS,
            # for example).
            try:
                rootgrp = MFDataset(file, 'r', aggdim='time')
            except IOError as msg:
                raise IOError('Unable to open file {} ({}). Aborting.'.format(file, msg))
    elif isinstance(file, list) and len(file) == 1:
        rootgrp = Dataset(file[0], 'r')
    else:
        rootgrp = Dataset(file, 'r')

    # Create a dict of the dimension names and their current sizes
    read_dims = {}
    for key, var in list(rootgrp.dimensions.items()):
        # Make the dimensions ranges so we can use them to extract all the
        # values.
        read_dims[key] = '0:{}'.format(str(len(var)))

    # Compare the dimensions in the netCDF file with those provided. If we've
    # been given a dict of dimensions which differs from those in the netCDF
    # file, then use those.
    if dims:
        common_keys = set(read_dims).intersection(list(dims.keys()))
        for k in common_keys:
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
            print('Found {}'.format(key), end=' ')
            sys.stdout.flush()

        if key in vars:
            var_dims = rootgrp.variables[key].dimensions

            to_extract = [read_dims[d] for d in var_dims]

            # If we have no dimensions, we must have only a single value, in
            # which case set the dimensions to empty and append the function to
            # extract the value.
            if not to_extract:
                to_extract = '.getValue()'

            # Thought I'd finally figured out how to replace the eval approach,
            # but I still can't get past the indexing needed to be able to
            # subset the data.
            # FVCOM[key] = rootgrp.variables.get(key)[0:-1]
            # I know, I know, eval() is evil.
            get_data = 'rootgrp.variables[\'{}\']{}'.format(key, str(to_extract).replace('\'', ''))
            FVCOM[key] = eval(get_data)

            # Get all attributes for this variable.
            if atts:
                attributes[key] = {}
                # Grab all the attributes for this variable.
                for varatt in rootgrp.variables[key].ncattrs():
                    attributes[key][varatt] = getattr(rootgrp.variables[key], varatt)

            if datetimes and key in ('Times', 'time', 'Itime', 'Itime2') and not done_datetimes:
                # Convert the time data to datetime objects. How we do this
                # depends on which we hit first - `Times', `time', `Itime' or
                # `Itime2'. For the former, we need to parse the strings, for the
                # latter we can leverage num2date from the netCDF4 module and
                # use the time units attribute.
                if key == 'Times':
                    try:
                        # Check if we've only extracted a single time step, in
                        # which case we don't need to use a list comprehension
                        # to get the datetimes.
                        if isinstance(FVCOM[key][0], np.ndarray):
                            FVCOM['datetime'] = np.asarray([datetime.strptime(''.join(i).strip(), '%Y-%m-%dT%H:%M:%S.%f') for i in FVCOM[key].astype(str)])
                        else:
                            FVCOM['datetime'] = np.asarray(datetime.strptime(''.join(FVCOM[key].astype(str)).strip(), '%Y-%m-%dT%H:%M:%S.%f'))
                    except ValueError:
                        # Try a different format before bailing out.
                        if isinstance(FVCOM[key][0], np.ndarray):
                            FVCOM['datetime'] = np.asarray([datetime.strptime(''.join(i).strip(), '%Y/%m/%d %H:%M:%S.%f') for i in FVCOM[key].astype(str)])
                        else:
                            FVCOM['datetime'] = np.asarray(datetime.strptime(''.join(FVCOM[key].astype(str)).strip(), '%Y/%m/%d %H:%M:%S.%f'))
                    done_datetimes = True
                elif key == 'time':
                    FVCOM['datetime'] = num2date(FVCOM[key],
                                                 rootgrp.variables[key].units)
                    done_datetimes = True
                elif key == 'Itime':
                    got_itime = True
                elif key == 'Itime2':
                    got_itime2 = True

            if noisy:
                if len(str(to_extract)) < 60:
                    print('(extracted {})'.format(str(to_extract).replace('\'', '')))
                else:
                    print('(extracted given indices)')

        elif noisy:
                print()

    # If: 1. we haven't got datetime in the output 2. we've been asked to get it and 3. we've got both Itime and
    # Itime2, then make datetime from those.
    if datetimes and got_itime and got_itime2 and 'datetime' not in FVCOM:
        FVCOM['datetime'] = num2date(FVCOM['Itime'] + (FVCOM['Itime2'] / 1000 / 24 / 60 / 60),
                                     rootgrp.variables['Itime'].units)

    # Close the open file.
    rootgrp.close()

    if atts:
        return FVCOM, attributes
    else:
        return FVCOM


def read_probes(files, noisy=False, locations=False, datetimes=False):
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
    datetimes : bool, optional
        Set to True to convert times to datetimes. Defaults to False (returns
        modified Julian Days).

    Returns
    -------
    times : np.ndarray
        Modified Julian Day times for the extracted time series.
    values : np.ndarray
        Array of the extracted time series values.
    positions : np.ndarray, optional
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

    if not isinstance(files, (list, tuple)):
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
            try:
                times = data[:, 0]
                values = data[:, 1:]
            except IndexError:
                times = data[0]
                values = data[1:]
            positions = lonlatz
        else:
            try:
                times = np.hstack((times, data[:, 0]))
                values = np.vstack((values, data[:, 1:]))
            except IndexError:
                times = np.hstack((times, data[0]))
                values = np.vstack((values, data[1:]))

            positions = np.vstack((positions, lonlatz))

        if noisy:
            print('done.')

    if datetimes:
        times = num2date(times, "days since 1858-11-17 00:00:00")

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
    file : str
        File to which to save the probes.
    mjd : np.ndarray, list, tuple
        Date/time in Modified Julian Day
    timeseries : np.ndarray
        Data to write out (vector/array for 1D/2D). Shape should be
        [time, values], where values can be 1D or 2D.
    datatype : tuple, list, tuple
        List with the metadata. Give the long name (e.g. `Temperature') and the
        units (e.g. `Celsius').
    site : str
        Name of the output location.
    depth : float
        Depth at the time series location.
    sigma : np.ndarray, list, tupel, optional
        Start and end indices of the sigma layer of time series (if
        depth-resolved, -1 otherwise).
    lonlat : np.ndarray, list, optional
        Coordinates (spherical)
    xy : np.ndarray, list, optional
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
            fmt = '{}\n'.format(fmt.strip())

        # Dump the data (this may be slow).
        for line in np.column_stack((mjd, timeseries)):
            f.write(fmt.format(*line))


def get_river_config(file_name, noisy=False, zeroindex=False):
    """
    Parse an FVCOM river namelist file to extract the parameters and their values. Returns a dict of the parameters
    with the associated values for all the rivers defined in the namelist.

    Parameters
    ----------
    file_name : str
        Full path to an FVCOM Rivers name list.
    noisy : bool, optional
        Set to True to enable verbose output. Defaults to False.
    zeroindex : bool, optional
        Set to True to convert indices from 1-based to 0-based. Defaults to False.

    Returns
    -------
    rivers : dict
        Dict of the parameters for each river defined in the name list.
        Dictionary keys are the name list parameter names (e.g. RIVER_NAME).

    Notes
    -----

    The indices returned in RIVER_GRID_LOCATION are 1-based (i.e. read in raw
    from the nml file). For use in Python, you can either subtract 1 yourself,
    or pass zeroindex=True to this function.

    """

    rivers = {}
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()

            if line and not line.startswith('&') and not line.startswith('/'):
                param, value = [i.strip(",' ") for i in line.split('=')]
                if param in rivers:
                    rivers[param].append(value)
                else:
                    rivers[param] = [value]

        if noisy:
            print('Found {} rivers.'.format(len(rivers['RIVER_NAME'])))

    if zeroindex and 'RIVER_GRID_LOCATION' in rivers:
        rivers['RIVER_GRID_LOCATION'] = [int(i) - 1 for i in rivers['RIVER_GRID_LOCATION']]

    return rivers
