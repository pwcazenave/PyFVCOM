# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys
import copy
import inspect

import numpy as np
import matplotlib.tri as tri

from warnings import warn
from datetime import datetime, timedelta
from netCDF4 import Dataset, MFDataset, num2date, date2num

from PyFVCOM.coordinate import lonlat_from_utm, utm_from_lonlat
from PyFVCOM.grid import unstructured_grid_volume, nodes2elems, vincenty_distance, haversine_distance


class FileReader(object):
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

    def __init__(self, fvcom, variables=[], dims=[], zone='30N', debug=False):
        """
        Parameters
        ----------
        fvcom : str
            Path to an FVCOM netCDF.
        variables : list-like, optional
            List of variables to extract. If omitted, no variables are extracted, which means you won't be able to
            add this object to another one which does have variables in it.
        dims : dict, optional
            Dictionary of dimension names along which to subsample e.g. dims={'time': [0, 100], 'nele': [0, 10, 100],
            'node': 100}.
            Times (time) is specified as a range; vertical (siglay, siglev) and horizontal dimensions (node,
            nele) can be list-like. Time can also be specified as either a time string (e.g. '2000-01-25
            23:00:00.00000') or given as a datetime object.
            Any combination of dimensions is possible; omitted dimensions are loaded in their entirety.
            To extract a single time (e.g. the 9th in python indexing), specify the range as [9, 10].
            Negative indices are supported. To load from the 10th to the last time can be written as 'time': [9, -1]).
            A special dimension of 'wesn' can be used to specify a bounding box within which to extract the model
            grid and data.
        zone : str, list-like, optional
            UTM zones (defaults to '30N') for conversion of UTM to spherical coordinates.
        debug : bool, optional
            Set to True to enable debug output. Defaults to False.

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
        self._fvcom = fvcom
        self._zone = zone
        self._bounding_box = False
        # We may modify the dimensions (for negative indexing), so make a deepcopy (copy isn't sufficient) so
        # successive calls to FileReader from MFileReader work properly.
        self._dims = copy.deepcopy(dims)
        # Silently convert a string variable input to an iterable list.
        if isinstance(variables, str):
            variables = [variables]
        self._variables = variables

        # Prepare this object with all the objects we'll need later on (data, dims, time, grid, atts).
        self._prep()

        # Get the things to iterate over for a given object. This is a bit hacky, but until or if I create separate
        # classes for the dims, time, grid and data objects, this'll have to do.
        self.obj_iter = lambda x: [a for a in dir(x) if not a.startswith('__')]

        self.ds = Dataset(self._fvcom, 'r')

        for dim in self.ds.dimensions:
            setattr(self.dims, dim, self.ds.dimensions[dim].size)

        # Convert negative indexing to positive in dimensions to extract. We do this since we end up using range for
        # the extraction of each dimension since you can't (easily) pass slices as arguments.
        for dim in self._dims:
            negatives = [i < 0 for i in self._dims[dim] if isinstance(i, int)]
            if negatives and dim != 'wesn':
                for i, value in enumerate(negatives):
                    if value:
                        self._dims[dim][i] = self._dims[dim][i] + self.ds.dimensions[dim].size + 1
            # If we've been given a region to load (W/E/S/N), set a flag to extract only nodes and elements which
            # fall within that region.
            if dim == 'wesn':
                self._bounding_box = True

        self._load_time()
        self._load_grid()

        if variables:
            self._load_data(self._variables)

    def __eq__(self, other):
        # For easy comparison of classes.
        return self.__dict__ == other.__dict__

    def __add__(self, FVCOM, debug=False):
        """
        This special method means we can stack two FileReader objects in time through a simple addition (e.g. fvcom1
        += fvcom2)

        Parameters
        ----------
        FVCOM : PyFVCOM.FileReader
            Previous time to which to add ourselves.

        Returns
        -------
        NEW : PyFVCOM.FileReader
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
        node_compare = self.dims.nele == FVCOM.dims.nele
        nele_compare = self.dims.node == FVCOM.dims.node
        siglay_compare = self.dims.siglay == FVCOM.dims.siglay
        siglev_compare = self.dims.siglev == FVCOM.dims.siglev
        time_compare = self.time.datetime[-1] <= FVCOM.time.datetime[0]
        data_compare = self.obj_iter(self.data) == self.obj_iter(FVCOM.data)
        old_data = self.obj_iter(self.data)
        new_data = self.obj_iter(FVCOM.data)
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
                                                                                    FVCOM.time.datetime[0]))
        if not data_compare:
            raise ValueError('Loaded data sets for each FVCOM class must match.')
        if not (old_data == new_data) and (old_data or new_data):
            warn('Subsequent attempts to load data for this merged object will only load data from the first object. '
                 'Load data into each object before merging them.')

        # Copy ourselves to a new version for concatenation. self is the old so we get appended to by the new.
        idem = copy.copy(self)

        # Go through all the parts of the data with a time dependency and concatenate them. Leave the grid alone.
        for var in self.obj_iter(idem.data):
            if 'time' in idem.ds.variables[var].dimensions:
                setattr(idem.data, var, np.concatenate((getattr(idem.data, var), getattr(FVCOM.data, var))))
        for time in self.obj_iter(idem.time):
            setattr(idem.time, time, np.concatenate((getattr(idem.time, time), getattr(FVCOM.time, time))))

        # Remove duplicate times.
        time_indices = np.arange(len(idem.time.time))
        _, dupes = np.unique(idem.time.time, return_index=True)
        dupe_indices = np.setdiff1d(time_indices, dupes)
        for var in self.obj_iter(idem.data):
            # Only delete things with a time dimension.
            if 'time' in idem.ds.variables[var].dimensions:
                time_axis = idem.ds.variables[var].dimensions.index('time')
                setattr(idem.data, var, np.delete(getattr(idem.data, var), dupe_indices, axis=time_axis))
        for time in self.obj_iter(idem.time):
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

    def _prep(self):
        # Create empty object for the grid, dimension, data and time data. This ought to be possible with nested
        # classes, but I can't figure it out. That approach would also mean we can set __iter__ to make the object
        # iterable without the need for obj_iter, which is a bit of a hack. It might also make FileReader object
        # pickleable, meaning we can pass them with multiprocessing. Another day, perhaps.
        self.data = type('data', (object,), {})()
        self.dims = type('dims', (object,), {})()
        self.atts = type('atts', (object,), {})()
        self.grid = type('grid', (object,), {})()
        self.time = type('time', (object,), {})()

        # Add docstrings for the relevant objects.
        self.data.__doc__ = "This object will contain data as loaded from the netCDFs specified. Use " \
                            "`FileReader.load_data' to get specific data (optionally at specific locations, times and" \
                            " depths)."
        self.dims.__doc__ = "This contains the dimensions of the data from the given netCDFs."
        self.atts.__doc__ = "This contains the attributes for each variable in the given netCDFs as a dictionary."
        self.grid.__doc__ = "FVCOM grid information. Missing spherical or cartesian coordinates are automatically " \
                            "created depending on which is missing."
        self.time.__doc__ = "This contains the time data for the given netCDFs. Missing standard FVCOM time variables " \
                            "are automatically created."

    def _load_time(self):
        """ Populate a time object with additional useful time representations from the netCDF time data. """

        time_variables = ('time', 'Times', 'Itime', 'Itime2')
        got_time, missing_time = [], []
        for time in time_variables:
            # Since not all of the time_variables specified above are required, only try to load the data if they
            # exist. We'll raise an error if we don't find any of them though.
            if time in self.ds.variables:
                setattr(self.time, time, self.ds.variables[time][:])
                got_time.append(time)
                attributes = type('attributes', (object,), {})()
                for attribute in self.ds.variables[time].ncattrs():
                    setattr(attributes, attribute, getattr(self.ds.variables[time], attribute))
                setattr(self.atts, time, attributes)
            else:
                missing_time.append(time)

        if len(missing_time) == len(time_variables):
            warn('No time variables found in the netCDF.')
        else:
            if 'Times' in got_time:
                # Overwrite the existing Times array with a more sensibly shaped one.
                self.time.Times = np.asarray([''.join(t.astype(str)).strip() for t in self.time.Times])

            # Make whatever we got into datetime objects and use those to make everything else. Note: the `time' variable
            # is often the one with the lowest precision, so use the others preferentially over that.
            if 'Times' not in got_time:
                if 'time' in got_time:
                    _dates = num2date(self.time.time, units=getattr(self.ds.variables['time'], 'units'))
                elif 'Itime' in got_time and 'Itime2' in got_time:
                    _dates = num2date(self.time.Itime + self.time.Itime2 / 1000.0 / 60 / 60, units=getattr(self.ds.variables['Itime'], 'units'))
                try:
                    self.time.Times = np.array([datetime.strftime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in _dates])
                except ValueError:
                    self.time.Times = np.array([datetime.strftime(d, '%Y/%m/%d %H:%M:%S.%f') for d in _dates])
                # Add the relevant attribute for the Times variable.
                attributes = type('attributes', (object,), {})()
                setattr(attributes, 'time_zone', 'UTC')
                setattr(self.atts, 'Times', attributes)

            if 'time' not in got_time:
                if 'Times' in got_time:
                    try:
                        _dates = np.array([datetime.strptime(''.join(t.astype(str)).strip(), '%Y-%m-%dT%H:%M:%S.%f') for t in self.time.Times])
                    except ValueError:
                        _dates = np.array([datetime.strptime(''.join(t.astype(str)).strip(), '%Y/%m/%d %H:%M:%S.%f') for t in self.time.Times])
                elif 'Itime' in got_time and 'Itime2' in got_time:
                    _dates = num2date(self.time.Itime + self.time.Itime2 / 1000.0 / 60 / 60, units=getattr(self.ds.variables['Itime'], 'units'))
                # We're making Modified Julian Days here to replicate FVCOM's 'time' variable.
                self.time.time = date2num(_dates, units='days since 1858-11-17 00:00:00')
                # Add the relevant attributes for the time variable.
                attributes = type('attributes', (object,), {})()
                setattr(attributes, 'units', 'days since 1858-11-17 00:00:00')
                setattr(attributes, 'long_name', 'time')
                setattr(attributes, 'format', 'modified julian day (MJD)')
                setattr(attributes, 'time_zone', 'UTC')
                setattr(self.atts, 'time', attributes)

            if 'Itime' not in got_time and 'Itime2' not in got_time:
                if 'Times' in got_time:
                    try:
                        _dates = np.array([datetime.strptime(''.join(t.astype(str)).strip(), '%Y-%m-%dT%H:%M:%S.%f') for t in self.time.Times])
                    except ValueError:
                        _dates = np.array([datetime.strptime(''.join(t.astype(str)).strip(), '%Y/%m/%d %H:%M:%S.%f') for t in self.time.Times])
                elif 'time' in got_time:
                    _dates = num2date(self.time.time, units=getattr(self.ds.variables['time'], 'units'))
                # We're making Modified Julian Days here to replicate FVCOM's 'time' variable.
                _datenum = date2num(_dates, units='days since 1858-11-17 00:00:00')
                self.time.Itime = np.floor(_datenum)
                self.time.Itime2 = (_datenum - np.floor(_datenum)) * 1000 * 60 * 60  # microseconds since midnight
                attributes = type('attributes', (object,), {})()
                setattr(attributes, 'units', 'days since 1858-11-17 00:00:00')
                setattr(attributes, 'format', 'modified julian day (MJD)')
                setattr(attributes, 'time_zone', 'UTC')
                setattr(self.atts, 'Itime', attributes)
                attributes = type('attributes', (object,), {})()
                setattr(attributes, 'units', 'msec since 00:00:00')
                setattr(attributes, 'time_zone', 'UTC')
                setattr(self.atts, 'Itime2', attributes)

            # Additional nice-to-have time representations.
            if 'Times' in got_time:
                try:
                    self.time.datetime = np.array([datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in self.time.Times])
                except ValueError:
                    self.time.datetime = np.array([datetime.strptime(d, '%Y/%m/%d %H:%M:%S.%f') for d in self.time.Times])
                attributes = type('attributes', (object,), {})()
                setattr(attributes, 'long_name', 'Python datetime.datetime')
                setattr(self.atts, 'datetime', attributes)
            else:
                self.time.datetime = _dates
            self.time.matlabtime = self.time.time + 678942.0  # convert to MATLAB-indexed times from Modified Julian Date.
            attributes = type('attributes', (object,), {})()
            setattr(attributes, 'long_name', 'MATLAB datenum')
            setattr(self.atts, 'matlabtime', attributes)

            # Clip everything to the time indices if we've been given them. Update the time dimension too.
            if 'time' in self._dims:
                if all([isinstance(i, (datetime, str)) for i in self._dims['time']]):
                    # Convert datetime dimensions to indices in the currently loaded data.
                    self._dims['time'][0] = self.time_to_index(self._dims['time'][0])
                    self._dims['time'][1] = self.time_to_index(self._dims['time'][1]) + 1  # make the indexing inclusive
                for time in self.obj_iter(self.time):
                    setattr(self.time, time, getattr(self.time, time)[self._dims['time'][0]:self._dims['time'][1]])
            self.dims.time = len(self.time.time)

    def _load_grid(self):
        """ Load the grid data.

        Convert from UTM to spherical if we haven't got those data in the existing output file.

        """

        grid_metrics = ['nbe', 'ntsn', 'nbsn', 'ntve', 'nbve', 'art1', 'art2', 'a1u', 'a2u']
        grid_variables = ['lon', 'lat', 'x', 'y', 'lonc', 'latc', 'xc', 'yc',
                          'h', 'siglay', 'siglev']

        # Get the grid data.
        for grid in grid_variables:
            try:
                setattr(self.grid, grid, self.ds.variables[grid][:])
                # Save the attributes.
                attributes = type('attributes', (object,), {})()
                for attribute in self.ds.variables[grid].ncattrs():
                    setattr(attributes, attribute, getattr(self.ds.variables[grid], attribute))
                setattr(self.atts, grid, attributes)
            except KeyError:
                # Make zeros for this missing variable so we can convert from the non-missing data below.
                if grid.endswith('c'):
                    setattr(self.grid, grid, np.zeros(self.dims.nele).T)
                else:
                    setattr(self.grid, grid, np.zeros(self.dims.node).T)
            except ValueError as value_error_message:
                warn('Variable {} has a problem with the data. Setting value as all zeros.'.format(grid))
                print(value_error_message)
                setattr(self.grid, grid, np.zeros(self.ds.variables[grid].shape))

        # Load the grid metrics data separately as we don't want to set a bunch of zeros for missing data.
        for metric in grid_metrics:
            if metric in self.ds.variables:
                setattr(self.grid, metric, self.ds.variables[metric][:])
                # Save the attributes.
                attributes = type('attributes', (object,), {})()
                for attribute in self.ds.variables[metric].ncattrs():
                    setattr(attributes, attribute, getattr(self.ds.variables[metric], attribute))
                setattr(self.atts, metric, attributes)

            # Fix the indexing and shapes of the grid metrics variables. Only transpose and offset indexing for nbe.
            try:
                if metric == 'nbe':
                    setattr(self.grid, metric, getattr(self.grid, metric).T - 1)
                else:
                    setattr(self.grid, metric, getattr(self.grid, metric))
            except AttributeError:
                # We don't have this variable, so just pass by silently.
                pass

        try:
            self.grid.nv = self.ds.variables['nv'][:].astype(int)  # force integers even though they should already be so
            self.grid.triangles = copy.copy(self.grid.nv.T - 1)  # zero-indexed for python
        except KeyError:
            # If we don't have a triangulation, make one.
            triangulation = tri.Triangulation(self.grid.lon, self.grid.lat)
            self.grid.triangles = triangulation.triangles
            self.grid.nv = self.grid.triangles.T + 1

        # Fix broken triangulations if necessary.
        if self.grid.nv.min() != 1:
            if self._debug:
                print('Fixing broken triangulation. Current minimum for nv is {} and for triangles is {} but they '
                      'should be 1 and 0, respectively.'.format(self.grid.nv.min(), self.grid.triangles.min()))
            self.grid.nv = (self.ds.variables['nv'][:].astype(int) - self.ds.variables['nv'][:].astype(int).min()) + 1
            self.grid.triangles = copy.copy(self.grid.nv.T) - 1

        # If we've been given an element dimension to subsample in, fix the triangulation here. We should really do
        # this for the nodes too.
        if 'nele' in self._dims:
            if self._debug:
                print('Fix triangulation table as we have been asked for only specific elements.')
                print('Triangulation table minimum/maximum: {}/{}'.format(self.grid.nv[:, self._dims['nele']].min(),
                                                                          self.grid.nv[:, self._dims['nele']].max()))
            # Redo the triangulation here too.
            new_nv = copy.copy(self.grid.nv[:, self._dims['nele']])
            for i, new in enumerate(np.unique(new_nv)):
                new_nv[new_nv == new] = i
            self.grid.nv = new_nv + 1
            self.grid.triangles = new_nv.T

        # Update dimensions to match those we've been given, if any. Omit time here as we shouldn't be touching that
        # dimension for any variable in use in here.
        for dim in self._dims:
            if dim != 'time':
                setattr(self.dims, dim, len(self._dims[dim]))

        # Add compatibility for FVCOM3 (these variables are only specified on the element centres in FVCOM4+ output
        # files). Only create the element centred values if we have the same number of nodes as in the triangulation.
        # This does not occur if we've been asked to extract an incompatible set of nodes and elements, for whatever
        # reason (e.g. testing). We don't add attributes for the data if we've created it as doing so is a pain.
        for var in 'h_center', 'siglay_center', 'siglev_center':
            try:
                setattr(self.grid, var, self.ds.variables[var][:])
                # Save the attributes.
                attributes = type('attributes', (object,), {})()
                for attribute in self.ds.variables[var].ncattrs():
                    setattr(attributes, attribute, getattr(self.ds.variables[var], attribute))
                setattr(self.atts, var, attributes)
            except KeyError:
                if self.grid.nv.max() == len(self.grid.x):
                    try:
                        setattr(self.grid, var, nodes2elems(getattr(self.grid, var.split('_')[0]), self.grid.triangles))
                    except IndexError:
                        # Maybe the array's the wrong way around. Flip it and try again.
                        setattr(self.grid, var, nodes2elems(getattr(self.grid, var.split('_')[0]).T, self.grid.triangles))

        # Convert the given W/E/S/N coordinates into node and element IDs to subset.
        if self._bounding_box:
            self._dims['node'] = np.argwhere((self.grid.lon > self._dims['wesn'][0]) &
                                             (self.grid.lon < self._dims['wesn'][1]) &
                                             (self.grid.lat > self._dims['wesn'][2]) &
                                             (self.grid.lat < self._dims['wesn'][3])).flatten()
            self._dims['nele'] = np.argwhere((self.grid.lonc > self._dims['wesn'][0]) &
                                             (self.grid.lonc < self._dims['wesn'][1]) &
                                             (self.grid.latc > self._dims['wesn'][2]) &
                                             (self.grid.latc < self._dims['wesn'][3])).flatten()

        # If we've been given dimensions to subset in, do that now. Loading the data first and then subsetting
        # shouldn't be a problem from a memory perspective because if you don't have enough memory for the grid data,
        # you probably won't have enough for actually working with the outputs. Also update dimensions to match the
        # given dimensions.
        if 'node' in self._dims:
            self.dims.node = len(self._dims['node'])
            for var in 'x', 'y', 'lon', 'lat', 'h', 'siglay', 'siglev':
                try:
                    node_index = self.ds.variables[var].dimensions.index('node')
                    var_shape = [i for i in np.shape(self.ds.variables[var])]
                    var_shape[node_index] = self.dims.node
                    if 'siglay' in self._dims and 'siglay' in self.ds.variables[var].dimensions:
                        var_shape[self.ds.variables[var].dimensions.index('siglay')] = self.dims.siglay
                    elif 'siglev' in self._dims and 'siglev' in self.ds.variables[var].dimensions:
                        var_shape[self.ds.variables[var].dimensions.index('siglev')] = self.dims.siglev
                    _temp = np.empty(var_shape)
                    if 'siglay' in self.ds.variables[var].dimensions:
                        for ni, node in enumerate(self._dims['node']):
                            if 'siglay' in self._dims:
                                _temp[..., ni] = self.ds.variables[var][self._dims['siglay'], node]
                            else:
                                _temp[..., ni] = self.ds.variables[var][:, node]
                    elif 'siglev' in self.ds.variables[var].dimensions:
                        for ni, node in enumerate(self._dims['node']):
                            if 'siglev' in self._dims:
                                _temp[..., ni] = self.ds.variables[var][self._dims['siglev'], node]
                            else:
                                _temp[..., ni] = self.ds.variables[var][:, node]
                    else:
                        for ni, node in enumerate(self._dims['node']):
                            _temp[..., ni] = self.ds.variables[var][..., node]
                except KeyError:
                    if 'siglay' in var:
                        _temp = np.empty((self.dims.siglay, self.dims.node))
                    elif 'siglev' in var:
                        _temp = np.empty((self.dims.siglev, self.dims.node))
                    else:
                        _temp = np.empty(self.dims.node)
                setattr(self.grid, var, _temp)
        if 'nele' in self._dims:
            self.dims.nele = len(self._dims['nele'])
            for var in 'xc', 'yc', 'lonc', 'latc', 'h_center', 'siglay_center', 'siglev_center':
                try:
                    nele_index = self.ds.variables[var].dimensions.index('nele')
                    var_shape = [i for i in np.shape(self.ds.variables[var])]
                    var_shape[nele_index] = self.dims.nele
                    if 'siglay' in self._dims and 'siglay' in self.ds.variables[var].dimensions:
                        var_shape[self.ds.variables[var].dimensions.index('siglay')] = self.dims.siglay
                    elif 'siglev' in self._dims and 'siglev' in self.ds.variables[var].dimensions:
                        var_shape[self.ds.variables[var].dimensions.index('siglev')] = self.dims.siglev
                    _temp = np.empty(var_shape)
                    if 'siglay' in self.ds.variables[var].dimensions:
                        for ni, nele in enumerate(self._dims['nele']):
                            if 'siglay' in self._dims:
                                _temp[..., ni] = self.ds.variables[var][self._dims['siglay'], nele]
                            else:
                                _temp[..., ni] = self.ds.variables[var][:, nele]
                    elif 'siglev' in self.ds.variables[var].dimensions:
                        for ni, nele in enumerate(self._dims['nele']):
                            if 'siglev' in self._dims:
                                _temp[..., ni] = self.ds.variables[var][self._dims['siglev'], nele]
                            else:
                                _temp[..., ni] = self.ds.variables[var][:, nele]
                    else:
                        for ni, nele in enumerate(self._dims['nele']):
                            _temp[..., ni] = self.ds.variables[var][..., nele]
                except KeyError:
                    # FVCOM3 files don't have h_center, siglay_center and siglev_center, so make var_shape manually.
                    if var.startswith('siglev'):
                        var_shape = [self.dims.siglev, self.dims.nele]
                    elif var.startswith('siglay'):
                        var_shape = [self.dims.siglay, self.dims.nele]
                    else:
                        var_shape = self.dims.nele
                    _temp = np.zeros(var_shape)
                setattr(self.grid, var, _temp)

        # Check if we've been given vertical dimensions to subset in too, and if so, do that. Check we haven't
        # already done this if the 'node' and 'nele' sections above first.
        for var in 'siglay', 'siglev', 'siglay_center', 'siglev_center':
            short_dim = copy.copy(var)
            # Assume we need to subset this one unless 'node' or 'nele' are missing from self._dims. If they're in
            # self._dims, we've already subsetted in the 'node' and 'nele' sections above, so doing it again here
            # would fail.
            subset_variable = True
            if 'node' in self._dims or 'nele' in self._dims:
                subset_variable = False
            # Strip off the _center to match the dimension name.
            if short_dim.endswith('_center'):
                short_dim = short_dim.split('_')[0]
            if short_dim in self._dims:
                if short_dim in self.ds.variables[var].dimensions and subset_variable:
                    _temp = getattr(self.grid, var)[self._dims[short_dim], ...]
                    setattr(self.grid, var, _temp)

        # Check ranges and if zero assume we're missing that particular type, so convert from the other accordingly.
        self.grid.lon_range = np.ptp(self.grid.lon)
        self.grid.lat_range = np.ptp(self.grid.lat)
        self.grid.lonc_range = np.ptp(self.grid.lonc)
        self.grid.latc_range = np.ptp(self.grid.latc)
        self.grid.x_range = np.ptp(self.grid.x)
        self.grid.y_range = np.ptp(self.grid.y)
        self.grid.xc_range = np.ptp(self.grid.xc)
        self.grid.yc_range = np.ptp(self.grid.yc)

        # Only do the conversions when we have more than a single point since the relevant ranges will be zero with
        # only one position.
        if self.dims.node > 1:
            if self.grid.lon_range == 0 and self.grid.lat_range == 0:
                self.grid.lon, self.grid.lat = lonlat_from_utm(self.grid.x, self.grid.y, zone=self._zone)
            if self.grid.lon_range == 0 and self.grid.lat_range == 0:
                self.grid.x, self.grid.y, _ = utm_from_lonlat(self.grid.lon, self.grid.lat)
        if self.dims.nele > 1:
            if self.grid.lonc_range == 0 and self.grid.latc_range == 0:
                self.grid.lonc, self.grid.latc = lonlat_from_utm(self.grid.xc, self.grid.yc, zone=self._zone)
            if self.grid.lonc_range == 0 and self.grid.latc_range == 0:
                self.grid.xc, self.grid.yc, _ = utm_from_lonlat(self.grid.lonc, self.grid.latc)

    def _load_data(self, variables=None):
        """ Wrapper to load the relevant parts of the data in the netCDFs we have been given.

        TODO: This could really do with a decent set of tests to make sure what I'm trying to do is actually what's
        being done.

        """

        # Get a list of all the variables from the netCDF dataset.
        if not variables:
            variables = list(self.ds.variables.keys())

        got_time = 'time' in self._dims
        got_horizontal = 'node' in self._dims or 'nele' in self._dims
        got_vertical = 'siglay' in self._dims or 'siglev' in self._dims

        if self._debug:
            print(self._dims.keys())
            print('time: {} vertical: {} horizontal: {}'.format(got_time, got_vertical, got_horizontal))

        if got_time:
            start, end = self._dims['time']
        else:
            start, end = False, False  # load everything

        nodes, elements, layers, levels = False, False, False, False
        # Make sure we don't have single values for the dimensions otherwise everything gets squeezed and figuring out
        # what dimension is where gets difficult.
        if 'node' in self._dims:
            nodes = self._dims['node']
            if isinstance(nodes, int):
                nodes = [nodes]
        if 'nele' in self._dims:
            elements = self._dims['nele']
            if isinstance(elements, int):
                elements = [elements]
        if 'siglay' in self._dims:
            layers = self._dims['siglay']
            if isinstance(layers, int):
                layers = [layers]
        if 'siglev' in self._dims:
            levels = self._dims['siglev']
            if isinstance(levels, int):
                levels = [levels]
        self.load_data(variables, start=start, end=end, node=nodes, nele=elements, layer=layers, level=levels)

        # Update the dimensions to match the data.
        self._update_dimensions(variables)

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
                print('{}: {} dimension, old/new: {}/{}'.format(self._fvcom, dim, getattr(self.dims, dim), unique_dims[dim]))
            setattr(self.dims, dim, unique_dims[dim])

    def load_data(self, var, start=False, end=False, stride=False, node=False, nele=False, layer=False, level=False):
        """ Add a given variable/variables at the given indices. If any indices are omitted or Falsey, return all
        data for the missing dimensions.

        Parameters
        ----------
        var : list-like, str
            List of variables to load.
        start, end, stride : int, optional
            Start and end of the time range to load. If given, stride sets the increment in times (defaults to 1). If
            omitted, start and end default to all times.
        node : list-like, int, optional
            Horizontal node indices to load (defaults to all positions).
        nele : list-like, int, optional
            Horizontal element indices to load (defaults to all positions).
        layer : list-like, int, optional
            Vertical layer indices to load (defaults to all positions).
        layer : list-like, int, optional
            Vertical level indices to load (defaults to all positions).

        """

        if self._debug:
            print('start: {}, end: {}, stride: {}, node: {}, nele: {}, layer: {}, level: {}'.format(start, end, stride, node, nele, layer, level))

        # Check if we've got iterable variables and make one if not.
        try:
            _ = (e for e in var)
        except TypeError:
            var = [var]

        # For backwards compatibility
        siglay = layer
        siglev = level

        # Save the inputs so we can loop through the variables without checking the last loop's values (which
        # otherwise leads to difficult to fix behaviour).
        original_node = copy.copy(node)
        original_nele = copy.copy(nele)
        original_layer = copy.copy(siglay)
        original_level = copy.copy(siglev)

        # Make the time here as it's independent of the variable in question (unlike the horizontal and vertical
        # dimensions).
        if not stride:
            stride = 1
        if not start:
            start = 0

        for v in var:
            if self._debug:
                print('Loading: {}'.format(v))
            # Get this variable's dimensions and shape
            var_dim = self.ds.variables[v].dimensions
            var_shape = self.ds.variables[v].shape
            var_size_dict = dict(zip(var_dim, var_shape))
            if 'time' not in var_dim:
                # Should we error here or carry on having warned?
                warn('{} does not contain a time dimension.'.format(v))
                possible_indices = {}
            else:
                # make the end of the stride if not supplied
                if not end:
                    end = var_size_dict['time']
                time = np.arange(start,end,stride)
                possible_indices = {'time': time}
            # Save any attributes associated with this variable before trying to load the data.
            attributes = type('attributes', (object,), {})()
            for attribute in self.ds.variables[v].ncattrs():
                setattr(attributes, attribute, getattr(self.ds.variables[v], attribute))
            setattr(self.atts, v, attributes)

            # We've not been told to subset in any dimension, so just return early with all the data.
            if not (start or end or stride or original_layer or original_level or original_node or original_nele):
                if self._debug:
                    print('0: no dims')
                setattr(self.data, v, self.ds.variables[v][:])
            else:
                # Populate indices for omitted values. Warn we don't have a sigma layer dimension, but carry on. We
                # don't need to make dummy data because if this file doesn't have this dimension, it certainly
                # won't have any data which include it.
                if not isinstance(original_layer, (list, tuple, np.ndarray)):
                    if not original_layer:
                        try:
                            siglay = np.arange(self.dims.siglay)
                        except AttributeError:
                            warn('{} does not contain a sigma layer dimension.'.format(v))
                            pass
                possible_indices['siglay'] = siglay
                if not isinstance(original_level, (list, tuple, np.ndarray)):
                    if not original_level:
                        try:
                            siglev = np.arange(self.dims.siglev)
                        except AttributeError:
                            warn('{} does not contain a sigma level dimension.'.format(v))
                            pass
                possible_indices['siglev'] = siglev
                if not isinstance(original_node, (list, tuple, np.ndarray)):
                    if not original_node:
                        # I'm not sure if this is a really terrible idea (from a performance perspective).
                        node = np.arange(self.dims.node)
                possible_indices['node'] = node
                if not isinstance(original_nele, (list, tuple, np.ndarray)):
                    if not original_nele:
                        # I'm not sure if this is a really terrible idea (from a performance perspective).
                        nele = np.arange(self.dims.nele)
                possible_indices['nele'] = nele

                var_index_dict = {}
                for this_key, this_size in var_size_dict.items():
                    # Try and add the indices for the various dimensions present in var_dim. If there is a
                    # dimension which isn't present (e.g. bedlay for sediment) then it creates a range covering the
                    # whole slice.
                    try:
                        var_index_dict[this_key] = possible_indices[this_key]
                    except KeyError:
                        var_index_dict[this_key] = np.arange(var_size_dict[this_key])

                # Need to reorder to get back to the order the dimensions are in the netcdf (since the dictionary is
                # unordered).
                ordered_coords = [list(var_index_dict[this_key]) for this_key in var_dim]
                try:
                    setattr(self.data, v, self.ds.variables[v][ordered_coords])
                except MemoryError:
                    raise MemoryError("Variable {} too large for RAM. Use `dims' to load subsets in space or time or "
                                      "`variables' to request only certain variables.".format(v))

    def closest_time(self, when):
        """ Find the index of the closest time to the supplied time (datetime object). """
        try:
            return np.argmin(np.abs(self.time.datetime - when))
        except AttributeError:
            self.load_time()
            return np.argmin(np.abs(self.time.datetime - when))

    def closest_node(self, where, cartesian=False, threshold=None, vincenty=False, haversine=False):
        """
        Find the index of the closest node to the supplied position (x, y). Set `cartesian' to True for cartesian
        coordinates (defaults to spherical).

        Parameters
        ----------
        where : list-like
            Arbitrary x, y position for which to find the closest model grid position.
        cartesian : bool, optional
            Set to True to use cartesian coordinates. Defaults to False.
        threshold : float, optional
            Give a threshold distance beyond which the closest grid is considered too far away. Units are the same as
            the coordinates in `where', unless using lat/lon and vincenty when it is in metres. Return None when
            beyond threshold.
        vincenty : bool, optional
            Use vincenty distance calculation. Allows specification of point in lat/lon but threshold in metres.
        haversine : bool, optional
            Use the simpler but much faster Haversine distance calculation. Allows specification of point in lat/lon but threshold in metres.

        Returns
        -------
        index : int, None
            Grid index which falls closest to the supplied position. If `threshold' is set and the distance from the
            supplied position to the nearest model node exceeds that threshold, `index' is None.

        """

        if not vincenty or not haversine:
            if cartesian:
                x, y = self.grid.x, self.grid.y
            else:
                x, y = self.grid.lon, self.grid.lat
            dist = np.sqrt((x - where[0])**2 + (y - where[1])**2)
        elif vincenty:
            grid_pts = np.asarray([self.grid.lon, self.grid.lat]).T
            where_pt_rep = np.tile(np.asarray(where), (len(self.grid.lon),1))
            dist = np.asarray([vincenty_distance(pt_1, pt_2) for pt_1, pt_2 in zip(grid_pts, where_pt_rep)])*1000
        elif haversine:
            grid_pts = np.asarray([self.grid.lon, self.grid.lat]).T
            where_pt_rep = np.tile(np.asarray(where), (len(self.grid.lon),1))
            dist = np.asarray([haversine_distance(pt_1, pt_2) for pt_1, pt_2 in zip(grid_pts, where_pt_rep)])*1000
        index = np.argmin(dist)
        if threshold:
            if dist.min() < threshold:
                index = np.argmin(dist)
            else:
                index = None

        return index

    def closest_element(self, where, cartesian=False, threshold=None, vincenty=False):
        """
        Find the index of the closest element to the supplied position (x, y). Set `cartesian' to True for cartesian
        coordinates (defaults to spherical).

        Parameters
        ----------
        where : list-like
            Arbitrary x, y position for which to find the closest model grid position.
        cartesian : bool, optional
            Set to True to use cartesian coordinates. Defaults to False.
        threshold : float, optional
            Give a threshold distance beyond which the closest grid is considered too far away. Units are the same as
            the coordinates in `where', unless using lat/lon and vincenty when it is in metres. Return None when
            beyond threshold.
        vincenty : bool, optional
            Use vincenty distance calculation. Allows specification of point in lat/lon but threshold in metres

        Returns
        -------
        index : int, None
            Grid index which falls closest to the supplied position. If `threshold' is set and the distance from the
            supplied position to the nearest model node exceeds that threshold, `index' is None.

        """
        if not vincenty:
            if cartesian:
                x, y = self.grid.xc, self.grid.yc
            else:
                x, y = self.grid.lonc, self.grid.latc
            dist = np.sqrt((x - where[0])**2 + (y - where[1])**2)
        else:
            grid_pts = np.asarray([self.grid.lonc, self.grid.latc]).T
            where_pt_rep = np.tile(np.asarray(where), (len(self.grid.lonc),1))
            dist = np.asarray([vincenty_distance(pt_1, pt_2) for pt_1, pt_2 in zip(grid_pts, where_pt_rep)])*1000

        index = np.argmin(dist)
        if threshold:
            if dist.min() < threshold:
                index = np.argmin(dist)
            else:
                index = None

        return index

    def grid_volume(self):
        """
        Calculate the grid volume (optionally time varying) for the loaded grid.

        If the surface elevation data have been loaded (`zeta'), the volume varies with time, otherwise, the volume
        is for the mean water depth (`h').

        Returns
        -------
        self.depth_volume : ndarray
            Depth-resolved volume.
        self.volume : ndarray
            Depth-integrated volume.

        """

        if not hasattr(self.data, 'zeta'):
            surface_elevation = np.zeros((self.dims.node, self.dims.time))
        else:
            surface_elevation = self.data.zeta

        self.depth_volume, self.volume = unstructured_grid_volume(self.grid.art1, self.grid.h, surface_elevation, self.grid.siglev, depth_integrated=True)

    def time_to_index(self, target_time, tolerance=False):
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

        if not isinstance(target_time, datetime):
            try:
                target_time = datetime.strptime(target_time, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                # Try again in case we've not been given fractional seconds, just to be nice.
                target_time = datetime.strptime(target_time, '%Y-%m-%d %H:%M:%S')

        time_diff = np.abs(self.time.datetime - target_time)
        if not tolerance:
            time_idx = np.argmin(time_diff)
        else:
            if np.min(time_diff) <= timedelta(seconds=tolerance):
                time_idx = np.argmin(time_diff)
            else:
                time_idx = None

        return time_idx


def MFileReader(fvcom, *args, **kwargs):
    """ Wrapper around FileReader for loading multiple files at once.

    Parameters
    ----------
    fvcom : list-like, str
        List of files to load.

    Additional arguments are passed to `PyFVCOM.read.FileReader'.

    Returns
    -------
    FVCOM : PyFVCOM.read.FileReader
        Concatenated data from the files in `fvcom'.

    """

    if isinstance(fvcom, str):
        FVCOM = FileReader(fvcom, *args, **kwargs)
    else:
        for file in fvcom:
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

        self.obj_iter = lambda x: [a for a in dir(x) if not a.startswith('__')]

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
        for obj in self.obj_iter(self.data):
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
    Quiet : bool
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

    def __init__(self, input_dict, filename_out, Quiet=False, format='NETCDF3_CLASSIC'):
        self.filename_out = filename_out
        self.input_dict = input_dict
        self.Quiet = Quiet
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
            if not self.Quiet:
                print('No netCDF created:')
                print('  No dimension key found (!! has to be \"dimensions\"!!!)')
            return()

        # Create global attributes.
        if 'global attributes' in self.input_dict:
            for k, v in self.input_dict['global attributes'].items():
                rootgrp.setncattr(k, v)
        else:
            if not self.Quiet:
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
            if not self.Quiet:
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
            fmt = '{}\n'.format(fmt.strip())

        # Dump the data (this may be slow).
        for line in np.column_stack((mjd, timeseries)):
            f.write(fmt.format(*line))


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
