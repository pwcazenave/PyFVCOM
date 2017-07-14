from __future__ import print_function

import sys
import copy
import inspect

import numpy as np

from warnings import warn
from datetime import datetime
from netCDF4 import Dataset, MFDataset, num2date, date2num

from PyFVCOM.ll2utm import lonlat_from_utm, utm_from_lonlat


class FileReader:
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
        variables : list-like
            List of variables to extract. If omitted, no variables are extracted, which means you won't be able to
            add this object to another one which does have variables in it.
        dims : dict
            Dictionary of dimension names along which to subsample e.g. dims={'time': [0, 100], 'nele': [0, 10, 100],
            'node': 100}. Times are specified as ranges; horizontal and vertical dimensions (siglay, siglev, node,
            nele) can be list-like. Any combination of dimensions is possible. Omitted dimensions are loaded in their
            entirety.
        zone : str, list-like
            UTM zones (defaults to '30N') for conversion of UTM to spherical coordinates.
        debug : bool
            Set to True to enable debug output. Defaults to False.

        Author(s)
        ---------
        Pierre Cazenave (Plymouth Marine Laboratory)

        Todo
        ----
        - Add support for specifying a subset in space with a bounding box (w/e/s/n)
        - Add support for specifying a time period with datetime objects (start:end)

        """
        self._debug = debug
        self._fvcom = fvcom
        self._zone = zone
        self._dims = dims
        # Silently convert a string variable input to an iterable list.
        if isinstance(variables, str):
            variables = [variables]
        self._variables = variables

        # Prepare this object with all the objects we'll need later on (data, dims, time, grid).
        self._prep()

        # Get the things to iterate over for a given object. This is a bit hacky, but until or if I create separate
        # classes for the dims, time, grid and data objects, this'll have to do.
        self.obj_iter = lambda x: [a for a in dir(x) if not a.startswith('__')]

        self.ds = Dataset(self._fvcom, 'r')

        for dim in self.ds.dimensions:
            setattr(self.dims, dim, self.ds.dimensions[dim].size)

        self._load_time()
        self._load_grid()

        if variables:
            try:
                self._load_timeseries(self._variables)
            except MemoryError:
                raise MemoryError("Data too large for RAM. Use `dims' to load subsets in space or time or "
                                  "`variables' to request only certain variables.")

    def __eq__(self, other):
        # For easy comparison of classes.
        return self.__dict__ == other.__dict__

    def __add__(self, FVCOM, debug=False):
        """ This special method means we can stack two FVCOM objects in time through a simple addition (e.g. fvcom1 +=
        fvcom2)

        Parameters
        ----------
        FVCOM : PyFVCOM.FVCOM
            Previous time to which to add ourselves.

        Returns
        -------
        NEW : PyFVCOM.FVCOM
            Concatenated (in time) `PyFVCOM.FVCOM' class.

        Notes
        -----
        - fvcom1 and fvcom2 have to cover the exact same spatial domain
        - last time step of fvcom1 must be <= to the first time step of fvcom2
        - if data have not been loaded, then subsequent loads will load only the data from the first netCDF. Make
        sure you load all your data before you merge objects.

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
            raise ValueError('Horizontal nodal data are incompatible.')
        if not nele_compare:
            raise ValueError('Horizontal element data are incompatible.')
        if not siglay_compare:
            raise ValueError('Vertical sigma layers are incompatible.')
        if not siglev_compare:
            raise ValueError('Vertical sigma levels are incompatible.')
        if not time_compare:
            raise ValueError("Time periods are incompatible (`fvcom2' must be greater than or equal to `fvcom')."
                             "`fvcom1' has end {} and `fvcom2' has start {}".format(self.time.datetime[-1], FVCOM.time.datetime[0]))
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
                # This is hopefully one of the additional time variable which doesn't exist in the netCDF dataset.
                # Just delete the relevant indices by assuming that time is the first axis.
                setattr(idem.time, time, np.delete(getattr(idem.time, time), dupe_indices, axis=0))

        # Update dimensions accordingly.
        idem.dims.time = len(idem.time.time)

        return idem

    def _prep(self):
        # Create empty object for the grid, dimension, data and time data.
        self.grid = type('grid', (object,), {})()
        self.dims = type('dims', (object,), {})()
        self.data = type('data', (object,), {})()
        self.time = type('time', (object,), {})()

        # Add docstrings for the relevant objects.
        self.data.__doc__ = "This object will contain data as loaded from the netCDFs specified. Use " \
                            "`FVCOM.load_timeseries' to get specific data (optionally at specific locations, times and" \
                            " depths)."
        self.dims.__doc__ = "This contains the dimensions of the data from the given netCDFs."
        self.grid.__doc__ = "Use `FVCOM.load_grid' to populate this with the FVCOM grid information. Missing " \
                            "spherical or cartesian coordinates are automatically created depending on which is " \
                            "missing."
        self.time.__doc__ = "This contains the time data for the given netCDFs. Missing standard FVCOM time variables " \
                            "are automatically created."

    def _load_time(self):
        """ Populate a time object with additional useful time representations from the netCDF time data.
        """

        time_variables = ('time', 'Times', 'Itime', 'Itime2')
        got_time, missing_time = [], []
        for time in time_variables:
            # Since not all of the time_variables specified above are required, only try to load the data if they
            # exist. We'll raise an error if we don't find any of them though.
            if time in self.ds.variables:
                setattr(self.time, time, self.ds.variables[time][:])
                got_time.append(time)
            else:
                missing_time.append(time)

        if len(missing_time) == len(time_variables):
            raise ValueError('No time variables found in the netCDF.')

        if 'Times' in got_time:
            # Overwrite the existing Times array with a more sensibly shaped one.
            self.time.Times = np.asarray([''.join(t.astype(str)).strip() for t in self.time.Times])

        # Make whatever we got into datetime objects and use those to make everything else. Note: the `time' variable
        # is often the one with the lowest precision, so use the others preferentially over that.
        if 'Times' not in got_time:
            if 'time' in got_time:
                _dates = num2date(self.time, units=getattr(self.ds.variables['time'], 'units'))
            elif 'Itime' in got_time and 'Itime2' in got_time:
                _dates = num2date(self.Itime + self.Itime2 / 1000.0 / 60 / 60, units=getattr(self.ds.variables['Itime'], 'units'))
            try:
                self.time.Times = [datetime.strftime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in _dates]
            except ValueError:
                self.time.Times = [datetime.strftime(d, '%Y/%m/%d %H:%M:%S.%f') for d in _dates]

        if 'time' not in got_time:
            if 'Times' in got_time:
                try:
                    _dates = [datetime.strptime(''.join(t.astype(str)).strip(), '%Y-%m-%dT%H:%M:%S.%f') for t in self.time.Times]
                except ValueError:
                    _dates = [datetime.strptime(''.join(t.astype(str)).strip(), '%Y/%m/%d %H:%M:%S.%f') for t in self.time.Times]
            elif 'Itime' in got_time and 'Itime2' in got_time:
                _dates = num2date(self.Itime + self.Itime2 / 1000.0 / 60 / 60, units=getattr(self.ds.variables['Itime'], 'units'))
            # We're making Modified Julian Days here to replicate FVCOM's 'time' variable.
            self.time.time = date2num(_dates, units='days since 1858-11-17 00:00:00')

        if 'Itime' not in got_time and 'Itime2' not in got_time:
            if 'Times' in got_time:
                try:
                    _dates = [datetime.strptime(''.join(t.astype(str)).strip(), '%Y-%m-%dT%H:%M:%S.%f') for t in self.time.Times]
                except ValueError:
                    _dates = [datetime.strptime(''.join(t.astype(str)).strip(), '%Y/%m/%d %H:%M:%S.%f') for t in self.time.Times]
            elif 'time' in got_time:
                _dates = num2date(self.time, units=getattr(self.ds.variables['time'], 'units'))
            # We're making Modified Julian Days here to replicate FVCOM's 'time' variable.
            _datenum = date2num(_dates, units='days since 1858-11-17 00:00:00')
            self.time.Itime = np.floor(_datenum)
            self.time.Itime = (_datenum - np.floor(_datenum)) * 1000 * 60 * 60  # microseconds since midnight

        # Additional nice-to-have time representations.
        if 'Times' in got_time:
            try:
                self.time.datetime = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in self.time.Times]
            except ValueError:
                self.time.datetime = [datetime.strptime(d, '%Y/%m/%d %H:%M:%S.%f') for d in self.time.Times]
        else:
            self.time.datetime = _dates
        self.time.matlabtime = self.time.time + 678942.0  # convert to MATLAB-indexed times from Modified Julian Date.

        # Clip everything to the time indices if we've been given them.
        if 'time' in self._dims:
            for time in self.obj_iter(self.time):
                setattr(self.time, time, getattr(self.time, time)[self._dims['time'][0]:self._dims['time'][1]])

    def _load_grid(self):
        """ Load the grid data.

        Convert from UTM to spherical if we haven't got those data in the existing output file.

        """

        _range = lambda x: x.max() - x.min()

        self.grid.nv = self.ds.variables['nv'][:]
        self.grid.triangles = self.grid.nv.T - 1  # zero-indexed for python

        # Get the grid data.
        # TODO: If we have a subset in nodes/elems, we need to adjust the coordinate ranges here.
        # if self._subset:
        #     for grid in lon, lat, x, y:
        #         setattr(self.grid, grid, getattr(self.grid, grid)[idx]
        for grid in 'lon', 'lat', 'x', 'y', 'lonc', 'latc', 'xc', 'yc', 'h', 'siglay', 'siglev':
            setattr(self.grid, grid, self.ds.variables[grid][:])

        # Add compatibility for FVCOM3 (these variables are only specified on the element centres in FVCOM4+ output
        # files).
        if 'h_center' in self.ds.variables:
            self.grid.h_center = self.ds.variables['h_center'][:]
        else:
            self.grid.h_center = nodes2elems(self.grid.h, self.grid.triangles)

        if 'siglay_center' in self.ds.variables:
            self.grid.siglay_center = self.ds.variables['siglay_center'][:]
        else:
            self.grid.siglay_center = nodes2elems(self.grid.siglay, self.grid.triangles)

        if 'siglev_center' in self.ds.variables:
            self.grid.siglev_center = self.ds.variables['siglev_center'][:]
        else:
            self.grid.siglev_center = nodes2elems(self.grid.siglev, self.grid.triangles)

        # If we've been given dimensions to subset in, do that now. Loading the data first and then subsetting
        # shouldn't be a problem from a memory perspective because if you don't have enough memory for the grid data,
        # you probably won't have enough for actually working with the outputs. Also update dimensions to match the
        # given dimensions.
        if 'node' in self._dims:
            self.dims.node = len(self._dims['node'])
            for var in 'x', 'y', 'lon', 'lat', 'h', 'siglay', 'siglev':
                node_index = self.ds.variables[var].dimensions.index('node')
                var_shape = [i for i in np.shape(self.ds.variables[var])]
                var_shape[node_index] = self.dims.node
                _temp = np.empty(var_shape)
                for ni, node in enumerate(self._dims['node']):
                    _temp[..., ni] = self.ds.variables[var][..., node]
                setattr(self.grid, var, _temp)
        if 'nele' in self._dims:
            self.dims.nele = len(self._dims['nele'])
            for var in 'xc', 'yc', 'lonc', 'latc', 'h_center', 'siglay_center', 'siglev_center':
                try:
                    nele_index = self.ds.variables[var].dimensions.index('nele')
                    var_shape = [i for i in np.shape(self.ds.variables[var])]
                    var_shape[nele_index] = self.dims.nele
                except KeyError:
                    # FVCOM3 files don't have h_center, siglay_center and siglev_center, so make var_shape manually.
                    if var.startswith('siglev'):
                        var_shape = [len(self.dims.siglev), len(self._dims['nele'])]
                    else:
                        var_shape = [len(self.dims.siglay), len(self._dims['nele'])]
                _temp = np.empty(var_shape)
                for ni, nele in enumerate(self._dims['nele']):
                    _temp[..., ni] = self.ds.variables[var][..., nele]
                setattr(self.grid, var, _temp)

            # Redo the triangulation here too.
            new_nv = copy.copy(self.grid.nv[:, self._dims['nele']])
            for i, new in enumerate(np.unique(new_nv)):
                new_nv[new_nv == new] = i
            self.grid.nv = new_nv
            self.grid.triangles = new_nv.T - 1

        # Check ranges and if zero assume we're missing that particular type, so convert from the other accordingly.
        self.grid.lon_range = _range(self.grid.lon)
        self.grid.lat_range = _range(self.grid.lat)
        self.grid.lonc_range = _range(self.grid.lonc)
        self.grid.latc_range = _range(self.grid.latc)
        self.grid.x_range = _range(self.grid.x)
        self.grid.y_range = _range(self.grid.y)
        self.grid.xc_range = _range(self.grid.xc)
        self.grid.yc_range = _range(self.grid.yc)

        if self.grid.lon_range == 0 and self.grid.lat_range == 0:
            self.grid.lon, self.grid.lat = lonlat_from_utm(self.grid.x, self.grid.y, zone=self._zone)
        if self.grid.lonc_range == 0 and self.grid.latc_range == 0:
            self.grid.lonc, self.grid.latc = lonlat_from_utm(self.grid.xc, self.grid.yc, zone=self._zone)
        if self.grid.lon_range == 0 and self.grid.lat_range == 0:
            self.grid.x, self.grid.y = utm_from_lonlat(self.grid.lon, self.grid.lat)
        if self.grid.lonc_range == 0 and self.grid.latc_range == 0:
            self.grid.xc, self.grid.yc = utm_from_lonlat(self.grid.lonc, self.grid.latc)

    def _load_timeseries(self, variables=None):
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
        self.load_timeseries(variables, start=start, end=end, node=nodes, nele=elements, layer=layers, level=levels)

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

    def load_timeseries(self, var, start=False, end=False, stride=False, node=False, nele=False, layer=False, level=False):
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

        # Save the inputs so we can loop through the variables without checking the last loop's values (which
        # otherwise leads to difficult to fix behaviour).
        original_node = copy.copy(node)
        original_nele = copy.copy(nele)
        original_layer = copy.copy(layer)
        original_level = copy.copy(level)

        # Make the time here as it's independent of the variable in question (unlike the horizontal and vertical
        # dimensions).
        if not stride:
            stride = 1
        if start or end:
            time = np.arange(start, end, stride)
        else:
            time = np.arange(self.dims.time)

        for v in var:
            if self._debug:
                print('Loading: {}'.format(v))
            # Get this variable's dimensions
            var_dim = self.ds.variables[v].dimensions
            if 'time' not in var_dim:
                # Should we error here or carry on having warned?
                raise ValueError('{} does not contain a time dimension.'.format(v))

            # We've not been told to subset in any dimension, so just return early with all the data.
            if not (start or end or stride or original_layer or original_level or original_node or original_nele):
                if self._debug:
                    print('0: no dims')
                setattr(self.data, v, self.ds.variables[v][:])
            else:
                # Populate indices for omitted values.
                if not isinstance(original_layer, (list, tuple, np.ndarray)):
                    if not original_layer:
                        layer = np.arange(self.dims.siglay)
                if not isinstance(original_level, (list, tuple, np.ndarray)):
                    if not original_level:
                        level = np.arange(self.dims.siglev)
                if not isinstance(original_node, (list, tuple, np.ndarray)):
                    if not original_node:
                        # I'm not sure if this is a really terrible idea (from a performance perspective).
                        node = np.arange(self.dims.node)
                if not isinstance(original_nele, (list, tuple, np.ndarray)):
                    if not original_nele:
                        # I'm not sure if this is a really terrible idea (from a performance perspective).
                        nele = np.arange(self.dims.nele)


                # Check what dimensions this variables has an load the indices accordingly. This is probably an ideal
                # candidate for optimisation of the indexing (see numpy's fancy indexing or ravel()ing the whole
                # array and using ranges instead: https://stackoverflow.com/questions/14386822). This also assumes we
                # will only ever have a 3D array (4D really, x, y, z, t, but unstructured, so (x,y), z,
                # t). I can imagine a scenario where we have 4D but that isn't yet a requirement.
                temporal = 'time' in var_dim
                vertical = 'siglay' in var_dim or 'siglev' in var_dim
                horizontal = 'node' in var_dim or 'nele' in var_dim
                if temporal and vertical and horizontal:
                    if self._debug:
                        print('1: dims {}'.format(self._dims))
                    if 'siglay' in var_dim and 'node' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][time, layer, node])
                    elif 'siglev' in var_dim and 'node' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][time, level, node])
                    elif 'siglay' in var_dim and 'nele' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][time, layer, nele])
                    elif 'siglev' in var_dim and 'nele' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][time, level, nele])
                elif temporal and vertical and not horizontal:
                    if self._debug:
                        print('2: dims {}'.format(self._dims))
                    if 'siglay' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][time, layer, ...])
                    elif 'siglev' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][time, level, ...])
                elif temporal and not vertical and horizontal:
                    if self._debug:
                        print('3: dims {}'.format(self._dims))
                    if 'node' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][time, node])
                    elif 'nele' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][time, nele])
                elif not temporal and vertical and horizontal:
                    if self._debug:
                        print('4: dims {}'.format(self._dims))
                    if 'siglay' in var_dim and 'node' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][..., layer, node])
                    elif 'siglev' in var_dim and 'node' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][..., level, node])
                    elif 'siglay' in var_dim and 'nele' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][..., layer, nele])
                    elif 'siglev' in var_dim and 'nele' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][..., level, nele])
                elif not temporal and vertical and not horizontal:
                    if self._debug:
                        print('5: dims {}'.format(self._dims))
                    if 'siglay' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][..., layer, ...])
                    elif 'siglev' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][..., level, ...])
                elif not temporal and not vertical and horizontal:
                    if self._debug:
                        print('6: dims {}'.format(self._dims))
                    if 'node' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][..., node])
                    elif 'nele' in var_dim:
                        setattr(self.data, v, self.ds.variables[v][..., nele])
                else:
                    if self._debug:
                        print('7: dims {}'.format(self._dims))

    def closest_time(self, when):
        """ Find the index of the closest time to the supplied time (datetime object). """
        try:
            return np.argwhere(np.abs(self.time.datetime - when))
        except AttributeError:
            self.load_time()
            return np.argwhere(np.abs(self.time.datetime - when))

    def closest_node(self, where, cartesian=False):
        """ Find the index of the closest node to the supplied position (x, y). Set `cartesian' to True for cartesian 
        coordinates (defaults to spherical). """

        if cartesian:
            return np.argwhere(np.sqrt((self.grid.x - where[0])**2 + (self.grid.x - where[1])**2))
        else:
            return np.argwhere(np.sqrt((self.grid.y - where[0])**2 + (self.grid.y - where[1])**2))

    def closest_element(self, where, cartesian=False):
        """ Find the index of the closest element to the supplied position (x, y). Set `cartesian' to True for cartesian 
        coordinates (defaults to spherical). """

        if cartesian:
            return np.argwhere(np.sqrt((self.grid.x - where[0])**2 + (self.grid.x - where[1])**2))
        else:
            return np.argwhere(np.sqrt((self.grid.y - where[0])**2 + (self.grid.y - where[1])**2))


def MFileReader(fvcom, *args, **kwargs):
    """ Wrapper around FileReader for loading multiple files at once.

    Parameters
    ----------
    fvcom : list-like, str
        List of files to load.

    Additional arguments are passed to `PyFVCOM.read_results.FileReader'.

    Returns
    -------
    FVCOM : PyFVCOM.read_results.FileReader
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
    # Check whether we'll be able to fulfill the datetime request.
    if datetimes and vars and not list(set(vars) & set(('Times', 'time'))):
        raise ValueError("Conversion to python datetimes has been requested "
                         "but no time variable (`Times' or `time') has been "
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

            if datetimes and key in ('Times', 'time') and not done_datetimes:
                # Convert the time data to datetime objects. How we do this
                # depends on which we hit first - `Times' or `time'. For the
                # former, we need to parse the strings, for the latter we can
                # leverage num2date from the netCDF4 module and use the time
                # units attribute.
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

            if noisy:
                if len(str(to_extract)) < 60:
                    print('(extracted {})'.format(str(to_extract).replace('\'', '')))
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
