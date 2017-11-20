"""
Tools to prepare data for an FVCOM run.

A very gradual port of the most used functions from the MATLAB toolbox:
    https://github.com/pwcazenave/fvcom-toolbox/tree/master/fvcom_prepro/

Author(s):

Mike Bedington (Plymouth Marine Laboratory)
Pierre Cazenave (Plymouth Marine Laboratory)

"""

import os
import numpy as np
import multiprocessing as mp

from netCDF4 import Dataset, date2num, num2date
from matplotlib.dates import date2num as mtime
from scipy.interpolate import RegularGridInterpolator
from dateutil.relativedelta import relativedelta
from datetime import datetime
from functools import partial
from warnings import warn
from utide import reconstruct, ut_constants
from utide.utilities import Bunch

from PyFVCOM.coordinate import *
from PyFVCOM.grid import Domain, grid_metrics, read_fvcom_obc, nodes2elems
from PyFVCOM.utilities import date_range


class Model(Domain):
    """ Hold all the model inputs. """
    def __init__(self, start, end, *args, **kwargs):

        # Inherit everything from PyFVCOM.grid.Domain, but extend it for our purposes.
        super().__init__(*args, **kwargs)

        self.start = start
        self.end = end

    def interp_sst_assimilation(self, sst_dir, year, serial=False, pool_size=None, noisy=False):
        """
        Interpolate SST data from remote sensing data onto the supplied model
        grid.

        Parameters
        ----------
        sst_dir : str
            Path to directory containing the SST data. Assumes there are directories per year within this directory.
        year : int
            Tear for which to generate SST data
        serial : bool, optional
            Run in serial rather than parallel. Defaults to parallel.
        pool_size : int, optional
            Specify number of processes for parallel run. By default it uses all available.
        noisy : bool, optional
            Set to True to enable some sort of progress output. Defaults to False.

        Returns
        -------
        Adds a new `sst' object with:
        sst : np.ndarray
            Interpolated SST time series for the supplied domain.
        time : np.ndarray
            List of python datetimes for the corresponding SST data.

        Example
        -------
        >>> from PyFVCOM.preproc import Model
        >>> sst_dir = '/home/mbe/Data/SST_data/2006/'
        >>> model = Model('/home/mbe/Models/FVCOM/tamar/tamar_v2_grd.dat',
        >>>     native_coordinates='cartesian', zone='30N')
        >>> model.interp_sst_assimilation(sst_dir, 2006, pool_size=20)
        >>> # Save to netCDF
        >>> model.write_sstgrd('casename_sstgrd.nc')

        Notes
        -----
        - Based on https://github.com/pwcazenave/fvcom-toolbox/tree/master/fvcom_prepro/interp_sst_assimilation.m.

        """

        # SST files. Try to prepend the end of the previous year and the start of the next year.
        sst_files = [os.path.join(sst_dir, str(year - 1), sorted(os.listdir(os.path.join(sst_dir, str(year - 1))))[-1])]
        sst_files += [os.path.join(sst_dir, str(year), i) for i in os.listdir(os.path.join(sst_dir, str(year)))]
        sst_files += [os.path.join(sst_dir, str(year + 1), sorted(os.listdir(os.path.join(sst_dir, str(year + 1))))[0])]

        if noisy:
            print('To do:\n{}'.format('|' * len(sst_files)), flush=True)

        # Read SST data files and interpolate each to the FVCOM mesh
        lonlat = np.array((self.grid.lon, self.grid.lat))

        if serial:
            results = []
            for sst_file in sst_files:
                results.append(self._inter_sst_worker(lonlat, sst_file, noisy))
        else:
            if not pool_size:
                pool = mp.Pool()
            else:
                pool = mp.Pool(pool_size)
            part_func = partial(self._inter_sst_worker, lonlat, noisy=noisy)
            results = pool.map(part_func, sst_files)
            pool.close()

        # Sort data and prepare date lists
        dates = np.empty(len(results)).astype(datetime)
        sst = np.empty((len(results), self.dims.node))
        for i, result in enumerate(results):
            dates[i] = result[0][0] + relativedelta(hours=12)  # FVCOM wants times at midday whilst the data are at midnight
            sst[i, :] = result[1]

        # Sort by time.
        idx = np.argsort(dates)
        dates = dates[idx]
        sst = sst[idx, :]

        # Store everything in an object.
        self.sst = type('sst', (object,), {})()
        self.sst.sst = sst
        self.sst.time = dates

    @staticmethod
    def _inter_sst_worker(fvcom_lonlat, sst_file, noisy=False):
        """ Multiprocessing worker function for the SST interpolation. """
        if noisy:
            print('.', end='', flush=True)

        with Dataset(sst_file, 'r') as sst_file_nc:
            sst_eo = np.squeeze(sst_file_nc.variables['analysed_sst'][:]) - 273.15  # Kelvin to Celsius
            mask = sst_file_nc.variables['mask']
            sst_eo[mask == 1] = np.nan
            sst_lon = sst_file_nc.variables['lon'][:]
            sst_lat = sst_file_nc.variables['lat'][:]
            time_out_dt = num2date(sst_file_nc.variables['time'][:], units=sst_file_nc.variables['time'].units)

        ft = RegularGridInterpolator((sst_lon, sst_lat), sst_eo.T, method='nearest', fill_value=None)
        interp_sst = ft(fvcom_lonlat.T)

        return time_out_dt, interp_sst

    def write_sstgrd(self, output_file, ncopts={'zlib': True, 'complevel': 7}, **kwargs):
        """
        Generate a sea surface temperature data assimilation file for the given FVCOM domain from the self.sst data.

        Parameters
        ----------
        output_file : str, pathlib.Path
            File to which to write SST data.
        ncopts : dict
            Dictionary of options to use when creating the netCDF variables. Defaults to compression on.

        Remaining arguments are passed to WriteForcing.
        """

        globals = {'year': str(np.argmax(np.bincount([i.year for i in self.sst.time]))),  # gets the most common year value
                   'title': 'FVCOM SST 1km merged product File',
                   'institution': 'Plymouth Marine Laboratory',
                   'source': 'FVCOM grid (unstructured) surface forcing',
                   'history': 'File created using PyFVCOM',
                   'references': 'http://fvcom.smast.umassd.edu, http://codfish.smast.umassd.edu, http://pml.ac.uk/modelling',
                   'Conventions': 'CF-1.0',
                   'CoordinateProjection': 'init=WGS84'}
        dims = {'nele': self.dims.nele, 'node': self.dims.node, 'time': 0, 'DateStrLen': 26, 'three': 3}

        with WriteForcing(str(output_file), dims, global_attributes=globals, clobber=True, format='NETCDF4', **kwargs) as sstgrd:
            # Add the variables.
            atts = {'long_name': 'nodel longitude', 'units': 'degrees_east'}
            sstgrd.add_variable('lon', self.grid.lon, ['node'], attributes=atts, ncopts=ncopts)
            atts = {'long_name': 'nodel latitude', 'units': 'degrees_north'}
            sstgrd.add_variable('lat', self.grid.lat, ['node'], attributes=atts, ncopts=ncopts)
            atts = {'units': 'days since 1858-11-17 00:00:00',
                    'delta_t': '0000-00-00 01:00:00',
                    'format': 'modified julian day (MJD)',
                    'time_zone': 'UTC'}
            sstgrd.add_variable('time', date2num(self.sst.time, units='days since 1858-11-17 00:00:00'),
                                ['time'], attributes=atts, ncopts=ncopts)
            atts = {'long_name': 'Calendar Date',
                    'format': 'String: Calendar Time',
                    'time_zone': 'UTC'}
            sstgrd.add_variable('Times', [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in self.sst.time],
                                ['time', 'DateStrLen'], format='c', attributes=atts, ncopts=ncopts)
            atts = {'long_name': 'sea surface Temperature',
                    'units': 'Celsius Degree',
                    'grid': 'fvcom_grid',
                    'type': 'data'}
            sstgrd.add_variable('sst', self.sst.sst, ['time', 'node'], attributes=atts, ncopts=ncopts)

    def add_open_boundaries(self, obcfile, reload=False):
        """

        Parameters
        ----------
        obcfile : str, pathlib.Path
            FVCOM open boundary specification file.
        reload : bool
            Set to True to overwrite any automatically or already loaded open boundary nodes. Defaults to False.

        """
        if np.any(self.grid.obc_nodes) and np.any(self.grid.types) and reload:
            # We've already got some, so warn and return.
            warn('Open boundary nodes already loaded and reload set to False.')
            return
        else:
            self.grid.nodestrings, self.grid.types, _ = read_fvcom_obc(str(obcfile))

    def add_sponge_layer(self, nodes, radius, coefficient):
        """
        Add a sponge layer. If radius or coefficient are floats, apply the same value to all nodes.

        Parameters
        ----------
        nodes : list, np.ndarray
            Grid nodes at which to add the sponge layer.
        radius : float, list, np.ndarray
            The sponge layer radius at the given nodes.
        coefficient : float, list, np.ndarray
            The sponge layer coefficient at the given nodes.

        """

        if not np.any(self.gridobc_nodes):
            raise ValueError('No open boundary nodes specified; sponge nodes cannot be defined.')

        if isinstance(radius, (float, int)):
            radius = np.repeat(radius, np.shape(nodes))
        if isinstance(coefficient, (float, int)):
            coefficient = np.repeat(radius), np.shape(nodes)

        self.grid.sponge_radius = radius
        self.grid.sponge_coefficient = coefficient

        if hasattr(self.grid, 'sponge_nodes'):
            self.grid.sponge_nodes.append(nodes)
        else:
            self.grid.sponge_nodes = nodes

    def add_grid_metrics(self, noisy=False):
        """ Calculate grid metrics. """

        grid_metrics(self.grid.tri, noisy=noisy)

    def add_tpxo_tides(self, tpxo_harmonics, predict='zeta', interval=1, constituents=['M2'], serial=False, pool_size=None, noisy=False):
        """
        Add TPXO tides at the open boundary nodes.

        Parameters
        ----------
        tpxo_harmonics : str, pathlib.Path
            Path to the TPXO harmonics netCDF file to use.
        predict : str, optional
            Type of data to predict. Select 'zeta' (default), 'u' or 'v'.
        interval : float, optional
            Interval in hours at which to generate predicted tides.
        constituents : list, optional
            List of constituent names to use in UTide.reconstruct. Defaults to ['M2'].
        serial : bool, optional
            Run in serial rather than parallel. Defaults to parallel.
        pool_size : int, optional
            Specify number of processes for parallel run. By default it uses all available.
        noisy : bool, optional
            Set to True to enable some sort of progress output. Defaults to False.

        """

        # Store everything in an object.
        self.tide = type('tide', (object,), {})()
        forcing = []

        dates = date_range(self.start + relativedelta(days=-1), self.end + relativedelta(days=1), inc=interval)
        self.tide.time = dates
        # UTide needs MATLAB times.
        times = mtime(dates)

        for obc in self.grid.obc_nodes:
            latitudes = self.grid.lat[obc]

            if predict == 'zeta':
                amplitude_var, phase_var = 'ha', 'hp'
                xdim = len(obc)
            elif predict == 'u':
                amplitude_var, phase_var = 'ua', 'up'
                xdim = len(obc)
            elif predict == 'v':
                amplitude_var, phase_var = 'va', 'vp'
                xdim = len(obc)

            with Dataset(tpxo_harmonics, 'r') as tides:
                tpxo_const = [''.join(i).upper().strip() for i in tides.variables['con'][:].astype(str)]
                cidx = [tpxo_const.index(i) for i in constituents]
                amplitudes = np.empty((xdim, len(constituents))) * np.nan
                phases = np.empty((xdim, len(constituents))) * np.nan

                for xi, xy in enumerate(zip(self.grid.lon[obc], self.grid.lat[obc])):
                    idx = [np.argmin(np.abs(tides['lon_z'][:, 0] - xy[0])),
                           np.argmin(np.abs(tides['lat_z'][0, :] - xy[1]))]
                    amplitudes[xi, :] = tides.variables[amplitude_var][cidx, idx[0], idx[1]]
                    phases[xi, :] = tides.variables[phase_var][cidx, idx[0], idx[1]]

            # Prepare the UTide inputs.
            const_idx = np.asarray([ut_constants['const']['name'].tolist().index(i) for i in constituents])
            frq = ut_constants['const']['freq'][const_idx]

            coef = Bunch(name=constituents, mean=0, slope=0)
            coef['aux'] = Bunch(reftime=729572.47916666674, lind=const_idx, frq=frq)
            coef['aux']['opt'] = Bunch(twodim=False, nodsatlint=False, nodsatnone=False,
                                       gwchlint=False, gwchnone=False, notrend=False, prefilt=[])

            args = [(latitudes[i], times, coef, amplitudes[i], phases[i], noisy) for i in range(xdim)]
            if serial:
                results = []
                for arg in args:
                    results.append(self._predict_tide(arg))
            else:
                if not pool_size:
                    pool = mp.Pool()
                else:
                    pool = mp.Pool(pool_size)
                results = pool.map(self._predict_tide, args)
                pool.close()

            forcing.append(np.asarray(results))

        # Dump the results into the object.
        setattr(self.tide, predict, forcing)

    @staticmethod
    def _predict_tide(args):
        """
        For the given time and coefficients (in coef) reconstruct the tidal elevation or current component time
        series at the given latitude.

        Parameters
        ----------
        A single tuple with the following variables:

        lats : np.ndarray
            Latitudes of the positions to predict.
        times : ndarray
            Array of matplotlib datenums (see `matplotlib.dates.num2date').
        coef : utide.utilities.Bunch
            Configuration options for utide.
        amplituds : ndarray
            Amplitude of the relevant constituents shaped [nconst].
        phases : ndarray
            Array of the phase of the relevant constituents shaped [nconst].
        noisy : bool
            Set to true to enable verbose output. Defaults to False (no output).

        Returns
        -------
        zeta : ndarray
            Time series of surface elevations.

        Notes
        -----
        Uses utide.reconstruct() for the predicted tide.

        """
        lats, times, coef, amplitude, phase, noisy = args
        if np.isnan(lats):
            return None
        coef['aux']['lat'] = lats  # float
        coef['A'] = amplitude
        coef['g'] = phase
        coef['A_ci'] = np.zeros(amplitude.shape)
        coef['g_ci'] = np.zeros(phase.shape)
        pred = reconstruct(times, coef, verbose=False)
        zeta = pred['h']

        return zeta

    def write_tides(self, output_file, ncopts={'zlib': True, 'complevel': 7}, **kwargs):
        """
        Generate a tidal elevation forcing file for the given FVCOM domain from the self.tides data.

        Parameters
        ----------
        output_file : str, pathlib.Path
            File to which to write open boundary tidal elevation forcing data.
        ncopts : dict
            Dictionary of options to use when creating the netCDF variables. Defaults to compression on.

        Remaining arguments are passed to WriteForcing.

        """

        globals = {'type': 'FVCOM TIME SERIES ELEVATION FORCING FILE',
                   'title': 'TPXO tides',
                   'history': 'File created using PyFVCOM'}
        dims = {'nobc': np.sum(self.dims.nobc), 'time': 0, 'DateStrLen': 26}

        with WriteForcing(str(output_file), dims, global_attributes=globals, clobber=True, format='NETCDF4', **kwargs) as elev:
            # Add the variables.
            atts = {'long_name': 'Open Boundary Node Number', 'grid': 'obc_grid'}
            elev.add_variable('obc_nodes', self.grid.obc_nodes, ['nobc'], attributes=atts, ncopts=ncopts)
            atts = {'long_name': 'internal mode iteration number'}
            # Not sure this variable is actually necessary.
            elev.add_variable('iint', np.arange(len(self.tides.time)), ['time'], attributes=atts, ncopts=ncopts, format=int)
            atts = {'units': 'days since 1858-11-17 00:00:00',
                    'delta_t': '0000-00-00 01:00:00',
                    'format': 'modified julian day (MJD)',
                    'time_zone': 'UTC'}
            elev.add_variable('time', date2num(self.tides.time, units='days since 1858-11-17 00:00:00'),
                                ['time'], attributes=atts, ncopts=ncopts)
            atts = {'long_name': 'Calendar Date',
                    'format': 'String: Calendar Time',
                    'time_zone': 'UTC'}
            elev.add_variable('Times', [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in self.tides.time],
                                ['time', 'DateStrLen'], format='c', attributes=atts, ncopts=ncopts)
            atts = {'long_name': 'Open Boundary Elevation',
                    'units': 'meters'}
            elev.add_variable('elevation', self.tides.zeta, ['time', 'nobc'], attributes=atts, ncopts=ncopts)

    def add_rivers(self, positions):
        """
        Add river nodes closest to the given locations.

        Parameters
        ----------
        domain : PyFVCOM.grid.Domain
            Model domain object.
        positions : np.ndarray
            Positions (in longitude/latitude).

        """
        pass

    def add_probes(self, positions, names, variables, interval, max_distance=np.inf):
        """
        Generate probe locations closest to the given locations.

        Parameters
        ----------
        positions : np.ndarray
            Positions (in longitude/latitude).
        names : np.ndarray, list
            Names of the probes defined by `positions'.
        variables : list, np.ndarray
            Variables for which to extract probe data.
        interval : float
            Interval (in seconds) at which to sample the model.
        max_distance : float, optional
            Give a maximum distance (in kilometres) beyond which the closest model grid position is considered too
            far away and thus that probe is skipped. By default, no distance filtering is applied.

        Provides
        --------
        A `probes' object is created in `self' which contains the following objects:

        file : list
            The file name to which the output will be saved.
        name : list
            The probe station names.
        grid : list
            The closest node or element IDs in the grid, depending in variable type (node-centred vs. element-centred).
        levels : list
            The vertical levels for the requested depth-resolved outputs (if any, otherwise None)
        description : list
            The descriptions of each requested variable.
        variables : list
            The variables requested for each position.
        long_name : list
            The long names of each variable.
        interval : float
            The interval at which the model is sampled.

        """

        # Store everything in an object to make it cleaner passing stuff around.
        self.probes = type('probes', (object,), {})()

        self.probes.interval = interval  # currently assuming the same for all probes

        # These lists are incomplete! Missing values just use the current variable name and no units.
        description_prefixes = {'el': 'Surface elevation at {}',
                                'u': 'u-velocity component at {}',
                                'v': 'v-velocity component at {}',
                                'ua': 'Depth-averaged u-velocity component at {}',
                                'va': 'Depth-averaged v-velocity component at {}',
                                'ww': 'Vertical velocity at {}',
                                'w': 'Vertical velocity on sigma levels at {}',
                                'rho1': 'Density at {}',
                                't1': 'Temperature at {}',
                                's1': 'Salinity at {}'}
        long_names_choices = {'el': 'Surface elevation (m)',
                              'v': 'u-velocity (ms^{-1})',
                              'u': 'v-velocity (ms^{-1})',
                              'va': 'Depth-averaged u-velocity (ms^{-1})',
                              'ua': 'Depth-averaged v-velocity (ms^{-1})',
                              'ww': 'Vertical velocity (ms^{-1})',
                              'w': 'Vertical velocity on sigma levels (ms^{-1})',
                              'rho1': 'Density (kg/m^{3})',
                              't1': 'Temperature (Celsius)',
                              's1': 'Salinity (PSU)'}

        self.probes.name = []
        self.probes.variables = []
        self.probes.grid = []
        self.probes.levels = []
        self.probes.description = []
        self.probes.long_names = []

        # We need to check whether we're a node- or element-based variable. Since there are only a small number of
        # element-centred variables available as probe output, check for those, otherwise assume node-based.
        element_variables = ['u', 'v', 'ua', 'va', 'w', 'ww', 'uice2', 'vice2']
        depth_variables = ['u', 'v', 'w', 'ww']

        for (position, site) in zip(positions, names):
            current_name = []
            current_grid = []
            current_levels = []
            current_description = []
            current_long_names = []
            current_variables = []
            for variable in variables:
                if variable in element_variables:
                    grid_id = self.closest_element(position, threshold=max_distance, vincenty=True)
                else:
                    grid_id = self.closest_node(position, threshold=max_distance, vincenty=True)
                if variable in depth_variables:
                    sigma = [1, len(self.grid.siglay)]
                else:
                    sigma = None
                current_grid.append(grid_id)
                current_name.append('{}_{}.dat'.format(site, variable))
                current_levels.append(sigma)
                if variable in description_prefixes:
                    desc = description_prefixes[variable].format(site)
                else:
                    desc = '{} at {}'.format(variable, site)
                current_description.append(desc)
                if variable in long_names_choices:
                    long = long_names_choices[variable]
                else:
                    long = '{}'.format(variable)
                current_long_names.append(long)
                current_variables.append(variable)

            self.probes.grid.append(current_grid)
            self.probes.name.append(current_name)
            self.probes.variables.append(current_variables)
            self.probes.levels.append(current_levels)
            self.probes.description.append(current_description)
            self.probes.long_names.append(current_long_names)

    def write_probes(self, output_file):
        """
        Take the output of add_probes and write it to FVCOM-formatted ASCII.

        Parameters
        ----------
        output_file : str
            Path to the output file name list to create.

        """
        pass

        if not hasattr(self, 'probes'):
            raise AttributeError('No probes object found. Please run PyFVCOM.preproc.add_probes() first.')

        with open(output_file, 'w') as f:
            grid = self.probes.grid
            name = self.probes.name
            levels = self.probes.levels
            description = self.probes.description
            long_names = self.probes.long_names
            variables = self.probes.variables
            # First level of iteration is the site. Transpose with map.
            for probes in list(map(list, zip(*[grid, name, levels, description, long_names, variables]))):
                # Second level is the variable
                for loc, site, sigma, desc, long_name, variable in list(map(list, zip(*probes))):
                    # Skip positions with grid IDs as None. These are sites which were too far from the nearest grid
                    # point.
                    if not grid:
                        continue

                    f.write('&NML_PROBE\n')
                    f.write(' PROBE_INTERVAL = "seconds={:.1f}",\n'.format(self.probes.interval))
                    f.write(' PROBE_LOCATION = {:d},\n'.format(loc))
                    f.write(' PROBE_TITLE = "{}",\n'.format(site))
                    # If we've got something which is vertically resolved, output the vertical levels.
                    if np.any(sigma):
                        f.write(' PROBE_LEVELS = {:d} {:d},\n'.format(*sigma))
                    f.write(' PROBE_DESCRIPTION = "{}",\n'.format(desc))
                    f.write(' PROBE_VARIABLE = "{}",\n'.format(variable))
                    f.write(' PROBE_VAR_NAME = "{}"\n'.format(long_name))
                    f.write('/\n')

class WriteForcing:
    """ Create an FVCOM netCDF input file. """

    def __init__(self, filename, dimensions, global_attributes=None, **kwargs):
        """ Create a netCDF file.

        Parameters
        ----------
        filename : str, pathlib.Path
            Output netCDF path.
        dimensions : dict
            Dictionary of dimension names and sizes.
        global_attributes : dict, optional
            Global attributes to add to the netCDF file.
        Remaining arguments are passed to netCDF4.Dataset.

        Returns
        -------
        nc : netCDF4.Dataset
            The netCDF file object.

        """

        self.nc = Dataset(str(filename), 'w', **kwargs)

        for dimension in dimensions:
            self.nc.createDimension(dimension, dimensions[dimension])

        for attribute in global_attributes:
            setattr(self.nc, attribute, global_attributes[attribute])

    def add_variable(self, name, data, dimensions, attributes=None, format='f4', ncopts={}):
        """
        Create a `name' variable with the given `attributes' and `data'.

        Parameters
        ----------
        name : str
            Variable name to add.
        data : np.ndararay, list, float, str
            Data to add to the netCDF file object.
        dimensions : list, tuple
            List of dimension names to apply to the new variable.
        attributes : dict, optional
            Attributes to add to the netCDF variable object.
        format : str, optional
            Data format for the new variable. Defaults to 'f4' (float32).
        ncopts : dict
            Dictionary of options to use when creating the netCDF variables.

        """

        var = self.nc.createVariable(name, format, dimensions, **ncopts)
        for attribute in attributes:
            setattr(var, attribute, attributes[attribute])

        var[:] = data

        setattr(self, name, var)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Tidy up the netCDF file handle. """
        self.nc.close()
