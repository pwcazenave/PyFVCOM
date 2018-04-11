"""
Tools to prepare data for an FVCOM run.

A very gradual port of the most used functions from the MATLAB toolbox:
    https://github.com/pwcazenave/fvcom-toolbox/tree/master/fvcom_prepro/

Author(s):

Mike Bedington (Plymouth Marine Laboratory)
Pierre Cazenave (Plymouth Marine Laboratory)

"""

import copy
import inspect
import multiprocessing
from datetime import datetime
from functools import partial
from pathlib import Path
from warnings import warn

import numpy as np
import scipy.optimize
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset, date2num, num2date, stringtochar
from scipy.interpolate import RegularGridInterpolator

from PyFVCOM.coordinate import utm_from_lonlat, lonlat_from_utm
from PyFVCOM.grid import Domain, grid_metrics, read_fvcom_obc, nodes2elems
from PyFVCOM.grid import OpenBoundary, find_connected_elements
from PyFVCOM.grid import find_bad_node
from PyFVCOM.grid import write_fvcom_mesh, connectivity, haversine_distance
from PyFVCOM.read import FileReader
from PyFVCOM.utilities.general import flatten_list
from PyFVCOM.utilities.time import date_range


class Model(Domain):
    """
    Everything related to making a new model run.

    There should be more use of objects here. For example, each open boundary should be a Boundary object which has
    methods for interpolating data onto it (tides, temperature, salinity, ERSEM variables etc.). The coastline could
    be an object which has methods related to rivers and checking depths. Likewise, the model grid object could
    contain methods for interpolating SST, creating restart files etc.

    """

    def __init__(self, start, end, *args, **kwargs):

        sampling = 1
        if 'sampling' in kwargs:
            sampling = kwargs['sampling']
            kwargs.pop('sampling')

        # Inherit everything from PyFVCOM.grid.Domain, but extend it for our purposes. This doesn't work with Python 2.
        super().__init__(*args, **kwargs)

        self.noisy = False
        self.debug = False
        if 'noisy' in kwargs:
            self.noisy = kwargs['noisy']

        # Initialise things so we can add attributes to them later.
        self.time = type('time', (), {})()
        self.sigma = type('sigma', (), {})()
        self.tide = type('tide', (), {})()
        self.sst = type('sst', (), {})()
        self.nest = type('nest', (), {})()
        self.regular = None

        # Make some potentially useful time representations.
        self.start = start
        self.end = end
        self.sampling = sampling
        self.__add_time()

        # Initialise the open boundary objects from the nodes we've read in from the grid (if any).
        self.__initialise_open_boundaries_on_nodes()

        # Initialise the river structure.
        self.__prep_rivers()

        # Add the coastline to the grid object for use later on.
        *_, bnd = connectivity(np.array((self.grid.lon, self.grid.lat)).T, self.grid.triangles)
        self.grid.coastline = np.argwhere(bnd)
        # Remove the open boundaries, if we have them.
        if np.any(self.grid.open_boundary_nodes):
            land_only = np.isin(np.squeeze(np.argwhere(bnd)), flatten_list(self.grid.open_boundary_nodes), invert=True)
            self.grid.coastline = np.squeeze(self.grid.coastline[land_only])

    def __prep_rivers(self):
        """ Create a few object and attributes which are useful for the river data. """
        self.river = type('river', (object,), {})()
        self.dims.river = 0  # assume no rivers.

        self.river.history = ''
        self.river.info = ''
        self.river.source = ''

    def __add_time(self):
        """
        Add time variables we might need for the various bits of processing.

        """

        self.time.datetime = date_range(self.start, self.end, inc=self.sampling)
        self.time.time = date2num(getattr(self.time, 'datetime'), units='days since 1858-11-17 00:00:00')
        self.time.Itime = np.floor(getattr(self.time, 'time'))  # integer Modified Julian Days
        self.time.Itime2 = (getattr(self.time, 'time') - getattr(self.time, 'Itime')) * 24 * 60 * 60 * 1000  # milliseconds since midnight
        self.time.Times = [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in getattr(self.time, 'datetime')]

    def __initialise_open_boundaries_on_nodes(self):
        """ Add the relevant node-based grid information for any open boundaries we've got. """

        self.open_boundaries = []
        if np.any(self.grid.open_boundary_nodes):
            for nodes in self.grid.open_boundary_nodes:
                self.open_boundaries.append(OpenBoundary(nodes))
                # Add the positions of the relevant bits of information.
                for attribute in ('lon', 'lat', 'x', 'y', 'h'):
                    try:
                        setattr(self.open_boundaries[-1].grid, attribute, getattr(self.grid, attribute)[nodes, ...])
                    except AttributeError:
                        pass
                # Add all the time data.
                setattr(self.open_boundaries[-1].time, 'start', self.start)
                setattr(self.open_boundaries[-1].time, 'end', self.end)

    def __update_open_boundaries(self):
        """
        Call this when we've done something which affects the open boundary objects and we need to update their
        properties.

        For example, this updates sigma information if we've added the sigma distribution to the Model object.

        """

        # Add the sigma data to any open boundaries we've got loaded.
        for boundary in self.open_boundaries:
            for attribute in self.obj_iter(self.sigma):
                try:
                    # Ignore element-based data for now.
                    if 'center' not in attribute:
                        setattr(boundary.sigma, attribute, getattr(self.sigma, attribute)[boundary.nodes, ...])
                except (IndexError, TypeError):
                    setattr(boundary.sigma, attribute, getattr(self.sigma, attribute))
                except AttributeError:
                    pass

    def write_grid(self, grid_file, depth_file=None):
        """
        Write out the unstructured grid data to file.

        grid_file : str, pathlib.Path
            Name of the file to which to write the grid.
        depth_file : str, pathlib.Path, optional
            If given, also write out the bathymetry file.

        """
        grid_file = str(grid_file)
        if depth_file:
            depth_file = str(depth_file)

        nodes = np.arange(self.dims.node) + 1
        if self.grid.native_coordinates.lower() == 'spherical':
            x, y = self.grid.lon, self.grid.lat
        else:
            x, y = self.grid.x, self.grid.y

        write_fvcom_mesh(self.grid.triangles, nodes, x, y, self.grid.h, grid_file, extra_depth=depth_file)

    def write_coriolis(self, coriolis_file):
        """
        Write an FVCOM-formatted Coriolis file.

        Parameters
        ----------
        coriolis_file : str, pathlib.Path
            Name of the file to which to write the coriolis data.

        """

        if isinstance(coriolis_file, str):
            coriolis_file = Path(coriolis_file)

        with coriolis_file.open('w') as f:
            if self.grid.native_coordinates.lower() == 'spherical':
                x, y = self.grid.lon, self.grid.lat
            else:
                x, y = self.grid.x, self.grid.y

            f.write('Node Number = {:d}\n'.format(self.dims.node))
            for line in zip(x, y, self.grid.lat):
                f.write('{:.6f} {:.6f} {:.6f}\n'.format(*line))

    def add_bed_roughness(self, roughness):
        """
        Add a uniform or spatially varying bed roughness to the model.

        Parameters
        ----------
        roughness : float, np.ndarray
            The bed roughness (in metres).

        """

        setattr(self.grid, 'roughness', roughness)

    def write_bed_roughness(self, roughness_file, ncopts={'zlib': True, 'complevel': 7}, **kwargs):
        """
        Write the bed roughness to netCDF.

        Parameters
        ----------
        roughness_file:
            File to which to write bed roughness data.
        ncopts : dict, optional
            Dictionary of options to use when creating the netCDF variables. Defaults to compression on.

        Remaining arguments are passed to WriteForcing.

        """
        globals = {'title': 'bottom roughness',
                   'history': 'File created using {} from PyFVCOM'.format(inspect.stack()[0][3])}
        dims = {'nele': self.dims.nele}

        with WriteForcing(str(roughness_file), dims, global_attributes=globals, clobber=True, format='NETCDF4', **kwargs) as z0:
            # Add the variables.
            atts = {'long_name': 'bottom roughness', 'units': 'm', 'type': 'data'}
            z0.add_variable('z0b', self.grid.roughness, ['nele'], attributes=atts, ncopts=ncopts)
            # Pretty sure this variable isn't necessary for an ordinary physics run. At least, we've never written it
            #  to file to date.
            atts = {'long_name': 'bottom roughness minimum', 'units': 'None', 'type': 'data'}
            z0.add_variable('cbcmin', None, ['nele'], attributes=atts, ncopts=ncopts)

    def interp_sst_assimilation(self, sst_dir, offset=0, serial=False, pool_size=None, noisy=False):
        """
        Interpolate SST data from remote sensing data onto the supplied model
        grid.

        Parameters
        ----------
        sst_dir : str, pathlib.Path
            Path to directory containing the SST data. Assumes there are directories per year within this directory.
        offset : int, optional
            Number of days by which to offset the time period in the time series. Defaults to zero.
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

        if isinstance(sst_dir, str):
            sst_dir = Path(sst_dir)

        # Make daily data.
        dates = date_range(self.start - relativedelta(days=offset), self.end + relativedelta(days=offset))

        sst_files = []
        for date in dates:
            sst_base = sst_dir / Path(str(date.year))
            sst_files += list(sst_base.glob('*{}*.nc'.format(date.strftime('%Y%m%d'))))

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
                pool = multiprocessing.Pool()
            else:
                pool = multiprocessing.Pool(pool_size)
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
                   'history': 'File created using {} from PyFVCOM'.format(inspect.stack()[0][3]),
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
            sstgrd.write_fvcom_time(self.sst.time)
            atts = {'long_name': 'sea surface Temperature',
                    'units': 'Celsius Degree',
                    'grid': 'fvcom_grid',
                    'type': 'data'}
            sstgrd.add_variable('sst', self.sst.sst, ['time', 'node'], attributes=atts, ncopts=ncopts)

    def add_sigma_coordinates(self, sigma_file, noisy=False):
        """
        Read in a sigma coordinates file and apply to the grid object.

        Parameters
        ----------
        sigma_file : str, pathlib.Path
            FVCOM sigma coordinates .dat file.

        Notes
        -----
        This is more or less a direct python translation of the original MATLAB fvcom-toolbox function read_sigma.m

        """

        sigma_file = str(sigma_file)

        # Make an object to store the sigma data.
        self.sigma = type('sigma', (object,), {})()

        with open(sigma_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                option, value = line.split('=')
                option = option.strip().lower()
                value = value.strip()

                # Grab the various bits we need.
                if option == 'number of sigma levels':
                    nlev = int(value)
                elif option == 'sigma coordinate type':
                    sigtype = value
                elif option == 'sigma power':
                    sigpow = float(value)
                elif option == 'du':
                    du = float(value)
                elif option == 'dl':
                    dl = float(value)
                elif option == 'min constant depth':
                    min_constant_depth = float(value)
                elif option == 'ku':
                    ku = float(value)
                elif option == 'kl':
                    kl = float(value)
                elif option == 'zku':
                    s = [float(i) for i in value.split(' ')]
                    zku = np.zeros(ku)
                    for i in range(len(ku)):
                        zku[i] = s[i]
                elif option == 'zkl':
                    s = [float(i) for i in value.split(' ')]
                    zkl = np.zeros(kl)
                    for i in range(len(kl)):
                        zkl[i] = s[i]

        # Do some checks if we've got uniform or generalised coordinates to make sure the input is correct.
        if sigtype == 'GENERALIZED':
            if len(zku) != ku:
                raise ValueError('Number of zku values does not match the number specified in ku')
            if len(zkl) != kl:
                raise ValueError('Number of zkl values does not match the number specified in kl')

        # Calculate the sigma level distributions at each grid node.
        if sigtype.lower() == 'generalized':
            sigma_levels = np.empty((self.dims.node, nlev)) * np.nan
            for i in range(self.dims.node):
                sigma_levels[i, :] = self.sigma_generalized(nlev, dl, du, kl, ku, zkl, zku, self.grid.h[i], min_constant_depth)
        elif sigtype.lower() == 'uniform':
            sigma_levels = np.repeat(self.sigma_geometric(nlev, 1), self.dims.node).reshape(self.dims.node, -1)
        elif sigtype.lower() == 'geometric':
            sigma_levels = np.repeat(self.sigma_geometric(nlev, sigpow), self.dims.node).reshape(self.dims.node, -1)
        elif sigtype.lower() == 'tanh':
            sigma_levels = np.repeat(self.sigma_tanh(nlev, dl, du), self.dims.node).reshape(self.dims.node, -1)
        else:
            raise ValueError('Unrecognised sigtype {} (is it supported?)'.format(sigtype))

        # Create a sigma layer variable (i.e. midpoint in the sigma levels).
        sigma_layers = sigma_levels[:, 0:-1] + (np.diff(sigma_levels, axis=1) / 2)

        self.sigma.type = sigtype
        self.sigma.layers = sigma_layers
        self.sigma.levels = sigma_levels
        self.sigma.layers_center = nodes2elems(self.sigma.layers.T, self.grid.triangles).T
        self.sigma.levels_center = nodes2elems(self.sigma.levels.T, self.grid.triangles).T
        self.sigma.power = sigpow

        # Make some depth-resolved sigma distributions.
        self.sigma.layers_z = self.grid.h[:, np.newaxis] * self.sigma.layers
        self.sigma.layers_center_z = self.grid.h_center[:, np.newaxis] * self.sigma.layers_center
        self.sigma.levels_z = self.grid.h [:, np.newaxis] * self.sigma.levels
        self.sigma.levels_center_z = self.grid.h_center[:, np.newaxis]  * self.sigma.levels_center

        # Make some dimensions
        self.dims.levels = nlev
        self.dims.layers = self.dims.levels - 1

        # Print the sigma file configuration we've parsed.
        if noisy:
            # Should be present in all sigma files.
            print('nlev\t{:d}\n'.format(nlev))
            print('sigtype\t%s\n'.format(sigtype))

            # Only present in geometric sigma files.
            if sigtype == 'GEOMETRIC':
                print('sigpow\t{:d}\n'.format(sigpow))

            # Only in the generalised or uniform sigma files.
            if sigtype == 'GENERALIZED':
                print('du\t{:d}\n'.format(du))
                print('dl\t{:d}\n'.format(dl))
                print('min_constant_depth\t%f\n'.format(min_constant_depth))
                print('ku\t{:d}\n'.format(ku))
                print('kl\t{:d}\n'.format(kl))
                print('zku\t{:d}\n'.format(zku))
                print('zkl\t{:d}\n'.format(zkl))

        # Update the open boundaries.
        self.__update_open_boundaries()

    def sigma_generalized(self, levels, dl, du, kl, ku, zkl, zku, h, hmin):
        """
        Generate a generalised sigma coordinate distribution.

        Parameters
        ----------
        levels : int
            Number of sigma levels.
        dl : float
            The lower depth boundary from the bottom, down to which the layers are uniform thickness.
        du : float
            The upper depth boundary from the surface, up to which the layers are uniform thickness.
        kl : float
            Number of layers in the upper water column
        ku : float
            Number of layers in the lower water column
        zkl : list, np.ndarray
            Upper water column layer thicknesses.
        zku : list, np.ndarray
            Lower water column layer thicknesses.
        h : float
            Water depth.
        hmin : float
            Minimum water depth.

        Returns
        -------
        dist : np.ndarray
            Generalised vertical sigma coordinate distribution.

        """

        dist = np.zeros(levels)

        if h < hmin:
            dist = self.sigma_tanh(levels, du, dl)
        else:
            dr = (h - du - dl) / h / (levels - ku - kl - 1)

            for k in range(1, ku + 1):
                dist[k] = dist[k - 1] - zku[k - 1] / h

            for k in range(ku + 1, levels - kl):
                dist[k] = dist[k - 1] - dr

            kk = 0
            for k in range(-kl, 0):
                dist[k] = dist[k - 1] - zkl[kk] / h
                kk += 1

        return dist

    def sigma_geometric(self, levels, p_sigma):
        """
        Generate a geometric sigma coordinate distribution.

        Parameters
        ----------
        levels : int
            Number of sigma levels.
        p_sigma : float
            Power value. 1 for uniform sigma layers, 2 for parabolic function. See page 308-309 in the FVCOM manual
            for examples.

        Returns
        -------
        dist : np.ndarray
            Geometric vertical sigma coordinate distribution.

        """

        dist = np.empty(levels) * np.nan

        if p_sigma == 1:
            for k in range(levels):
                dist[k] = -((k - 1) / (levels - 1))**p_sigma
        else:
            split = int(np.floor((levels + 1) / 2))
            for k in range(split):
                dist[k] = -(k / ((levels + 1) / 2 - 1))**p_sigma / 2
            # Mirror the first half to make the second half of the parabola. We need to offset by one if we've got an
            # odd number of levels.
            if levels % 2 == 0:
                dist[split:] = -(1 - -dist[:split])[::-1]
            else:
                dist[split:] = -(1 - -dist[:split - 1])[::-1]

        return dist

    def sigma_tanh(self, levels, dl, du):
        """
        Generate a hyperbolic tangent vertical sigma coordinate distribution.

        Parameters
        ----------
        levels : int
            Number of sigma levels (layers + 1)
        dl : float
            The lower depth boundary from the bottom down to which the coordinates are parallel with uniform thickness.
        du : float
            The upper depth boundary from the surface up to which the coordinates are parallel with uniform thickness.

        Returns
        -------
        dist : np.ndarray
            Hyperbolic tangent vertical sigma coordinate distribution.

        """

        kbm1 = levels - 1

        dist = np.zeros(levels)

        # Loop has to go to kbm1 + 1 (or levels) since python ranges stop before the end point.
        for k in range(1, levels):
            x1 = dl + du
            x1 = x1 * (kbm1 - k) / (kbm1)
            x1 = x1 - dl
            x1 = np.tanh(x1)
            x2 = np.tanh(dl)
            x3 = x2 + np.tanh(du)
            # k'th position starts from 1 which is right because we want the initial value to be zero for sigma levels.
            dist[k] = (x1 + x2) / x3 - 1

        return dist

    def hybrid_sigma_coordinate(self, levels, transition_depth, upper_layer_depth, lower_layer_depth,
                                total_upper_layers, total_lower_layers, noisy=False):
        """
        Create a hybrid vertical coordinate system.

        Parameters
        ----------
        levels : int
            Number of vertical levels.
        transition_depth : float
            Transition depth of the hybrid coordinates
        upper_layer_depth : float
            Upper water boundary thickness (metres)
        lower_layer_depth : float
            Lower water boundary thickness (metres)
        total_upper_layers : int
            Number of layers in the DU water column
        total_lower_layers : int
            Number of layers in the DL water column

        Populates
        ---------
        self.dims.layers : int
            Number of sigma layers.
        self.dims.levels : int
            Number of sigma levels.
        self.sigma.levels : np.ndarray
            Sigma levels at the nodes
        self.sigma.layers : np.ndarray
            Sigma layers at the nodes
        self.sigma.levels_z : np.ndarray
            Water depth levels at the nodes
        self.sigma.layers_z : np.ndarray
            Water depth layers at the nodes
        self.sigma.levels_center : np.ndarray
            Sigma levels at the elements
        self.sigma.layers_center : np.ndarray
            Sigma layers at the elements
        self.sigma.levels_z_center : np.ndarray
            Water depth levels at the elements
        self.sigma.layers_z_center : np.ndarray
            Water depth layers at the elements

        """

        # Make an object to store the sigma data.
        self.sigma = type('sigma', (object,), {})()

        self.dims.levels = levels
        self.dims.layers = self.dims.levels - 1

        # Optimise the transition depth to minimise the error between the uniform region and the hybrid region.
        if noisy:
            print('Optimising the hybrid coordinates... ')
        upper_layer_thickness = np.repeat(upper_layer_depth / total_upper_layers, total_upper_layers)
        lower_layer_thickness = np.repeat(lower_layer_depth / total_lower_layers, total_lower_layers)
        optimisation_settings = {'maxfun': 5000, 'maxiter': 5000, 'ftol': 10e-5, 'xtol': 1e-7}
        fparams = lambda depth_guess: self.__hybrid_coordinate_hmin(depth_guess, self.dims.levels,
                                                                    upper_layer_depth, lower_layer_depth,
                                                                    total_upper_layers, total_lower_layers,
                                                                    upper_layer_thickness, lower_layer_thickness)
        optimised_depth = scipy.optimize.fmin(func=fparams, x0=transition_depth, disp=False, **optimisation_settings)
        min_error = transition_depth - optimised_depth  # this isn't right
        self.sigma.transition_depth = optimised_depth

        if noisy:
            print('Hmin found {} with a maximum error in vertical distribution of {} metres\n'.format(optimised_depth,
                                                                                                      min_error))

        # Calculate the sigma level distributions at each grid node.
        sigma_levels = np.empty((self.dims.node, self.dims.levels)) * np.nan
        for i in range(self.dims.node):
            sigma_levels[i, :] = self.sigma_generalized(levels, lower_layer_depth, upper_layer_depth,
                                                        total_lower_layers, total_upper_layers,
                                                        lower_layer_thickness, upper_layer_thickness,
                                                        self.grid.h[i], optimised_depth)

        # Create a sigma layer variable (i.e. midpoint in the sigma levels).
        sigma_layers = sigma_levels[:, 0:-1] + (np.diff(sigma_levels, axis=1) / 2)

        # Add to the grid object.
        self.sigma.type = 'GENERALIZED'  # hybrid is a special case of generalised vertical coordinates
        self.sigma.upper_layer_depth = upper_layer_depth
        self.sigma.lower_layer_depth = lower_layer_depth
        self.sigma.total_upper_layers = total_upper_layers
        self.sigma.total_lower_layers = total_lower_layers
        self.sigma.upper_layer_thickness = upper_layer_thickness
        self.sigma.lower_layer_thickness = lower_layer_thickness
        self.sigma.layers = sigma_layers
        self.sigma.levels = sigma_levels
        # Transpose on the way in and out so the slicing within nodes2elems works properly.
        self.sigma.layers_center = nodes2elems(self.sigma.layers.T, self.grid.triangles).T
        self.sigma.levels_center = nodes2elems(self.sigma.levels.T, self.grid.triangles).T

        # Make some depth-resolved sigma distributions.
        self.sigma.layers_z = self.grid.h[:, np.newaxis] * self.sigma.layers
        self.sigma.layers_center_z = self.grid.h_center[:, np.newaxis] * self.sigma.layers_center
        self.sigma.levels_z = self.grid.h [:, np.newaxis] * self.sigma.levels
        self.sigma.levels_center_z = self.grid.h_center[:, np.newaxis]  * self.sigma.levels_center

        # Update the open boundaries.
        self.__update_open_boundaries()

    def __hybrid_coordinate_hmin(self, h, levels, du, dl, ku, kl, zku, zkl):
        """
        Helper function to find the relevant minimum depth.

        Parameters
        ----------
        h : float
            Transition depth of the hybrid coordinates?
        levels : int
            Number of vertical levels (layers + 1)
        du : float
            Upper water boundary thickness (metres)
        dl : float
            Lower water boundary thickness (metres)
        ku : int
            Layer number in the water column of DU
        kl : int
            Layer number in the water column of DL

        Returns
        -------
        zz : float
            Minimum water depth.

        """
        # This is essentially identical to self.sigma_tanh, so we should probably just use that instead.
        z0 = self.sigma_tanh(levels, du, dl)
        z2 = np.zeros(levels)

        # s-coordinates
        x1 = (h - du - dl)
        x2 = x1 / h
        dr = x2 / (levels - ku - kl - 1)

        for k in range(1, ku + 1):
            z2[k] = z2[k - 1] - (zku[k - 1] / h)

        for k in range(ku + 2, levels - kl):
            z2[k] = z2[k - 1] - dr

        kk = 0
        for k in range(levels - kl + 1, levels):
            kk += 1
            z2[k] = z2[k - 1] - (zkl[kk] / h)

        zz = np.max(h * z0 - h * z2)

        return zz

    def write_sigma(self, sigma_file):
        """
        Write the sigma distribution to file.

        Parameters
        ----------
        sigma_file : str, pathlib.Path
            Path to which to save sigma data.

        TODO:
        -----
        - Add support for writing all the sigma file formats.

        """

        if isinstance(sigma_file, str):
            sigma_file = Path(sigma_file)

        with sigma_file.open('w') as f:
            # All types of sigma distribution have the two following lines.
            f.write('NUMBER OF SIGMA LEVELS = {:d}\n'.format(self.dims.levels))
            f.write('SIGMA COORDINATE TYPE = {}\n'.format(self.sigma.type))
            if self.sigma.type.lower() == 'generalized':
                f.write('DU = {:4.1f}\n'.format(self.sigma.upper_layer_depth))
                f.write('DL = {:4.1f}\n'.format(self.sigma.lower_layer_depth))
                # Why do we go to all the trouble of finding the transition depth only to round it anyway?
                f.write('MIN CONSTANT DEPTH = {:10.1f}\n'.format(np.round(self.sigma.transition_depth[0])))  # don't like the [0]
                f.write('KU = {:d}\n'.format(self.sigma.total_upper_layers))
                f.write('KL = {:d}\n'.format(self.sigma.total_lower_layers))
                # Add the thicknesses with a loop.
                f.write('ZKU = ')
                for ii in self.sigma.upper_layer_thickness:
                    f.write('{:4.1f}'.format(ii))
                f.write('\n')
                f.write('ZKL = ')
                for ii in self.sigma.lower_layer_thickness:
                    f.write('{:4.1f}'.format(ii))
                f.write('\n')
            elif self.sigma.type.lower() == 'geometric':
                f.write('SIGMA POWER = {:.1f}\n'.format(self.sigma.power))

    def add_open_boundaries(self, obc_file, reload=False):
        """
        Add open boundaries from a given FVCOM-formatted open boundary file.

        Parameters
        ----------
        obc_file : str, pathlib.Path
            FVCOM open boundary specification file.
        reload : bool
            Set to True to overwrite any automatically or already loaded open boundary nodes. Defaults to False.

        """
        if np.any(self.grid.open_boundary_nodes) and np.any(self.grid.types) and reload:
            # We've already got some, so warn and return.
            warn('Open boundary nodes already loaded and reload set to False.')
            return
        else:
            self.grid.open_boundary_nodes, self.grid.types, _ = read_fvcom_obc(str(obc_file))

    def write_sponge(self, sponge_file):
        """
        Write out the sponge data to an FVCOM-formatted ASCII file.

        Parameters
        ----------
        sponge_file : str, pathlib.Path
            Path to the file to create.

        """

        if isinstance(sponge_file, str):
            sponge_file = Path(sponge_file)

        # Work through all the open boundary objects collecting all the information we need and then dump that to file.
        radius = []
        coefficient = []
        for boundary in self.open_boundaries:
            radius += boundary.sponge_radius.tolist()
            coefficient += boundary.sponge_coefficient.tolist()

        # I feel like this should be in self.dims.
        number_of_nodes = len(radius)

        with sponge_file.open('w') as f:
            f.write('Sponge Node Number = {:d}\n'.format(number_of_nodes))
            for node in zip(np.arange(number_of_nodes) + 1, radius, coefficient):
                f.write('{} {:.6f} {:.6f}\n'.format(*node))

    def add_grid_metrics(self, noisy=False):
        """
        Calculate grid metrics.

        Parameters
        ----------
        noisy : bool, optional
            Set to True to enable verbose output. Defaults to False.

        """

        grid_metrics(self.grid.tri, noisy=noisy)

    def write_tides(self, output_file, ncopts={'zlib': True, 'complevel': 7}, **kwargs):
        """
        Generate a tidal elevation forcing file for the given FVCOM domain from the self.tides data.

        Parameters
        ----------
        output_file : str, pathlib.Path
            File to which to write open boundary tidal elevation forcing data.
        ncopts : dict, optional
            Dictionary of options to use when creating the netCDF variables. Defaults to compression on.

        Remaining arguments are passed to WriteForcing.

        """

        globals = {'type': 'FVCOM TIME SERIES ELEVATION FORCING FILE',
                   'title': 'TPXO tides',
                   'history': 'File created using {} from PyFVCOM'.format(inspect.stack()[0][3])}
        dims = {'nobc': self.dims.open_boundary_nodes, 'time': 0, 'DateStrLen': 26}

        with WriteForcing(str(output_file), dims, global_attributes=globals, clobber=True, format='NETCDF4', **kwargs) as elev:
            # Add the variables.
            atts = {'long_name': 'Open Boundary Node Number', 'grid': 'obc_grid'}
            # Don't forget to offset the open boundary node IDs by one to account for Python indexing!
            elev.add_variable('obc_nodes', np.asarray(flatten_list(self.grid.open_boundary_nodes)) + 1, ['nobc'], attributes=atts, ncopts=ncopts, format='i')
            atts = {'long_name': 'internal mode iteration number'}
            # Not sure this variable is actually necessary.
            elev.add_variable('iint', np.arange(len(self.tide.time)), ['time'], attributes=atts, ncopts=ncopts, format='i')
            elev.write_fvcom_time(self.tide.time)
            atts = {'long_name': 'Open Boundary Elevation', 'units': 'meters'}
            elev.add_variable('elevation', np.asarray(self.tide.zeta).T, ['time', 'nobc'], attributes=atts, ncopts=ncopts)

    def add_rivers(self, positions, names, times, flux, temperature, salinity, threshold=np.inf, history='', info='', ersem=None, sediments=None):
        """
        Add river nodes closest to the given locations.

        Parameters
        ----------
        positions : np.ndarray
            Positions (in longitude/latitude).
        names : np.ndarray
            River names as strings.
        times : np.ndarray
            Array of datetime objects for the river data.
        flux : np.ndarray
            River discharge data (m^3s{^-1}) [time, river]
        temperature : np.ndarray
            River temperature data (degrees Celsius) [time, river]
        salinity : np.ndarray
            River salinity data (PSU) [time, river]
        threshold : float, optional
            Distance (in kilometres) beyond which a model node is considered too far from the current river position.
            Such rivers are omitted from the forcing.
        history : str
            String added to the `history' global attribute.
        info : str
            String added to the `info' global attribute.
        ersem : dict
            If supplied, a dictionary whose keys are variable names to add to the river object and whose values are
            the corresponding river data. These should match the shape of the flux, temperature and salinity data.
        sediment : dict
            If supplied, either dictionary whose keys are variable names to add to the river object and whose values are
            the corresponding river data. These should match the shape of the flux, temperature and salinity data.

        Provides
        --------
        node : list, np.ndarray
            List of model grid nodes at which rivers will be discharged.
        flux : np.ndarray
            Time series of the river flux.
        temperature : np.ndarray
            Time series of the river temperature.
        salinity : np.ndarray
            Time series of the river salinity.

        If `ersem' is True, then the variables supplied in the `ersem' dict are also added to the `river' object.
        Note: a number of variables are automatically created if not given within the `ersem' dict, based on values
        from PML's Western Channel Observatory L4 buoy data. These are: 'Z4_c', 'Z5c', 'Z5n', 'Z5p', 'Z6c', 'Z6n' and
        'Z6p'.

        If `sediment' is supplied, then the variables in the sediment are added. Cohesive sediments are expected to have
        names like 'mud_*' and non-cohesive sediments names like 'sand_*'.

        TO DO: Add Reg's formula for calculating spm from flux.

        """

        # Overwrite history/info attributes if we've been given them.
        if history:
            self.river.history = history

        if info:
            self.river.info = info

        self.river.time = times

        nodes = []
        river_index = []
        grid_pts = np.squeeze(np.asarray([self.grid.lon[self.grid.coastline], self.grid.lat[self.grid.coastline]]).T)
        for ri, position in enumerate(positions):
            # We can't use closest_node here as the candidates we need to search within are the coastline nodes only
            # (closest_node works on the currently loaded model grid only).
            dist = np.asarray([haversine_distance(pt_1, position) for pt_1 in grid_pts])
            breached_distance = dist < threshold
            if np.any(breached_distance):
                nodes.append(self.grid.coastline[np.argmin(dist)][0])
                river_index.append(ri)

        self.river.node = nodes
        self.dims.river = len(river_index)

        # If we have no rivers within the domain, just set everything to empty lists.
        if self.dims.river == 0:
            self.river.names = []
            for var in ('flux', 'salinity', 'temperature'):
                setattr(self.river, var, [])
            if ersem:
                for var in ersem:
                    setattr(self.river, var, [])
                # Do the extras too.
                for var in ('Z4_c', 'Z5_c', 'Z5_n', 'Z5_p', 'Z6_c', 'Z6_n', 'Z6_p'):
                    setattr(self.river, var, [])
        else:
            self.river.names = [names[i] for i in river_index]
            setattr(self.river, 'flux', flux[:, river_index])
            setattr(self.river, 'salinity', salinity[:, river_index])
            setattr(self.river, 'temperature', temperature[:, river_index])

            if ersem:
                for variable in ersem:
                    setattr(self.river, variable, ersem[variable][:, river_index])

                # Add small zooplankton values if we haven't been given any already. Taken to be 10^-6 of Western
                # Channel Observatory L4 initial conditions.
                fac = 10**-6
                extra_data = {'Z4_c': 1.2 * fac,
                              'Z5_c': 7.2 * fac,
                              'Z5_n': 0.12 * fac,
                              'Z5_p': 0.0113 * fac,
                              'Z6_c': 2.4 * fac,
                              'Z6_n': 0.0505 * fac,
                              'Z6_p': 0.0047 * fac}
                for extra in extra_data:
                    if not hasattr(self.river, extra):
                        setattr(self.river, extra, extra_data[extra])
                
            if sediments:
                for variable in sediments:
                    setattr(self.river, variable, sediments[variable][:, river_index])
                
    def check_rivers(self, max_discharge=None, min_depth=None, open_boundary_proximity=None, noisy=False):
        """
        Check the river nodes are suitable for an FVCOM run. By default, this only checks for rivers attached to
        elements which are bound on two sides by coastline.

        Parameters
        ----------
        max_discharge : float, optional
            Set a maximum discharge (in m^3s^{-1}) to supply to a single river node. This is useful for reducing the
            likelihood of crashes due to massive influxes of freshwater into a relatively small element.
        min_depth : float, optional
            Set a minimum depth (in metres) for river nodes. Shallower river nodes are set to this minimum depth.
        open_boundary_proximity : float, optional
            Remove rivers within some radius (in kilometres) of an open boundary node.

        """
        self.river.bad_nodes = []

        # Do nothing here if we have no rivers.
        if self.dims.river == 0:
            return

        if max_discharge:
            # Find rivers in excess of the given discharge maximum.
            big_rivers = np.unique(np.argwhere(self.river.flux > max_discharge)[:,1])
            if np.any(big_rivers):
                for this_river in big_rivers:
                    no_of_splits = np.ceil(np.max(self.river.flux[:,this_river])/max_discharge)
                    print('River {} split into {}'.format(this_river, no_of_splits))
                    original_river_name = self.river.names[this_river]
                    each_flux = self.river.flux[:,this_river]/no_of_splits # everything else is concentrations so can just be copied

                    for this_i in np.arange(2,no_of_splits +1):
                        self.river.names.append('{}_{:d}'.format(original_river_name, int(this_i)))        
                    
                    self.river.flux[:, this_river] = each_flux # everything else is concentrations so can just be copied
                    
                    # Collect all variables to add columns for
                    all_vars = ['flux', 'temperature', 'salinity']
    
                    # Ersem variables if they're in there
                    N_names = list(filter(lambda x: 'N' in x, list(self.river.__dict__.keys())))
                    Z_names = list(filter(lambda x: 'Z' in x, list(self.river.__dict__.keys())))
                    O_names = list(filter(lambda x: 'O' in x, list(self.river.__dict__.keys())))

                    # And sediment ones
                    muddy_sediment_names = list(filter(lambda x: 'mud_' in x, list(self.river.__dict__.keys())))
                    sandy_sediment_names = list(filter(lambda x: 'sand_' in x, list(self.river.__dict__.keys())))

                    all_vars = flatten_list([all_vars, N_names, Z_names, O_names, muddy_sediment_names, sandy_sediment_names]) 

                    for this_var in all_vars:
                        self._add_river_col(this_var, this_river, no_of_splits -1)

                    original_river_node = self.river.node[this_river]
                    for this_new_riv in np.arange(1, no_of_splits):    
                        self.river.node.append(self._find_near_free_node(original_river_node))
                    print('Flux array shape {} x {}'.format(self.river.flux.shape[0], self.river.flux.shape[1]))
                    print('Node list length {}'.format(len(self.river.node)))


        # Move rivers in bad nodes
        for i, node in enumerate(self.river.node):
            bad = find_bad_node(self.grid.triangles, node)
            if bad:
                self.river.node[i] = self._find_near_free_node(node)
                        
        if min_depth:
            shallow_rivers = np.argwhere(self.grid.h[self.river.node] < min_depth)
                      
            for this_shallow_node in self.grid.coastline[self.grid.h[self.grid.coastline] < min_depth]:
                self.river.bad_nodes.append(this_shallow_node)
            if np.any(shallow_rivers):
                for this_river in shallow_rivers:
                    self.river.node[this_river[0]] = self._find_near_free_node(self.river.node[this_river[0]])

        if open_boundary_proximity:
            # Remove nodes close to the open boundary joint with the coastline. Identifying the coastline/open
            # boundary joining nodes is simply a case of taking the first and last node ID for each open boundary.
            # Using that position, we can find any river nodes which fall within that distance and simply remove
            # their data from the relevant self.river arrays.
            boundary_river_indices = []
            grid_pts = np.asarray([self.grid.lon[self.river.node], self.grid.lat[self.river.node]]).T
            obc_ll = np.asarray([self.grid.lon[self.grid.open_boundary_nodes], self.grid.lat[self.grid.open_boundary_nodes]])
            dist = np.min(np.asarray([haversine_distance(obc_ll, this_riv_ll) for this_riv_ll in grid_pts]), axis=1) 
            breached_distance = dist < open_boundary_proximity
            
            to_remove = np.sum(breached_distance)
            if np.any(breached_distance):
                if noisy:
                    extra = ''
                    if to_remove > 1:
                        extra = 's'
                    print('Removing {} river{}'.format(to_remove, extra))

            boundary_river_indices = np.argwhere(breached_distance).tolist()
            # Now drop all those indices from the relevant river data.
            for field in self.obj_iter(self.river):
                if field not in ['time']:
                    setattr(self.river, field, np.delete(getattr(self.river, field), flatten_list(boundary_river_indices), axis=-1))

        # Update the dimension
        self.dims.river = len(self.river.node)

    def _add_river_col(self, var_name, col_to_copy, no_cols_to_add):
        """
        Helper function to copy the existing data for river variable to a new splinter river (when they are split for
        excessive discharge at one node

        Parameters
        ----------
        var_name : str
            Name of river attribute to alter
        col_to_copy : int
            The column (i.e. river) to copy from)
        no_cols_to_add : int
            The number of columns (i.e. extra rivers) to add to the end of the array
        """
        old_data = getattr(self.river, var_name)
        col_to_add = old_data[:, col_to_copy][:, np.newaxis]
        col_to_add = np.tile(col_to_add, [1, int(no_cols_to_add)])
        setattr(self.river, var_name, np.hstack([old_data, col_to_add]))

    def _find_near_free_node(self, start_node):
        if find_bad_node(self.grid.triangles, start_node) and ~np.any(np.isin(self.river.bad_nodes, start_node)):
            self.river.bad_nodes.append(start_node)
        elif not np.any(np.isin(self.river.node, start_node)):
            return start_node # start node is already free for use

        possible_nodes = []
        start_nodes = np.asarray([start_node])
        nodes_checked = start_nodes

        while len(possible_nodes) == 0:
            start_next = []
            for this_node in start_nodes:
                attached_nodes = self.grid.coastline[np.isin(self.grid.coastline, 
                        self.grid.triangles[np.any(np.isin(self.grid.triangles, this_node), axis=1),:].flatten())]
                attached_nodes = np.delete(attached_nodes, np.where(np.isin(attached_nodes,nodes_checked)))
                for this_candidate in attached_nodes:
                    if not np.any(np.isin(self.river.bad_nodes, this_candidate)) and not np.any(np.isin(self.river.node, this_candidate)):
                        if find_bad_node(self.grid.triangles, this_candidate):
                            self.river.bad_nodes.append(this_candidate)
                        else:
                            possible_nodes.append(this_candidate)
                start_next.append(attached_nodes)
            start_next = [i for sub_list in start_next for i in sub_list]

            nodes_checked = np.hstack([nodes_checked, start_nodes])
            start_nodes = np.unique(np.asarray(start_next).flatten())

        # if more than one possible node choose the closest
        if len(possible_nodes) > 1:        
            start_node_ll = [self.grid.lon[start_node], self.grid.lat[start_node]]
            possible_nodes_ll = [self.grid.lon[np.asarray(possible_nodes)], self.grid.lat[np.asarray(possible_nodes)]]
            dist = np.asarray([haversine_distance(pt_1, start_node_ll) for pt_1 in possible_nodes_ll])
            return possible_nodes[dist.argmin()]
        else:
            return possible_nodes[0]
    
    
    def write_river_forcing(self, output_file, ersem=False, ncopts={'zlib': True, 'complevel': 7}, sediments=False, **kwargs):
        """
        Write out an FVCOM river forcing netCDF file.

        Parameters
        ----------
        output_file : str, pathlib.Path
            File to which to write river forcing data.
        ersem : bool
            Set to True to add the ERSEM variables. Corresponding data must exist in self.rivers.
        ncopts : dict, optional
            Dictionary of options to use when creating the netCDF variables. Defaults to compression on.

        The self.river object should have, at a minimum, the following objects:
            - names : list of river names
            - nodes : list of grid nodes
            - time : list of datetimes
            - flux : river discharge data [time, river]
            - temperature : river temperature data [time, river]
            - salinity : river salinity data [time, river]
        If using ERSEM, then it should also contain:
            - N1_p : phosphate [time, river]
            - N3_n : nitrate [time, river]
            - N4_n : ammonium [time, river]
            - N5_s : silicate [time, river]
            - O2_o : oxygen [time, river]
            - O3_TA : total alkalinity [time, river]
            - O3_c : dissolved inorganic carbon [time, river]
            - O3_bioalk : bio-alkalinity [time, river]
            - Z4_c : mesozooplankton carbon [time, river]
        If using sediments then any objects of the self.river whose name matches 'mud_*' or 'sand_*' will be added
        to the output.

        Uses self.river.source for the 'title' global attribute in the netCDF and self.river.history for the 'info'
        global attribute. Both of these default to empty strings.

        Remaining arguments are passed to WriteForcing.

        """

        output_file = str(output_file)  # in case we've been given a pathlib.Path

        globals = {'type': 'FVCOM RIVER FORCING FILE',
                   'title': self.river.source,
                   'info': self.river.history,
                   'history': 'File created using {} from PyFVCOM'.format(inspect.stack()[0][3])}
        dims = {'namelen': 80, 'rivers': self.dims.river, 'time': 0, 'DateStrLen': 26}
        with WriteForcing(str(output_file), dims, global_attributes=globals, clobber=True, format='NETCDF4', **kwargs) as river:
            # We need to force the river names to be right-padded to 80 characters and transposed for the netCDF array.
            river_names = stringtochar(np.asarray(self.river.names, dtype='S80')) 
            river.add_variable('river_names', river_names, ['rivers', 'namelen'], format='c', ncopts=ncopts)

            river.write_fvcom_time(self.river.time, ncopts=ncopts)

            atts = {'long_name': 'river runoff volume flux', 'units': 'm^3s^-1'}
            river.add_variable('river_flux', self.river.flux, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'river runoff temperature', 'units': 'Celsius'}
            river.add_variable('river_temp', self.river.temperature, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

            atts = {'units': 'PSU'}
            river.add_variable('river_salt', self.river.salinity, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

            if ersem:
                atts = {'long_name': 'phosphate phosphorus', 'units': 'mmol P/m^3'}
                river.add_variable('N1_p', self.river.N1_p, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

                atts = {'long_name': 'nitrate nitrogen', 'units': 'mmol N/m^3'}
                river.add_variable('N3_n', self.river.N3_n, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

                atts = {'long_name': 'ammonium nitrogen', 'units': 'mmol N/m^3'}
                river.add_variable('N4_n', self.river.N4_n, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

                atts = {'long_name': 'silicate silicate', 'units': 'mmol Si/m^3'}
                river.add_variable('N5_s', self.river.N5_s, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

                atts = {'long_name': 'dissolved Oxygen', 'units': 'mmol O_2/m^3'}
                river.add_variable('O2_o', self.river.O2_o, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

                atts = {'long_name': 'carbonate total alkalinity', 'units': 'mmol C/m^3'}
                river.add_variable('O3_TA', self.river.O3_TA, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

                atts = {'long_name': 'carbonate total dissolved inorganic carbon', 'units': 'mmol C/m^3'}
                river.add_variable('O3_c', self.river.O3_c, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

                atts = {'long_name': 'carbonate bioalkalinity', 'units': 'umol/kg'}
                river.add_variable('O3_bioalk', self.river.O3_bioalk, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

                atts = {'long_name': 'mesozooplankton carbon', 'units': 'mg C/m^3'}
                river.add_variable('Z4_c', self.river.Z4_c, ['time', 'rivers'], attributes=atts, ncopts=ncopts)

                # Additional zooplankton variables.
                zooplankton_prefixes = ['Z5', 'Z6']
                zooplankton_suffixes = ['n', 'c', 'p']
                zooplankton_long_names = ['microzooplankton', 'nanoflagellates']
                nutrient_long_names = ['nitrogen', 'phosphorus', 'nitrogen']
                nutrient_units = {'mmol N/m^3', 'mmol P/m^3', 'mg C/m^3'}

                # Make the new variable names and add accordingly, but only if we don't already have them in the file.
                for prefix, zooplankton_name in zip(zooplankton_prefixes, zooplankton_long_names):
                    for suffix, nutrient_name, units in zip(zooplankton_suffixes, nutrient_long_names, nutrient_units):
                        if '{} {}'.format(zooplankton_name, nutrient_name) not in river.nc.variables:
                            atts = {'long_name': '{} {}'.format(zooplankton_name, nutrient_name),
                                    'units': units}
                            river.add_variable('{}_{}'.format(prefix, suffix),
                                               getattr(self.river, '{}_{}'.format(prefix, suffix)),
                                               ['time', 'rivers'],
                                               attributes=atts,
                                               ncopts=ncopts)
        
            if sediments:
                muddy_sediment_names = list(filter(lambda x:'mud_' in x, list(self.river.__dict__.keys())))
                sandy_sediment_names = list(filter(lambda x:'sand_' in x, list(self.river.__dict__.keys())))

                if muddy_sediment_names:
                    for this_sediment in muddy_sediment_names:
                        atts = {'long_name': '{} - muddy stuff'.format(this_sediment), 'units': 'kgm^-3'} 
                        river.add_variable(this_sediment, getattr(self.river, this_sediment), ['time', 'rivers'],
                                           attributes=atts, ncopts=ncopts)
    
                if sandy_sediment_names:
                    for this_sediment in sandy_sediment_names:
                        atts = {'long_name': '{} - sandy stuff'.format(this_sediment), 'units': 'kgm^-3'}    
                        river.add_variable(this_sediment, getattr(self.river, this_sediment), ['time', 'rivers'],
                                           attributes=atts, ncopts=ncopts)

    def write_river_namelist(self, output_file, forcing_file, vertical_distribution='uniform'):
        """
        Write an FVCOM river namelist file.

        Parameters
        ----------
        output_file : str, pathlib.Path
            Output file to which to write the river configuration.
        forcing_file : str, pathlib.Path
            File from which FVCOM will read the river forcing data.
        vertical_distribution : str, optional
            Vertical distribution of river input. Defaults to 'uniform'.

        """
        with open(output_file, 'w') as f:
            for ri in range(self.dims.river):
                f.write(' &NML_RIVER\n')
                f.write('  RIVER_NAME          = ''{}'',\n'.format(self.river.names[ri]))
                f.write('  RIVER_FILE          = ''{}'',\n'.format(forcing_file))
                f.write('  RIVER_GRID_LOCATION = {:d},\n'.format(self.river.node[ri] + 1))
                f.write('  RIVER_VERTICAL_DISTRIBUTION = {}\n'.format(vertical_distribution))
                f.write('  /\n')

    def read_nemo_rivers(self, nemo_file, remove_baltic=True):
        """
        Read a NEMO river netCDF file.

        Parameters
        ----------
        nemo_file : str, pathlib.Path
            Path to the NEMO forcing file.
        remove_baltic : bool, optional
            Remove the 'Baltic' rivers. These are included in the NEMO forcing since there is no open boundary for
            the Baltic; instead, the Baltic is represented as two river inputs. This messes up all sorts of things
            generally, so the default for this option is to remove them. Set to False to keep them.

        Returns
        -------
        nemo: dict
            A dictionary with the following keys:

            positions : np.ndarray
                NEMO river locations.
            times : np.ndarray
                NEMO river time series. Since the NEMO data is a climatology, this uses the self.start and self.end
                variables to create a matching time series for the river data.
            names : np.ndarray
                NEMO river names.
            flux : np.ndarray
                NEMO river discharge (m^3s^{-1}) [time, river]
            temperature : np.ndarray
                NEMO river temperature (degrees Celsius) [time, river]
            N4_n : np.ndarray
                NEMO river ammonia (mmol/m^3) [time, river]
            N3_n : np.ndarray
                NEMO river nitrate (mmol/m^3) [time, river]
            O2_o : np.ndarray
                NEMO river oxygen (mmol/m^3) [time, river]
            N1_p : np.ndarray
                NEMO river phosphate (mmol/m^3) [time, river]
            N5_s : np.ndarray
                NEMO river silicate (mmol/m^3) [time, river]
            O3_c : np.ndarray
                NEMO river dissolved inorganic carbon (mmol/m^3) [time, river]
            O3_TA : np.ndarray
                NEMO river total alkalinity (mmol/m^3) [time, river]
            O3_bioalk : np.ndarray
                NEMO river bio-alkalinity (umol/m^3 - note different units) [time, river]

        Notes
        -----
        This is mostly copy-pasted from the MATLAB fvcom-toolbox function get_NEMO_rivers.m.

        """

        baltic_lon = [10.7777, 12.5555]
        baltic_lat = [55.5998, 56.1331]

        nemo_variables = ['rodic', 'ronh4', 'rono3', 'roo', 'rop', 'rorunoff', 'rosio2',
                          'rotemper', 'rototalk', 'robioalk']
        sensible_names = ['O3_c', 'N4_n', 'N3_n', 'O2_o', 'N1_p', 'flux', 'N5_s',
                          'temperature', 'O3_TA', 'O3_bioalk']

        nemo = {}
        # NEMO river data are stored ['time', 'y', 'x'].
        with Dataset(nemo_file, 'r') as nc:
            number_of_times = nc.dimensions['time_counter'].size
            nemo['times'] = np.linspace(0, number_of_times, number_of_times + 1, endpoint=True)
            nemo['times'] = [self.start + relativedelta(days=i) for i in nemo['times']]
            nemo['lon'], nemo['lat'] = np.meshgrid(nc.variables['x'][:], nc.variables['y'][:])
            if remove_baltic:
                # Find the indices of the 'Baltic' rivers and drop them from everything we load.
                baltic_indices = []
                for baltic in zip(baltic_lon, baltic_lat):
                    x_index = np.argmin(np.abs(nc.variables['x'][:] - baltic[0]))
                    y_index = np.argmin(np.abs(nc.variables['y'][:] - baltic[1]))
                    baltic_indices.append((y_index, x_index))  # make the indices match the dimensions in the netCDF arrays

            for vi, var in enumerate(nemo_variables):
                nemo[sensible_names[vi]] = nc.variables[var][:]
                if remove_baltic:
                    for baltic_index in baltic_indices:
                        # Replace with zeros to match the other non-river data in the netCDF. Dimensions of the arrays are
                        # [time, y, x].
                        nemo[sensible_names[vi]][:, baltic_index[0], baltic_index[1]] = 0

            # Get the NEMO grid area for correcting units.
            area = nc.variables['dA'][:]

        # Flux in NEMO is specified in kg/m^{2}/s. FVCOM wants m^{3}/s. Divide by freshwater density to get m/s and
        # then multiply by the area of each element to get flux.
        nemo['flux'] /= 1000
        # Now multiply by the relevant area to (finally!) get to m^{3}/s.
        nemo['flux'] *= area
        # Set zero values to a very small number instead to avoid divide by zero errors below.
        temporary_flux = nemo['flux']
        temporary_flux[temporary_flux == 0] = 1e-8
        # Convert units from grams to millimoles where appropriate.
        nemo['N4_n'] = (nemo['N4_n'] / 14) * 1000 / temporary_flux  # g/s to mmol/m3
        nemo['N3_n'] = (nemo['N3_n'] / 14) * 1000 / temporary_flux  # g/s to mmol/m3
        nemo['O2_o'] = (nemo['O2_o'] / 16) * 1000 / temporary_flux  # Nemo oxygen concentrations are for O rather than O2
        nemo['N1_p'] = (nemo['N1_p'] / 35.5) * 1000 / temporary_flux  # g/s to mmol/m3
        nemo['N5_s'] = (nemo['N5_s'] / 28) * 1000 / temporary_flux  # g/s to mmol/m3
        nemo['O3_bioalk'] = nemo['O3_bioalk'] / temporary_flux / 1000  # bioalk is in umol/s need umol/kg
        nemo['O3_c'] = nemo['O3_c'] / 12 / temporary_flux * 1000  # dic is in gC/s need mmol/m3
        # Total alkalinity is already in umol/kg as expected by ERSEM.

        # Now we've got the data, use the flux data to find the indices of the rivers in the arrays and extract those
        # as time series per location. These data can then be passed to self.add_rivers fairly straightforwardly.
        mask = np.any(nemo['flux'].data, axis=0)
        for key in nemo:
            if key != 'times':
                try:
                    # Make the array time dimension appear first for compatibility with self.add_rivers. That pair of
                    # transposes are probably less than ideal, but I want to go home now.
                    nemo[key] = nemo[key][:, mask].T.reshape(-1, number_of_times).T
                except IndexError:
                    nemo[key] = nemo[key][mask]
        # Since the NEMO river don't have names, make some based on their position.
        nemo['names'] = ['river_{}_{}'.format(*i) for i in zip(nemo['lon'], nemo['lat'])]

        return nemo

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
                    sigma = [1, self.dims.layers]
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

    def add_nests(self, nest_levels, nesting_type=3):
        """
        Add a set of nested levels to each open boundary.

        Parameters
        ----------
        nest_levels : int
            Number of node levels in addition to the existing open boundary.
        nesting_type : int
            FVCOM nesting type (1, 2 or 3). Defaults to 3. Currently, only 3 is supported.

        Provides
        --------
        self.nests : list
            List of PyFVCOM.preproc.Nest objects.

        """

        self.nest = []

        for boundary in self.open_boundaries:
            self.nest.append(Nest(self.grid, self.sigma, boundary))
            # Add all the nested levels and assign weights as necessary.
            for _ in range(nest_levels):
                self.nest[-1].add_level()
            if nesting_type == 3:
                self.nest[-1].add_weights()

    def write_nested_forcing(self, nests, ncfile, type=3, **kwargs):
        """
        Write out the given nested forcing into the specified netCDF file.

        Parameters
        ----------
        nests : list
            List of PyFVCOM.grid.Nest objects.
        ncfile : str, pathlib.Path
            Path to the output netCDF file to created.
        type : int, optional
            Type of model nesting. Currently only type 3 (indirect) is supported. Defaults to 3.

        Remaining kwargs are passed to WriteForcing with the exception of ncopts which is passed to
        WriteForcing.add_variable.

        """

        # Get all the nodes, elements and weights ready for dumping to netCDF.
        nodes = flatten_list([boundary.nodes for nest in nests for boundary in nest.boundaries])
        elements = flatten_list([boundary.elements for nest in nests for boundary in nest.boundaries if np.any(boundary.elements)])
        if type == 3:
            weight_nodes = flatten_list([boundary.weight_node for nest in nests for boundary in nest.boundaries])
            weight_elements = flatten_list([boundary.weight_element for nest in nests for boundary in nest.boundaries if np.any(boundary.elements)])

        # Get all the interpolated data too. We need to concatenate in the same order as we've done above, so just be
        # careful.
        time_number = len(self.time.datetime)
        nodes_number = len(nodes)
        elements_number = len(elements)

        # Prepare the data.
        zeta = np.empty((time_number, nodes_number)) * np.nan
        ua = np.empty((time_number, elements_number)) * np.nan
        va = np.empty((time_number, elements_number)) * np.nan
        u = np.empty((time_number, self.dims.layers, elements_number)) * np.nan
        v = np.empty((time_number, self.dims.layers, elements_number)) * np.nan
        temperature = np.empty((time_number, self.dims.layers, nodes_number)) * np.nan
        salinity = np.empty((time_number, self.dims.layers, nodes_number)) * np.nan
        hyw = np.zeros((time_number, self.dims.layers, nodes_number))  # we never set this to anything other than zeros
        weight_nodes = np.repeat(weight_nodes, time_number, 0).reshape(time_number, -1)
        weight_elements = np.repeat(weight_elements, time_number, 0).reshape(time_number, -1)

        ncopts = {}
        if 'ncopts' in kwargs:
            ncopts = kwargs['ncopts']
            kwargs.pop('ncopts')

        # Define the global attributes
        globals = {'type': 'FVCOM nestING TIME SERIES FILE',
                   'title': 'FVCOM nestING TYPE {} TIME SERIES data for open boundary'.format(type),
                   'history': 'File created using {} from PyFVCOM'.format(inspect.stack()[0][3]),
                   'filename': str(ncfile),
                   'Conventions': 'CF-1.0'}

        dims = {'nele': elements_number, 'node': nodes_number, 'time': 0, 'DateStrLen': 26, 'three': 3,
                'siglay': self.dims.layers, 'siglev': self.dims.levels}

        # Fix the triangulation for the nested region.
        # nv = reduce_triangulation(self.grid.triangles, nodes).T + 1  # offset by one for FORTRAN indexing and transpose

        with WriteForcing(str(ncfile), dims, global_attributes=globals, clobber=True, format='NETCDF4', **kwargs) as nest:
            # Add standard times.
            nest.write_fvcom_time(self.time.datetime, ncopts=ncopts)

            # Add space variables.
            if self.debug:
                print('adding x to netCDF')
            atts = {'units': 'meters', 'long_name': 'nodal x-coordinate'}
            nest.add_variable('x', self.grid.x[nodes], ['node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding y to netCDF')
            atts = {'units': 'meters', 'long_name': 'nodal y-coordinate'}
            nest.add_variable('y', self.grid.y[nodes], ['node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding lon to netCDF')
            atts = {'units': 'degrees_east', 'standard_name': 'longitude', 'long_name': 'nodal longitude'}
            nest.add_variable('lon', self.grid.lon[nodes], ['node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding lat to netCDF')
            atts = {'units': 'degrees_north', 'standard_name': 'latitude', 'long_name': 'nodal latitude'}
            nest.add_variable('lat', self.grid.lat[nodes], ['node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding xc to netCDF')
            atts = {'units': 'meters', 'long_name': 'zonal x-coordinate'}
            nest.add_variable('xc', self.grid.xc[elements], ['nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding yc to netCDF')
            atts = {'units': 'meters', 'long_name': 'zonal y-coordinate'}
            nest.add_variable('yc', self.grid.yc[elements], ['nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding lonc to netCDF')
            atts = {'units': 'degrees_east', 'standard_name': 'longitude', 'long_name': 'zonal longitude'}
            nest.add_variable('lonc', self.grid.lonc[elements], ['nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding latc to netCDF')
            atts = {'units': 'degrees_north', 'standard_name': 'latitude', 'long_name': 'zonal latitude'}
            nest.add_variable('latc', self.grid.latc[elements], ['nele'], attributes=atts, ncopts=ncopts)

            # No attributes for nv in the existing nest files, so I won't add any here.
            # nest.add_variable('nv', nv, ['three', 'nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding siglay to netCDF')
            atts = {'long_name': 'Sigma Layers',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglay eta: zeta depth: h'}
            nest.add_variable('siglay', self.sigma.layers[nodes, :].T, ['siglay', 'node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding siglev to netCDF')
            atts = {'long_name': 'Sigma Levels',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma:siglev eta: zeta depth: h'}
            nest.add_variable('siglev', self.sigma.levels[nodes, :].T, ['siglev', 'node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding siglay_center to netCDF')
            atts = {'long_name': 'Sigma Layers',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglay_center eta: zeta_center depth: h_center'}
            nest.add_variable('siglay_center', self.sigma.layers_center[elements, :].T, ['siglay', 'nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding siglev_center to netCDF')
            atts = {'long_name': 'Sigma Levels',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglev_center eta: zeta_center depth: h_center'}
            nest.add_variable('siglev_center', self.sigma.levels_center[elements, :].T, ['siglev', 'nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding h to netCDF')
            atts = {'long_name': 'Bathymetry',
                    'standard_name': 'sea_floor_depth_below_geoid',
                    'units': 'm',
                    'positive': 'down',
                    'grid': 'Bathymetry_mesh',
                    'coordinates': 'x y',
                    'type': 'data'}
            nest.add_variable('h', self.grid.h[nodes], ['node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding h_center to netCDF')
            atts = {'long_name': 'Bathymetry',
                    'standard_name': 'sea_floor_depth_below_geoid',
                    'units': 'm',
                    'positive': 'down',
                    'grid': 'grid1 grid3',
                    'coordinates': 'latc lonc',
                    'grid_location': 'center'}
            nest.add_variable('h_center', self.grid.h_center[elements], ['nele'], attributes=atts, ncopts=ncopts)

            if type == 3:
                if self.debug:
                    print('adding weight_node to netCDF')
                atts = {'long_name': 'Weights for nodes in relaxation zone',
                        'units': 'no units',
                        'grid': 'fvcom_grid',
                        'type': 'data'}
                nest.add_variable('weight_node', weight_nodes, ['time', 'node'], attributes=atts, ncopts=ncopts)

                if self.debug:
                    print('adding weight_cell to netCDF')
                atts = {'long_name': 'Weights for elements in relaxation zone',
                        'units': 'no units',
                        'grid': 'fvcom_grid',
                        'type': 'data'}
                nest.add_variable('weight_cell', weight_elements, ['time', 'nele'], attributes=atts, ncopts=ncopts)

            # Now all the data.
            if self.debug:
                print('adding zeta to netCDF')
            atts = {'long_name': 'Water Surface Elevation',
                    'units': 'meters',
                    'positive': 'up',
                    'standard_name': 'sea_surface_height_above_geoid',
                    'grid': 'Bathymetry_Mesh',
                    'coordinates': 'time lat lon',
                    'type': 'data',
                    'location': 'node'}
            nest.add_variable('zeta', zeta, ['time','node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding ua to netCDF')
            atts = {'long_name': 'Vertically Averaged x-velocity',
                    'units': 'meters  s-1',
                    'grid': 'fvcom_grid',
                    'type': 'data'}
            nest.add_variable('ua', ua, ['time', 'nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding va to netCDF')
            atts = {'long_name': 'Vertically Averaged y-velocity',
                    'units': 'meters  s-1',
                    'grid': 'fvcom_grid',
                    'type': 'data'}
            nest.add_variable('va', va, ['time', 'nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding u to netCDF')
            atts = {'long_name': 'Eastward Water Velocity',
                    'units': 'meters  s-1',
                    'standard_name': 'eastward_sea_water_velocity',
                    'grid': 'fvcom_grid',
                    'coordinates': 'time siglay latc lonc',
                    'type': 'data',
                    'location': 'face'}
            nest.add_variable('u', u, ['time', 'siglay', 'nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding v to netCDF')
            atts = {'long_name': 'Northward Water Velocity',
                    'units': 'meters  s-1',
                    'standard_name': 'Northward_sea_water_velocity',
                    'grid': 'fvcom_grid',
                    'coordinates': 'time siglay latc lonc',
                    'type': 'data',
                    'location': 'face'}
            nest.add_variable('v', v, ['time', 'siglay', 'nele'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding temp to netCDF')
            atts = {'long_name': 'Temperature',
                    'standard_name': 'sea_water_temperature',
                    'units': 'degrees Celcius',
                    'grid': 'fvcom_grid',
                    'coordinates': 'time siglay lat lon',
                    'type': 'data',
                    'location': 'node'}
            nest.add_variable('temp', temperature, ['time', 'siglay', 'node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding salinity to netCDF')
            atts = {'long_name': 'Salinity',
                    'standard_name': 'sea_water_salinity',
                    'units': '1e-3',
                    'grid': 'fvcom_grid',
                    'coordinates': 'time siglay lat lon',
                    'type': 'data',
                    'location': 'node'}
            nest.add_variable('salinity', salinity, ['time', 'siglay', 'node'], attributes=atts, ncopts=ncopts)

            if self.debug:
                print('adding hyw to netCDF')
            atts = {'long_name': 'hydro static vertical velocity',
                    'units': 'meters s-1',
                    'grid': 'fvcom_grid',
                    'type': 'data',
                    'coordinates': 'time siglay lat lon'}
            nest.add_variable('hyw', hyw, ['time', 'siglay', 'node'], attributes=atts, ncopts=ncopts)

    def read_regular(self, *args, **kwargs):
        read_regular.__doc__
        self.regular = read_regular(*args, noisy=self.noisy, **kwargs)


class Nest(object):
    """
    Class to hold a set of open boundaries as OpenBoundary objects.

    This feels like it ought to be a superclass of OpenBoundary, but I can't wrap my head around how.

    """

    def __init__(self, grid, sigma, boundary):
        """
        Create a nested boundary object.

        Parameters
        ----------
        grid : PyFVCOM.grid.Domain
            The model grid within which the nest will sit.
        sigma : PyFVCOM.model.OpenBoundary.sigma
            The vertical sigma coordinate configuration for the current grid.
        boundary : PyFVCOM.grid.OpenBoundary, list
            An open boundary or list of open boundaries with which to initialise this nest.

        """

        self.debug = False

        self.obj_iter = lambda x: [a for a in dir(x) if not a.startswith('__')]

        self.grid = copy.copy(grid)
        self.sigma = copy.copy(sigma)

        if isinstance(boundary, list):
            self.boundaries = boundary
        elif isinstance(boundary, OpenBoundary):
            self.boundaries = [boundary]
        else:
            raise ValueError("Unsupported boundary type {}. Supply PyFVCOM.grid.OpenBoundary or `list'.".format(type(boundary)))
        # Add the sigma and grid structure attributes.
        self.__update_open_boundaries()

    def __update_open_boundaries(self):
        """
        Call this when we've done something which affects the open boundary objects and we need to update their
        properties.

        For example, this updates sigma information if we've added the sigma distribution to the Model object.

        """

        # Add the grid and sigma data to any open boundaries we've got loaded.
        for ii, boundary in enumerate(self.boundaries):
            if self.debug:
                print('adding grid info to boundary {} of {}'.format(ii + 1, len(self.boundaries)))
            for attribute in self.obj_iter(self.grid):
                if self.debug:
                    print('\t{}'.format(attribute))
                try:
                    if 'center' not in attribute:
                        setattr(boundary.grid, attribute, getattr(self.grid, attribute)[boundary.nodes, ...])
                    else:
                        if np.any(boundary.elements):
                            setattr(boundary.grid, attribute, getattr(self.grid, attribute)[boundary.elements, ...])
                except (IndexError, TypeError):
                    setattr(boundary.grid, attribute, getattr(self.grid, attribute))
                except AttributeError as e:
                    if self.debug:
                        print(e)
                    pass

            if self.debug:
                print('adding sigma info to boundary {} of {}'.format(ii + 1, len(self.boundaries)))
            for attribute in self.obj_iter(self.sigma):
                if self.debug:
                    print('\t{}'.format(attribute))
                try:
                    if 'center' not in attribute:
                        setattr(boundary.sigma, attribute, getattr(self.sigma, attribute)[boundary.nodes, ...])
                    else:
                        if np.any(boundary.elements):
                            setattr(boundary.sigma, attribute, getattr(self.sigma, attribute)[boundary.elements, ...])
                except (IndexError, TypeError):
                    setattr(boundary.sigma, attribute, getattr(self.sigma, attribute))
                except AttributeError as e:
                    if self.debug:
                        print(e)

    def add_level(self):
        """
        Function to add a nested level which is connected to the existing nested nodes and elements.

        This is useful for generating nested inputs from other model inputs (e.g. a regularly gridded model) in
        conjunction with PyFVCOM.grid.OpenBoundary.add_nested_forcing().

        Provides
        --------
        Adds a new PyFVCOM.grid.OpenBoundary object in self.boundaries

        """

        # Find all the elements connected to the last set of open boundary nodes.
        if not np.any(self.boundaries[-1].nodes):
            raise ValueError('No open boundary nodes in the current open boundary. Please add some and try again.')

        level_elements = find_connected_elements(self.boundaries[-1].nodes, self.grid.triangles)
        # Find the nodes and elements in the existing nests.
        nest_nodes = flatten_list([i.nodes for i in self.boundaries])
        nest_elements = flatten_list([i.elements for i in self.boundaries if np.any(i.elements)])

        # Get the nodes connected to the elements we've extracted.
        level_nodes = np.unique(self.grid.triangles[level_elements, :])
        # Remove ones we already have in the nest.
        unique_nodes = np.setdiff1d(level_nodes, nest_nodes)
        # Create a new open boundary from those nodes.
        new_boundary = OpenBoundary(unique_nodes)

        # Add the elements unique to the current nest level too.
        unique_elements = np.setdiff1d(level_elements, nest_elements)
        new_boundary.elements = unique_elements.tolist()

        # Grab the time from the previous one.
        setattr(new_boundary, 'time', self.boundaries[-1].time)

        self.boundaries.append(new_boundary)

        # Populate the grid and sigma objects too.
        self.__update_open_boundaries()

    def add_weights(self, power=0):
        """
        For the open boundaries in self.boundaries, add a corresponding weight for the nodes and elements to each one.

        Parameters
        ----------
        power : float, optional
            Give an optional power with which weighting decreases with each successive nest. Defaults to 0 (i.e.
            linear).

        Provides
        --------
        Populates the self.boundaries open boundary objects with the relevant weight_node and weight_element arrays.

        """

        for index, boundary in enumerate(self.boundaries):
            if power == 0:
                weight = 1 / (index + 1)
            else:
                weight = 1 / ((index + 1)**power)

            boundary.weight_node = np.repeat(weight, len(boundary.nodes))
            # We will always have one fewer sets of elements as the nodes bound the elements.
            if not np.any(boundary.elements) and index > 0:
                raise ValueError('No elements defined in this nest. Adding weights requires elements.')
            elif np.any(boundary.elements):
                boundary.weight_element = np.repeat(weight, len(boundary.elements))

    def add_tpxo_tides(self, *args, **kwargs):
        OpenBoundary.__doc__
        for boundary in self.boundaries:
            boundary.add_tpxo_tides(*args, **kwargs)

    def add_nested_forcing(self, *args, **kwargs):
        OpenBoundary.__doc__
        for ii, boundary in enumerate(self.boundaries):
            if self.debug:
                print('adding nested forcing for boundary {} of {}'.format(ii + 1, len(self.boundaries)))
            boundary.add_nested_forcing(*args, **kwargs)


def read_regular(regular, variables, noisy=False, **kwargs):
    """
    Read regularly gridded model data and provides a RegularReader object which mimics a FileReader object.

    Parameters
    ----------
    regular : str, pathlib.Path
        Files to read.
    variables : list
        Variables to extract. Variables missing in the files raise an error.
    noisy : bool, optional
        Set to True to enable verbose output. Defaults to False.
    Remaining keyword arguments are passed to RegularReader.

    Returns
    -------
    regular_model : PyFVCOM.preproc.RegularReader
        A RegularReader object with the requested variables loaded.

    """

    if 'variables' not in kwargs:
        kwargs.update({'variables': variables})

    for ii, file in enumerate(regular):
        if noisy:
            print('Loading file {}'.format(file))
        if ii == 0:
            regular_model = RegularReader(str(file), **kwargs)
        else:
            regular_model += RegularReader(str(file), **kwargs)

    return regular_model


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

        if global_attributes:
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
        if attributes:
            for attribute in attributes:
                setattr(var, attribute, attributes[attribute])

        var[:] = data

        setattr(self, name, var)

    def write_fvcom_time(self, time, **kwargs):
        """
        Write the four standard FVCOM time variables (time, Times, Itime, Itime2) for the given time series.

        Parameters
        ----------
        time : np.ndarray, list, tuple
            Times as datetime objects.

        """

        mjd = date2num(time, units='days since 1858-11-17 00:00:00')
        Itime = np.floor(mjd)  # integer Modified Julian Days
        Itime2 = (mjd - Itime) * 24 * 60 * 60 * 1000  # milliseconds since midnight
        Times = [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in time]

        # time
        atts = {'units': 'days since 1858-11-17 00:00:00',
                'format': 'modified julian day (MJD)',
                'long_name': 'time',
                'time_zone': 'UTC'}
        self.add_variable('time', mjd, ['time'], attributes=atts, **kwargs)
        # Itime
        atts = {'units': 'days since 1858-11-17 00:00:00',
                'format': 'modified julian day (MJD)',
                'time_zone': 'UTC'}
        self.add_variable('Itime', Itime, ['time'], attributes=atts, format='i', **kwargs)
        # Itime2
        atts = {'units': 'msec since 00:00:00', 'time_zone': 'UTC'}
        self.add_variable('Itime2', Itime2, ['time'], attributes=atts, format='i', **kwargs)
        # Times
        atts = {'long_name': 'Calendar Date', 'format': 'String: Calendar Time', 'time_zone': 'UTC'}
        self.add_variable('Times', Times, ['time', 'DateStrLen'], format='c', attributes=atts, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Tidy up the netCDF file handle. """
        self.nc.close()


class RegularReader(FileReader):
    """
    Class to read in regularly gridded model output. This provides a similar interface to a PyFVCOM.read.FileReader
    object but with an extra spatial dimension. This is currently based on CMEMS model outputs (i.e. NEMO).

    Author(s)
    ---------
    Pierre Cazenave (Plymouth Marine Laboratory)

    Credits
    -------
    This code leverages ideas (and in some cases, code) from PySeidon (https://github.com/GrumpyNounours/PySeidon)
    and PyLag-tools (https://gitlab.em.pml.ac.uk/PyLag/PyLag-tools).

    """

    def __add__(self, other, debug=False):
        """
        This special method means we can stack two RegularReader objects in time through a simple addition (e.g. nemo1
        += nemo2)

        """

        # Check we've already got all the same data objects before we start.
        lon_compare = self.dims.lon == other.dims.lon
        lat_compare = self.dims.lat == other.dims.lat
        depth_compare = self.dims.depth == other.dims.depth
        time_compare = self.time.datetime[-1] <= other.time.datetime[0]
        data_compare = self.obj_iter(self.data) == self.obj_iter(other.data)
        old_data = self.obj_iter(self.data)
        new_data = self.obj_iter(other.data)
        if not lon_compare:
            raise ValueError('Horizontal longitude data are incompatible.')
        if not lat_compare:
            raise ValueError('Horizontal latitude data are incompatible.')
        if not depth_compare:
            raise ValueError('Vertical depth layers are incompatible.')
        if not time_compare:
            raise ValueError("Time periods are incompatible (`fvcom2' must be greater than or equal to `fvcom1')."
                             "`fvcom1' has end {} and `fvcom2' has start {}".format(self.time.datetime[-1],
                                                                                    other.time.datetime[0]))
        if not data_compare:
            raise ValueError('Loaded data sets for each RegularReader class must match.')
        if not (old_data == new_data) and (old_data or new_data):
            warn('Subsequent attempts to load data for this merged object will only load data from the first object. '
                 'Load data into each object before merging them.')

        # Copy ourselves to a new version for concatenation. self is the old so we get appended to by the new.
        idem = copy.copy(self)

        for var in self.obj_iter(idem.data):
            if 'time' in idem.ds.variables[var].dimensions:
                setattr(idem.data, var, np.ma.concatenate((getattr(idem.data, var), getattr(other.data, var))))
        for time in self.obj_iter(idem.time):
            setattr(idem.time, time, np.concatenate((getattr(idem.time, time), getattr(other.time, time))))

        # Remove duplicate times.
        time_indices = np.arange(len(idem.time.time))
        _, dupes = np.unique(idem.time.time, return_index=True)
        dupe_indices = np.setdiff1d(time_indices, dupes)
        time_mask = np.ones(time_indices.shape, dtype=bool)
        time_mask[dupe_indices] = False
        for var in self.obj_iter(idem.data):
            # Only delete things with a time dimension.
            if 'time' in idem.ds.variables[var].dimensions:
                # time_axis = idem.ds.variables[var].dimensions.index('time')
                setattr(idem.data, var, getattr(idem.data, var)[time_mask, ...])  # assume time is first
                # setattr(idem.data, var, np.delete(getattr(idem.data, var), dupe_indices, axis=time_axis))
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

    def _load_time(self):
        """
        Populate a time object with additional useful time representations from the netCDF time data.
        """

        time = self.ds.variables['time'][:]

        # Make other time representations.
        self.time.datetime = num2date(time, units=getattr(self.ds.variables['time'], 'units'))
        if isinstance(self.time.datetime, (list, tuple, np.ndarray)):
            setattr(self.time, 'Times', np.array([datetime.strftime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in self.time.datetime]))
        else:
            setattr(self.time, 'Times', datetime.strftime(self.time.datetime, '%Y-%m-%dT%H:%M:%S.%f'))
        self.time.time = date2num(self.time.datetime, units='days since 1858-11-17 00:00:00')
        self.time.Itime = np.floor(self.time.time)
        self.time.Itime2 = (self.time.time - np.floor(self.time.time)) * 1000 * 60 * 60  # microseconds since midnight
        self.time.datetime = self.time.datetime
        self.time.matlabtime = self.time.time + 678942.0  # convert to MATLAB-indexed times from Modified Julian Date.

    def _load_grid(self):
        """
        Load the grid data.

        Convert from UTM to spherical if we haven't got those data in the existing output file.

        """

        grid_variables = ['lon', 'lat', 'x', 'y', 'depth']

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
                setattr(self.grid, grid, np.zeros((self.dims.lon, self.dims.lat)))
            except ValueError as value_error_message:
                warn('Variable {} has a problem with the data. Setting value as all zeros.'.format(grid))
                print(value_error_message)
                setattr(self.grid, grid, np.zeros(self.ds.variables[grid].shape))

        # Make the grid data the right shape for us to assume it's an FVCOM-style data set.
        # self.grid.lon, self.grid.lat = np.meshgrid(self.grid.lon, self.grid.lat)
        # self.grid.lon, self.grid.lat = self.grid.lon.ravel(), self.grid.lat.ravel()

        # Update dimensions to match those we've been given, if any. Omit time here as we shouldn't be touching that
        # dimension for any variable in use in here.
        for dim in self._dims:
            if dim != 'time':
                setattr(self.dims, dim, len(self._dims[dim]))

        # Convert the given W/E/S/N coordinates into node and element IDs to subset.
        if self._bounding_box:
            # We need to use the original Dataset lon and lat values here as they have the right shape for the
            # subsetting.
            self._dims['lon'] = np.argwhere((self.ds.variables['lon'][:] > self._dims['wesn'][0]) &
                                            (self.ds.variables['lon'][:] < self._dims['wesn'][1]))
            self._dims['lat'] = np.argwhere((self.ds.variables['lat'][:] > self._dims['wesn'][2]) &
                                            (self.ds.variables['lat'][:] < self._dims['wesn'][3]))

        related_variables = {'lon': ('x', 'lon'), 'lat': ('y', 'lat')}
        for spatial_dimension in 'lon', 'lat':
            if spatial_dimension in self._dims:
                setattr(self.dims, spatial_dimension, len(self._dims[spatial_dimension]))
                for var in related_variables[spatial_dimension]:
                    try:
                        spatial_index = self.ds.variables[var].dimensions.index(spatial_dimension)
                        var_shape = [i for i in np.shape(self.ds.variables[var])]
                        var_shape[spatial_index] = getattr(self.dims, spatial_dimension)
                        if 'depth' in (self._dims, self.ds.variables[var].dimensions):
                            var_shape[self.ds.variables[var].dimensions.index('depth')] = self.dims.siglay
                        _temp = np.empty(var_shape) * np.nan
                        if 'depth' in self.ds.variables[var].dimensions:
                            for ni, node in enumerate(self._dims[spatial_dimension]):
                                if 'depth' in self._dims:
                                    _temp[..., ni] = self.ds.variables[var][self._dims['depth'], node]
                                else:
                                    _temp[..., ni] = self.ds.variables[var][:, node]
                        else:
                            for ni, node in enumerate(self._dims[spatial_dimension]):
                                _temp[..., ni] = self.ds.variables[var][node]
                    except KeyError:
                        if 'depth' in var:
                            _temp = np.empty((self.dims.depth, getattr(self.dims, spatial_dimension)))
                        else:
                            _temp = np.empty(getattr(self.dims, spatial_dimension))
                setattr(self.grid, var, _temp)

        # Check if we've been given vertical dimensions to subset in too, and if so, do that. Check we haven't
        # already done this if the 'node' and 'nele' sections above first.
        for var in ['depth']:
            short_dim = copy.copy(var)
            # Assume we need to subset this one unless 'node' or 'nele' are missing from self._dims. If they're in
            # self._dims, we've already subsetted in the 'node' and 'nele' sections above, so doing it again here
            # would fail.
            subset_variable = True
            if 'lon' in self._dims or 'lat' in self._dims:
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
        self.grid.x_range = np.ptp(self.grid.x)
        self.grid.y_range = np.ptp(self.grid.y)

        # Only do the conversions when we have more than a single point since the relevant ranges will be zero with
        # only one position.
        if self.dims.lon > 1 and self.dims.lat > 1:
            if self.grid.lon_range == 0 and self.grid.lat_range == 0:
                self.grid.lon, self.grid.lat = lonlat_from_utm(self.grid.x, self.grid.y, zone=self._zone)
                self.grid.lon_range = np.ptp(self.grid.lon)
                self.grid.lat_range = np.ptp(self.grid.lat)
            if self.grid.lon_range == 0 and self.grid.lat_range == 0:
                self.grid.x, self.grid.y = utm_from_lonlat(self.grid.lon, self.grid.lat)
                self.grid.x_range = np.ptp(self.grid.x)
                self.grid.y_range = np.ptp(self.grid.y)

        # Make a bounding box variable too (spherical coordinates): W/E/S/N
        self.grid.bounding_box = (np.min(self.grid.lon), np.max(self.grid.lon),
                                  np.min(self.grid.lat), np.max(self.grid.lat))

    def load_data(self, var):
        """
        Load the given variable/variables.

        Parameters
        ----------
        var : list-like, str
            List of variables to load.

        """

        # Check if we've got iterable variables and make one if not.
        try:
            _ = (e for e in var)
        except TypeError:
            var = [var]

        for v in var:
            if v not in self.ds.variables:
                raise KeyError("Variable '{}' not present in {}.".format(v, self._fvcom))

            # Get this variable's dimensions
            var_dim = self.ds.variables[v].dimensions
            variable_shape = self.ds.variables[v].shape
            variable_indices = [np.arange(i) for i in variable_shape]
            for dimension in var_dim:
                if dimension in self._dims:
                    # Replace their size with anything we've been given in dims.
                    variable_index = var_dim.index(dimension)
                    variable_indices[variable_index] = self._dims[dimension]

            # Check the data we're loading is the same shape as our existing dimensions.
            lon_compare = self.ds.dimensions['lon'].size == self.dims.lon
            lat_compare = self.ds.dimensions['lat'].size == self.dims.lat
            depth_compare = self.ds.dimensions['depth'].size == self.dims.depth
            time_compare = self.ds.dimensions['time'].size == self.dims.time
            # Check again if we've been asked to subset in any dimension.
            if 'lon' in self._dims:
                lon_compare = len(self.ds.variables['lon'][self._dims['lon']]) == self.dims.lon
            if 'lat' in self._dims:
                lat_compare = len(self.ds.variables['lat'][self._dims['lat']]) == self.dims.lat
            if 'depth' in self._dims:
                depth_compare = len(self.ds.variables['depth'][self._dims['depth']]) == self.dims.depth
            if 'time' in self._dims:
                time_compare = len(self.ds.variables['time'][self._dims['time']]) == self.dims.time

            if not lon_compare:
                raise ValueError('Longitude data are incompatible. You may be trying to load data after having already '
                                 'concatenated a RegularReader object, which is unsupported.')
            if not lat_compare:
                raise ValueError('Latitude data are incompatible. You may be trying to load data after having already '
                                 'concatenated a RegularReader object, which is unsupported.')
            if not depth_compare:
                raise ValueError('Vertical depth layers are incompatible. You may be trying to load data after having '
                                 'already concatenated a RegularReader object, which is unsupported.')
            if not time_compare:
                raise ValueError('Time period is incompatible. You may be trying to load data after having already '
                                 'concatenated a RegularReader object, which is unsupported.')

            if 'time' not in var_dim:
                # Should we error here or carry on having warned?
                warn("{} does not contain a `time' dimension.".format(v))

            attributes = type('attributes', (object,), {})()
            for attribute in self.ds.variables[v].ncattrs():
                setattr(attributes, attribute, getattr(self.ds.variables[v], attribute))
            setattr(self.atts, v, attributes)

            data = self.ds.variables[v][variable_indices]  # data are automatically masked
            setattr(self.data, v, data)

    def closest_element(self, *args, **kwargs):
        """ Compatibility function. """
        return self.closest_node(*args, **kwargs)

    def closest_node(self, where, cartesian=False, threshold=np.inf, vincenty=False, haversine=False):
        if cartesian:
            raise ValueError('No cartesian coordinates defined')
        else:
            lat_rav, lon_rav = np.meshgrid(self.grid.lat, self.grid.lon)
            x, y = lon_rav.ravel(), lat_rav.ravel()

        index = self._closest_point(x, y, x, y, where, threshold=threshold, vincenty=vincenty, haversine=haversine)
        if len(index) == 1:
            index = index[0]
        return np.unravel_index(index, (len(self.grid.lon), len(self.grid.lat)))


class HYCOMReader(RegularReader):
    """
    Class for reading HYCOM data.

    """

    def __add__(self, other, debug=False):
        """
        This special method means we can stack two RegularReader objects in time through a simple addition (e.g. nemo1
        += nemo2)

        """

        # This is only subtly different from the one in RegularReader, but since the dimensions associated with each
        # variable differ in name, we have to adjust the code here. This is bound to introduce bugs eventually.

        # Check we've already got all the same data objects before we start.
        lon_compare = self.dims.lon == other.dims.lon
        lat_compare = self.dims.lat == other.dims.lat
        depth_compare = self.dims.depth == other.dims.depth
        time_compare = self.time.datetime[-1] <= other.time.datetime[0]
        data_compare = self.obj_iter(self.data) == self.obj_iter(other.data)
        old_data = self.obj_iter(self.data)
        new_data = self.obj_iter(other.data)
        if not lon_compare:
            raise ValueError('Horizontal longitude data are incompatible.')
        if not lat_compare:
            raise ValueError('Horizontal latitude data are incompatible.')
        if not depth_compare:
            raise ValueError('Vertical depth layers are incompatible.')
        if not time_compare:
            raise ValueError("Time periods are incompatible (`fvcom2' must be greater than or equal to `fvcom1')."
                             "`fvcom1' has end {} and `fvcom2' has start {}".format(self.time.datetime[-1],
                                                                                    other.time.datetime[0]))
        if not data_compare:
            raise ValueError('Loaded data sets for each RegularReader class must match.')
        if not (old_data == new_data) and (old_data or new_data):
            warn('Subsequent attempts to load data for this merged object will only load data from the first object. '
                 'Load data into each object before merging them.')

        # Copy ourselves to a new version for concatenation. self is the old so we get appended to by the new.
        idem = copy.copy(self)

        for var in self.obj_iter(idem.data):
            if 'MT' in idem.ds.variables[var].dimensions:
                setattr(idem.data, var, np.ma.concatenate((getattr(idem.data, var), getattr(other.data, var))))
        for time in self.obj_iter(idem.time):
            setattr(idem.time, time, np.concatenate((getattr(idem.time, time), getattr(other.time, time))))

        # Remove duplicate times.
        time_indices = np.arange(len(idem.time.time))
        _, dupes = np.unique(idem.time.time, return_index=True)
        dupe_indices = np.setdiff1d(time_indices, dupes)
        time_mask = np.ones(time_indices.shape, dtype=bool)
        time_mask[dupe_indices] = False
        for var in self.obj_iter(idem.data):
            # Only delete things with a time dimension.
            if 'MT' in idem.ds.variables[var].dimensions:
                # time_axis = idem.ds.variables[var].dimensions.index('time')
                setattr(idem.data, var, getattr(idem.data, var)[time_mask, ...])  # assume time is first
                # setattr(idem.data, var, np.delete(getattr(idem.data, var), dupe_indices, axis=time_axis))
        for time in self.obj_iter(idem.time):
            try:
                time_axis = idem.ds.variables[time].dimensions.index('MT')
                setattr(idem.time, time, np.delete(getattr(idem.time, time), dupe_indices, axis=time_axis))
            except KeyError:
                # This is hopefully one of the additional time variables which doesn't exist in the netCDF dataset.
                # Just delete the relevant indices by assuming that time is the first axis.
                setattr(idem.time, time, np.delete(getattr(idem.time, time), dupe_indices, axis=0))
            except ValueError:
                # If we're fiddling around with the HYCOMReader data, we might not have the right name for the time
                # dimension, so the .index('time') will fail. Just assume that that is the case and therefore time is
                # the first dimension.
                setattr(idem.time, time, np.delete(getattr(idem.time, time), dupe_indices, axis=0))

        # Update dimensions accordingly.
        idem.dims.time = len(idem.time.time)

        return idem


    def _load_time(self):
        """
        Populate a time object with additional useful time representations from the netCDF time data.

        """

        # Fake it till we make it by adding variables to the HYCOM data which match the other files we use. I get the
        # feeling this should all be in __init__() really.

        # For each variable, replace its dimension names with our standard ones.
        standard_variables = {'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'}
        standard_dimensions = {'X': 'lon', 'Y': 'lat', 'MT': 'time', 'Depth': 'depth'}
        for var in list(self.ds.variables.keys()):
            if var in standard_variables:
                self.ds.variables[standard_variables[var]] = self.ds.variables[var]
        # Also make dimension attributes for the standard dimension names.
        for dim in standard_dimensions:
            setattr(self.dims, standard_dimensions[dim], getattr(self.dims, dim))

        time = self.ds.variables['time'][:]

        # Make other time representations.
        self.time.datetime = num2date(time, units=getattr(self.ds.variables['time'], 'units'))
        if isinstance(self.time.datetime, (list, tuple, np.ndarray)):
            setattr(self.time, 'Times', np.array([datetime.strftime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in self.time.datetime]))
        else:
            setattr(self.time, 'Times', datetime.strftime(self.time.datetime, '%Y-%m-%dT%H:%M:%S.%f'))
        self.time.time = date2num(self.time.datetime, units='days since 1858-11-17 00:00:00')
        self.time.Itime = np.floor(self.time.time)
        self.time.Itime2 = (self.time.time - np.floor(self.time.time)) * 1000 * 60 * 60  # microseconds since midnight
        self.time.datetime = self.time.datetime
        self.time.matlabtime = self.time.time + 678942.0  # convert to MATLAB-indexed times from Modified Julian Date.

    def _load_grid(self):
        """
        Load the grid data.

        Convert from UTM to spherical if we haven't got those data in the existing output file.

        """

        # This is only subtly different from the one in RegularReader, but since the dimensions associated with each
        # variable differ in name, we have to adjust the code here. This is bound to introduce bugs eventually.

        grid_variables = ['lon', 'lat', 'x', 'y', 'depth']

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
                setattr(self.grid, grid, np.zeros((self.dims.lon, self.dims.lat)))
            except ValueError as value_error_message:
                warn('Variable {} has a problem with the data. Setting value as all zeros.'.format(grid))
                setattr(self.grid, grid, np.zeros(self.ds.variables[grid].shape))

        # Fix the longitudes.
        _lon = getattr(self.grid, 'lon') % int(self.ds.variables['lon'].modulo.split(' ')[0])
        _lon[_lon > 180] -= 360
        setattr(self.grid, 'lon', _lon)

        # Make the grid data the right shape for us to assume it's an FVCOM-style data set.
        # self.grid.lon, self.grid.lat = np.meshgrid(self.grid.lon, self.grid.lat)
        # self.grid.lon, self.grid.lat = self.grid.lon.ravel(), self.grid.lat.ravel()

        # Update dimensions to match those we've been given, if any. Omit time here as we shouldn't be touching that
        # dimension for any variable in use in here.
        for dim in self._dims:
            if dim != 'MT':
                setattr(self.dims, dim, len(self._dims[dim]))

        # Convert the given W/E/S/N coordinates into node and element IDs to subset.
        if self._bounding_box:
            # We need to use the original Dataset lon and lat values here as they have the right shape for the
            # subsetting.

            # HYCOM longitude values are modulo this value. Latitudes are just as is. No idea why.
            weird_modulo = int(self.ds.variables['lon'].modulo.split(' ')[0])
            hycom_lon = self.ds.variables['lon'][:] % weird_modulo
            hycom_lon[hycom_lon > 180] -= 360  # make range -180 to 180.
            self._dims['X'] = (hycom_lon > self._dims['wesn'][0]) & (hycom_lon < self._dims['wesn'][1])
            # Latitude is much more straightforward.
            self._dims['Y'] = (self.ds.variables['lat'][:] > self._dims['wesn'][2]) & \
                              (self.ds.variables['lat'][:] < self._dims['wesn'][3])
            # self._dims['X'] = np.argwhere((self.ds.variables['lon'][:] > self._dims['wesn'][0]) &
            #                               (self.ds.variables['lon'][:] < self._dims['wesn'][1]))
            # self._dims['Y'] = np.argwhere((self.ds.variables['lat'][:] > self._dims['wesn'][2]) &
            #                               (self.ds.variables['lat'][:] < self._dims['wesn'][3]))

        related_variables = {'X': ('Longitude', 'lon'), 'Y': ('Latitude', 'lat')}
        for spatial_dimension in 'X', 'Y':
            if spatial_dimension in self._dims:
                spatial_index = self.ds.variables[spatial_dimension].dimensions.index(spatial_dimension)
                setattr(self.dims, spatial_dimension, self._dims[spatial_dimension].shape[spatial_index])
                for var in related_variables[spatial_dimension]:
                    try:
                        var_shape = [i for i in np.shape(self.ds.variables[var])]
                        if 'Depth' in (self._dims, self.ds.variables[var].dimensions):
                            var_shape[self.ds.variables[var].dimensions.index('Depth')] = self.dims.siglay
                        _temp = np.empty(var_shape) * np.nan
                        # This doesn't work with the HYCOM data at the moment. I haven't translated this from FVCOM's
                        # approach yet.
                        if 'Depth' in self.ds.variables[var].dimensions:
                            # First get the depth layers, then get the horizontal positions. Untested!
                            # TODO: Test this!
                            if 'Depth' in self._dims:
                                _temp = self.ds.variables[var][self._dims['Depth'], ...]
                                _temp[self._dims[spatial_dimension]] = self.ds.variables[var][:][self._dims[spatial_dimension]]
                            else:
                                _temp = self.ds.variables[var][self._dims['Depth'], ...]
                                _temp[self._dims[spatial_dimension]] = self.ds.variables[var][:][self._dims[spatial_dimension]]
                        else:
                            _temp[self._dims[spatial_dimension]] = self.ds.variables[var][:][self._dims[spatial_dimension]]
                    except KeyError:
                        # Try and do something vaguely useful.
                        if 'depth' in var:
                            _temp = np.empty((self.dims.depth, getattr(self.dims, spatial_dimension)))
                        else:
                            _temp = np.empty(getattr(self.dims, spatial_dimension))
                setattr(self.grid, var, _temp)

            if self._bounding_box:
                # Make the indices non-dimensional for the spatial dimensions.
                self._dims[spatial_dimension] = np.ravel_multi_index(np.argwhere(self._dims[spatial_dimension]).T,
                                                                     self._dims[spatial_dimension].shape)

        # Check if we've been given vertical dimensions to subset in too, and if so, do that. Check we haven't
        # already done this if the 'node' and 'nele' sections above first.
        for var in ['depth']:
            short_dim = copy.copy(var)
            # Assume we need to subset this one unless 'node' or 'nele' are missing from self._dims. If they're in
            # self._dims, we've already subsetted in the 'node' and 'nele' sections above, so doing it again here
            # would fail.
            subset_variable = True
            if 'X' in self._dims or 'Y' in self._dims:
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
        self.grid.x_range = np.ptp(self.grid.x)
        self.grid.y_range = np.ptp(self.grid.y)

        # Only do the conversions when we have more than a single point since the relevant ranges will be zero with
        # only one position.
        if self.dims.lon > 1 and self.dims.lat > 1:
            if self.grid.lon_range == 0 and self.grid.lat_range == 0:
                self.grid.lon, self.grid.lat = lonlat_from_utm(self.grid.x, self.grid.y, zone=self._zone)
                self.grid.lon_range = np.ptp(self.grid.lon)
                self.grid.lat_range = np.ptp(self.grid.lat)
            if self.grid.lon_range == 0 and self.grid.lat_range == 0:
                self.grid.x, self.grid.y = utm_from_lonlat(self.grid.lon, self.grid.lat)
                self.grid.x_range = np.ptp(self.grid.x)
                self.grid.y_range = np.ptp(self.grid.y)

        # Make a bounding box variable too (spherical coordinates): W/E/S/N
        self.grid.bounding_box = (np.min(self.grid.lon), np.max(self.grid.lon),
                                  np.min(self.grid.lat), np.max(self.grid.lat))

    def load_data(self, var):
        """
        Load the given variable/variables.

        Parameters
        ----------
        var : list-like, str
            List of variables to load.

        """

        # This is only subtly different from the one in RegularReader, but since the dimensions associated with each
        # variable differ in name, we have to adjust the code here. This is bound to introduce bugs eventually.

        # Check if we've got iterable variables and make one if not.
        try:
            _ = (e for e in var)
        except TypeError:
            var = [var]

        for v in var:
            if v not in self.ds.variables:
                raise KeyError("Variable '{}' not present in {}.".format(v, self._fvcom))

            # Get this variable's dimensions
            var_dim = self.ds.variables[v].dimensions
            variable_shape = self.ds.variables[v].shape
            variable_indices = [np.arange(i) for i in variable_shape]
            for dimension in var_dim:
                if dimension in self._dims:
                    # Replace their size with anything we've been given in dims.
                    variable_index = var_dim.index(dimension)
                    if self._bounding_box and dimension in ('X', 'Y'):
                        rows, columns = np.unravel_index(self._dims[dimension], (self.ds.dimensions['Y'].size, self.ds.dimensions['X'].size))
                        if dimension == 'X':
                            variable_indices[var_dim.index('X')] = np.unique(columns)
                        elif dimension == 'Y':
                            variable_indices[var_dim.index('Y')] = np.unique(rows)
                    else:
                        variable_indices[variable_index] = self._dims[dimension]

            # Check the data we're loading is the same shape as our existing dimensions.
            lon_compare = self.ds.dimensions['X'].size == self.dims.lon
            lat_compare = self.ds.dimensions['Y'].size == self.dims.lat
            depth_compare = self.ds.dimensions['Depth'].size == self.dims.depth
            time_compare = self.ds.dimensions['MT'].size == self.dims.time
            # Check again if we've been asked to subset in any dimension.
            if 'lon' in self._dims:
                lon_compare = len(self.ds.variables['X'][self._dims['lon']]) == self.dims.lon
            if 'lat' in self._dims:
                lat_compare = len(self.ds.variables['Y'][self._dims['lat']]) == self.dims.lat
            if 'depth' in self._dims:
                depth_compare = len(self.ds.variables['Depth'][self._dims['depth']]) == self.dims.depth
            if 'time' in self._dims:
                time_compare = len(self.ds.variables['MT'][self._dims['time']]) == self.dims.time

            if not lon_compare:
                raise ValueError('Longitude data are incompatible. You may be trying to load data after having already '
                                 'concatenated a RegularReader object, which is unsupported.')
            if not lat_compare:
                raise ValueError('Latitude data are incompatible. You may be trying to load data after having already '
                                 'concatenated a RegularReader object, which is unsupported.')
            if not depth_compare:
                raise ValueError('Vertical depth layers are incompatible. You may be trying to load data after having '
                                 'already concatenated a RegularReader object, which is unsupported.')
            if not time_compare:
                raise ValueError('Time period is incompatible. You may be trying to load data after having already '
                                 'concatenated a RegularReader object, which is unsupported.')

            if 'MT' not in var_dim:
                # Should we error here or carry on having warned?
                warn("{} does not contain an `MT' (time) dimension.".format(v))

            attributes = type('attributes', (object,), {})()
            for attribute in self.ds.variables[v].ncattrs():
                setattr(attributes, attribute, getattr(self.ds.variables[v], attribute))
            setattr(self.atts, v, attributes)

            data = self.ds.variables[v][variable_indices]  # data are automatically masked
            setattr(self.data, v, data)


def read_hycom(regular, variables, noisy=False, **kwargs):
    """
    Read regularly gridded model data and provides a HYCOMReader object which mimics a FileReader object.

    Parameters
    ----------
    regular : str, pathlib.Path
        Files to read.
    variables : list
        Variables to extract. Variables missing in the files raise an error.
    noisy : bool, optional
        Set to True to enable verbose output. Defaults to False.
    Remaining keyword arguments are passed to HYCOMReader.

    Returns
    -------
    hycom_model : PyFVCOM.preproc.HYCOMReader
        A HYCOMReader object with the requested variables loaded.

    """

    if 'variables' not in kwargs:
        kwargs.update({'variables': variables})

    for ii, file in enumerate(regular):
        if noisy:
            print('Loading file {}'.format(file))
        if ii == 0:
            hycom_model = HYCOMReader(str(file), **kwargs)
        else:
            hycom_model += HYCOMReader(str(file), **kwargs)

    return hycom_model


class Restart(FileReader):
    """
    Use and abuse FVCOM restart files.

    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Store which variables have been replaced so we can do the right thing when writing to netCDF (i.e. use the
        # replaced data rather than what's in the input restart file).
        self.replaced = []

    def replace_variable(self, variable, data):
        """
        Replace the values in `variable' with the given `data'.

        Parameters
        ----------
        variable : str
            The variable in the restart file to replace.
        data : numpy.ndarray
            The data with which to replace it.

        """

        setattr(self.data, variable, data)
        self.replaced.append(variable)

    def write_restart(self, restart_file, **ncopts):
        """
        Write out an FVCOM-formatted netCDF file based.

        Parameters
        ----------
        restart_file : str, pathlib.Path
            The output file to create.
        ncopts : dict
            The netCDF options passed as kwargs to netCDF4.Dataset.

        """

        with Dataset(restart_file, 'w', clobber=True, **ncopts) as ds:
            # Re-create all the dimensions and global attributes in the loaded restart file.
            for name, dimension in self.ds.dimensions.items():
                ds.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
            # Job-lot copy of the global attributes.
            ds.setncatts(self.ds.__dict__)

            # Make all the variables.
            for name, variable in self.ds.variables.items():
                x = ds.createVariable(name, variable.datatype, variable.dimensions)
                # Copy variable attributes all at once via dictionary
                ds[name].setncatts(self.ds[name].__dict__)
                if self._noisy:
                    print('Writing {}'.format(name), end=' ')
                if name in self.replaced:
                    if self._noisy:
                        print('NEW DATA')
                    ds[name][:] = getattr(self.data, name)
                else:
                    if self._noisy:
                        print('existing data')
                    ds[name][:] = self.ds[name][:]

