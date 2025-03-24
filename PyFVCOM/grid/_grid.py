"""
Tools for manipulating and converting unstructured grids in a range of formats.

"""

# TODO: This is a massive sprawling collection of functions. We should split it up into more sensible subdivisions
#  within PyFVCOM.grid to make it more manageable and generally more useable.

from __future__ import print_function, division

import copy
import math
import multiprocessing
import os
import sys
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy.spatial
import shapefile
import shapely.geometry
from dateutil.relativedelta import relativedelta
from matplotlib.dates import date2num as mtime
from matplotlib.tri import CubicTriInterpolator
from matplotlib.tri.triangulation import Triangulation
from netCDF4 import Dataset, date2num
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.spatial.qhull import QhullError
from utide import reconstruct, ut_constants
from utide.utilities import Bunch

from PyFVCOM.coordinate import utm_from_lonlat, lonlat_from_utm
from PyFVCOM.ocean import zbar
from PyFVCOM.utilities.general import PassiveStore, fix_range, cart2pol, pol2cart
from PyFVCOM.utilities.time import date_range


class GridReaderNetCDF(object):
    """ Read in and store a given FVCOM grid in our data format. """

    def __init__(self, filename, dims=None, zone='30N', debug=False, verbose=False):
        """
        Load the grid data.

        Convert from UTM to spherical if we haven't got those data in the existing output file.

        Parameters
        ----------
        filename : str, pathlib.Path
            The FVCOM netCDF file to read.
        dims : dict, optional
            Dictionary of dimension names along which to subsample e.g. dims={'nele': [0, 10, 100], 'node': 100}.
            All netCDF variable dimensions are specified as list of indices.
            Any combination of dimensions is possible; omitted dimensions are loaded in their entirety.
            Negative indices are supported.
            A special dimension of 'wesn' can be used to specify a bounding box within which to extract the model
            grid and data.
        zone : str, list-like, optional
            UTM zones (defaults to '30N') for conversion of UTM to spherical coordinates.
        verbose : bool, optional
            Set to True to enable verbose output. Defaults to False.
        debug : bool, optional
            Set to True to enable debug output. Defaults to False.

        """

        self._debug = debug
        self._noisy = verbose

        ds = Dataset(filename, 'r')
        self._dims = copy.deepcopy(dims)
        if self._dims is None:
            self._dims = {}

        self._bounding_box = False
        if 'wesn' in self._dims:
            self._bounding_box = True

        grid_metrics = {'ntsn': 'node', 'nbsn': 'node', 'ntve': 'node', 'nbve': 'node', 'art1': 'node', 'art2': 'node',
                        'a1u': 'nele', 'a2u': 'nele', 'nbe': 'nele'}
        grid_variables = ['lon', 'lat', 'x', 'y', 'lonc', 'latc', 'xc', 'yc', 'h', 'siglay', 'siglev']

        # Get the grid data.
        for grid in grid_variables:
            try:
                setattr(self, grid, ds.variables[grid][:])
                # Save the attributes.
                attributes = PassiveStore()
                for attribute in ds.variables[grid].ncattrs():
                    setattr(attributes, attribute, getattr(ds.variables[grid], attribute))
                # setattr(self.atts, grid, attributes)
            except KeyError:
                # Make zeros for this missing variable so we can convert from the non-missing data below.
                if grid.endswith('c'):
                    setattr(self, grid, np.zeros(ds.dimensions['nele'].size).T)
                else:
                    setattr(self, grid, np.zeros(ds.dimensions['node'].size).T)
            except ValueError as value_error_message:
                warn('Variable {} has a problem with the data. Setting value as all zeros.'.format(grid))
                print(value_error_message)
                setattr(self, grid, np.zeros(ds.variables[grid].shape))

        # And the triangulation
        try:
            self.nv = ds.variables['nv'][:].astype(int)  # force integers even though they should already be so
            self.triangles = copy.copy(self.nv.T - 1)  # zero-indexed for python
        except KeyError:
            # If we don't have a triangulation, make one. Warn that if we've made one, it might not match the
            # original triangulation used in the model run.
            if self._debug:
                print("Creating new triangulation since we're missing one", flush=True)
            triangulation = Triangulation(self.lon, self.lat)
            self.triangles = triangulation.triangles
            self.nv = copy.copy(self.triangles.T + 1)
            dims.nele = self.triangles.shape[0]
            warn('Triangulation created from node positions. This may be inconsistent with the original triangulation.')

        # Fix broken triangulations if necessary.
        if self.nv.min() != 1:
            if self._debug:
                print('Fixing broken triangulation. Current minimum for nv is {} and for triangles is {} but they '
                      'should be 1 and 0, respectively.'.format(self.nv.min(), self.triangles.min()), flush=True)
            self.nv = (ds.variables['nv'][:].astype(int) - ds.variables['nv'][:].astype(int).min()) + 1
            self.triangles = copy.copy(self.nv.T) - 1

        # Convert the given W/E/S/N coordinates into node and element IDs to subset.
        if self._bounding_box:
            self._make_subset_dimensions()

        # If we've been given a spatial dimension to subsample in, fix the triangulation.
        if 'nele' in self._dims or 'node' in self._dims:
            if self._debug:
                print('Fix triangulation table as we have been asked for only specific nodes/elements.', flush=True)

            if 'node' in self._dims:
                new_tri, new_ele = reduce_triangulation(self.triangles, self._dims['node'], return_elements=True)
                if not new_ele.size and 'nele' not in self._dims:
                    if self._noisy:
                        print('Nodes selected cannot produce new triangulation and no elements specified so including all element of which the nodes are members')
                    self._dims['nele'] = np.squeeze(np.argwhere(np.any(np.isin(self.triangles, self._dims['node']), axis=1)))
                    if self._dims['nele'].size == 1:  # Annoying error for the difference between array(n) and array([n])
                        self._dims['nele'] = np.asarray([self._dims['nele']])
                elif 'nele' not in self._dims:
                    if self._noisy:
                        print('Elements not specified but reducing to only those within the triangulation of selected nodes')
                    self._dims['nele'] = new_ele
                elif not np.array_equal(np.sort(new_ele), np.sort(self._dims['nele'])):
                    # Try culling some elements as what may happen is we have asked for some nodes which form an
                    # element but we haven't asked for that element.
                    #        a
                    #        /\
                    #       /  \
                    #      / 1  \
                    #    b/______\c
                    #
                    # That is, we've asked for nodes a, b and c but not element 1. The way to check this is to find
                    # all the nodes in the triangulation for the given elements and if those match the nodes we've
                    # asked for, use the elements to cull the triangulation object.
                    triangulation_nodes = np.unique(self.triangles[self._dims['nele']])
                    if np.all(triangulation_nodes == np.sort(self._dims['node'])):
                        new_tri = self.triangles[self._dims['nele']]
                        # Remap nodes to a new index. Work on a copy so we don't end up replacing a value more than
                        # once.
                        new_index = np.arange(0, len(self._dims['node']))
                        original_tri = new_tri.copy()
                        for this_old, this_new in zip(self._dims['node'], new_index):
                            new_tri[original_tri == this_old] = this_new
                    else:
                        if self._noisy:
                            print('Mismatch between given elements and nodes for triangulation, retaining original elements')
            else:
                if self._noisy:
                    print('Nodes not specified but reducing to only those within the triangulation of selected elements')
                self._dims['node'] = np.unique(self.triangles[self._dims['nele'], :])
                new_tri = reduce_triangulation(self.triangles, self._dims['node'])

            self.nv = new_tri.T + 1
            self.triangles = new_tri

        # If we have node/nele dimensions, subset the relevant arrays here. Make all zero arrays if we're missing
        # them in the netCDF file.
        spatial_variables = {'node': ['x', 'y', 'lon', 'lat', 'h', 'siglay', 'siglev'],
                             'nele': ['xc', 'yc', 'lonc', 'latc', 'h_center', 'siglay_center', 'siglev_center']}

        for spatial_dimension in spatial_variables:
            if spatial_dimension in self._dims:
                spatial_points = len(self._dims[spatial_dimension])
                for var in spatial_variables[spatial_dimension]:
                    try:
                        _temp = ds.variables[var][:]
                        _temp = _temp[..., self._dims[spatial_dimension]]
                    except KeyError:
                        if spatial_dimension == 'nele':
                            if self._noisy:
                                print('Adding element-centred {} for compatibility.'.format(var))
                        if 'siglay' in var:
                            var_shape = (ds.dimensions['siglay'].size, spatial_points)
                        elif 'siglev' in var:
                            var_shape = (ds.dimensions['siglev'].size, spatial_points)
                        else:
                            var_shape = spatial_points
                        _temp = np.zeros(var_shape)

                    setattr(self, var, _temp)

        # Load the grid metrics data separately as we don't want to set a bunch of zeros for missing data.
        for metric, grid_pos in grid_metrics.items():
            if metric in ds.variables:
                if grid_pos in self._dims:
                    metric_raw = ds.variables[metric][:]
                    setattr(self, metric, metric_raw[..., self._dims[grid_pos]])
                else:
                    setattr(self, metric, ds.variables[metric][:])
                # Save the attributes.
                attributes = PassiveStore()
                for attribute in ds.variables[metric].ncattrs():
                    setattr(attributes, attribute, getattr(ds.variables[metric], attribute))
                # setattr(self.atts, metric, attributes)

                # Fix the indexing and shapes of the grid metrics variables. Only transpose and offset indexing for nbe.
                if metric == 'nbe':
                    setattr(self, metric, getattr(self, metric).T - 1)

        # Add compatibility for FVCOM3 (these variables are only specified on the element centres in FVCOM4+ output
        # files). Only create the element centred values if we have the same number of nodes as in the triangulation.
        # This does not occur if we've been asked to extract an incompatible set of nodes and elements, for whatever
        # reason (e.g. testing). We don't add attributes for the data if we've created it as doing so is a pain.
        for var in 'h_center', 'siglay_center', 'siglev_center':
            if self._debug:
                print('Add element-based compatibility arrays', flush=True)
            try:
                if 'nele' in self._dims:
                    var_raw = ds.variables[var][:]
                    setattr(self, var, var_raw[..., self._dims['nele']])
                else:
                    setattr(self, var, ds.variables[var][:])
                # Save the attributes.
                attributes = PassiveStore()
                for attribute in ds.variables[var].ncattrs():
                    setattr(attributes, attribute, getattr(ds.variables[var], attribute))
                # setattr(self.atts, var, attributes)
            except KeyError:
                if self._noisy:
                    print('Missing {} from the netCDF file. Trying to recreate it from other sources.'.format(var))
                if self.nv.max() == len(self.x):
                    try:
                        setattr(self, var, nodes2elems(getattr(self, var.split('_')[0]), self.triangles))
                    except IndexError:
                        # Maybe the array's the wrong way around. Flip it and try again.
                        setattr(self, var, nodes2elems(getattr(self, var.split('_')[0]).T, self.triangles))
                else:
                    # The triangulation is invalid, so we can't properly move our existing data, so just set things
                    # to 0 but at least they're the right shape. Warn accordingly.
                    if self._noisy:
                        print('{} cannot be migrated to element centres (invalid triangulation). Setting to zero.'.format(var))
                    if var == 'siglev_center':
                        setattr(self, var, np.zeros((ds.dimensions['siglev'].size, dims.nele)))
                    elif var == 'siglay_center':
                        setattr(self, var, np.zeros((ds.dimensions['siglay'].size, dims.nele)))
                    elif var == 'h_center':
                        setattr(self, var, np.zeros((dims.nele)))
                    else:
                        raise ValueError('Inexplicably, we have a variable not in the loop we have defined.')

        # Check if we've been given only vertical dimensions to subset in too, and if so, do that. At this point,
        # if we've got node/element subsets given, we've already subsetted those.
        for var in 'siglay', 'siglev', 'siglay_center', 'siglev_center':
            # Only carry on if we have this variable in the output file with which we're working (mainly this
            # is for compatibility with FVCOM 3 outputs which do not have the _center variables).
            if var not in ds.variables:
                continue
            short_dim = copy.copy(var)
            # Strip off the _center to match the dimension name.
            if short_dim.endswith('_center'):
                short_dim = short_dim.split('_')[0]
            if short_dim in self._dims:
                if short_dim in ds.variables[var].dimensions:
                    if self._debug:
                        print(f'Subsetting {var} in the vertical ({short_dim} = {self._dims[short_dim]})', flush=True)
                    _temp = getattr(self, var)[self._dims[short_dim], ...]
                    setattr(self, var, _temp)

        # Make depth-resolved sigma data. This is useful for plotting things.
        for var in self:
            # Ignore previously created depth-resolved data (in the case where we're updating the grid with a call to
            # self._load_data() with dims supplied).
            if var.startswith('sig') and not var.endswith('_z'):
                # Make the depths negative down so we end up with positive down for {var}_z (since var is negative
                # down already).
                if var.endswith('_center'):
                    z = -self.h_center
                else:
                    z = -self.h

                if self._debug:
                    print(f'Making water depth vertical grid: {var}_z', flush=True)

                _original_sig = getattr(self, var)

                # Set the sigma data range to -1 to 0 (rather than close to -1 and 0) for siglay so that the maximum
                # depth value is equal to the actual depth.
                _fixed_sig = fix_range(_original_sig, -1, 0)
                # h_center can have a time dimension (when running with sediment transport and morphological
                # update enabled). As such, we need to account for that in the creation of the _z arrays.
                if np.ndim(z) > 1:
                    z = z[:, np.newaxis, :]
                    _fixed_sig = _fixed_sig[np.newaxis, ...]
                try:
                    setattr(self, '{}_z'.format(var), _fixed_sig * z)
                except ValueError:
                    # The arrays might be the wrong shape for broadcasting to work, so transpose and retranspose
                    # accordingly. This is less than ideal.
                    warn(f'Depth-resolved sigma {var} seems to be the wrong shape. Trying again.')
                    setattr(self, '{}_z'.format(var), (_fixed_sig.T * z).T)

        # Check ranges and if zero assume we're missing that particular type, so convert from the other accordingly.
        self.lon_range = np.ptp(self.lon)
        self.lat_range = np.ptp(self.lat)
        self.lonc_range = np.ptp(self.lonc)
        self.latc_range = np.ptp(self.latc)
        self.x_range = np.ptp(self.x)
        self.y_range = np.ptp(self.y)
        self.xc_range = np.ptp(self.xc)
        self.yc_range = np.ptp(self.yc)

        # Only do the conversions when we have more than a single point since the relevant ranges will be zero with
        # only one position. If we've got zeros for lon and lat, then we know our native coordinate type is cartesian.
        # Otherwise, it might be spherical.
        self.native_coordinates = 'not specified'
        if self.lon_range == 0 and self.x_range != 0:
            self.native_coordinates = 'cartesian'
        elif self.lon_range != 0 and self.x_range == 0:
            self.native_coordinates = 'spherical'

        if len(self.lon) > 1:
            if self.lon_range == 0 and self.lat_range == 0:
                self.lon, self.lat = lonlat_from_utm(self.x, self.y, zone=zone)
                self.lon_range = np.ptp(self.lon)
                self.lat_range = np.ptp(self.lat)
            if self.x_range == 0 and self.y_range == 0:
                self.x, self.y, _ = utm_from_lonlat(self.lon, self.lat)
                self.x_range = np.ptp(self.x)
                self.y_range = np.ptp(self.y)
        if len(self.lonc) > 1:
            if self.lonc_range == 0 and self.latc_range == 0:
                self.lonc, self.latc = lonlat_from_utm(self.xc, self.yc, zone=zone)
                self.lonc_range = np.ptp(self.lonc)
                self.latc_range = np.ptp(self.latc)
            if self.xc_range == 0 and self.yc_range == 0:
                self.xc, self.yc, _ = utm_from_lonlat(self.lonc, self.latc)
                self.xc_range = np.ptp(self.xc)
                self.yc_range = np.ptp(self.yc)

        # Make a bounding box variable too (spherical coordinates): W/E/S/N
        self.bounding_box = (np.min(self.lon), np.max(self.lon), np.min(self.lat), np.max(self.lat))

        ds.close()

        if self._debug:
            print('Finished loading grid from netCDF')

    def __iter__(self):
        return (a for a in dir(self) if not a.startswith('_'))

    def _update_dimensions(self, dims):
        """
        Update dimensions to match those we've been given, if any. Omit time here as we shouldn't be touching that
        # dimension for any variable in use in here.

        Parameters
        ----------
        dims : dict
            The dimensions to update.

        """
        for dim in dims:
            if dim != 'time':
                if self._noisy:
                    print('Resetting {} dimension length from {} to {}'.format(dim, getattr(dims, dim), len(dims[dim])))
                setattr(self._dims, dim, len(dims[dim]))

    def _make_subset_dimensions(self):
        """
        If the 'wesn' keyword has been included in the supplied dimensions, interactively select a region if the
        value of 'wesn' is not a shapely Polygon. If it is a shapely Polygon, use that for the subsetting.

        """

        bounding_poly = None
        if 'wesn' in self._dims:
            if isinstance(self._dims['wesn'], shapely.geometry.Polygon):
                bounding_poly = self._dims['wesn']
                if self._debug:
                    print('Subsetting the data with a polygon', flush=True)

        # Do the subset of our grid.
        if self._debug:
            print('Starting the subsetting...', end=' ', flush=True)
        self._dims['node'], self._dims['nele'], _ = subset_domain(self.lon, self.lat, self.triangles, bounding_poly)
        self._get_data_pattern = 'memory'
        # Remove the now useless 'wesn' dimension.
        self._dims.pop('wesn')
        if self._debug:
            print('done.', flush=True)


class _GridReader(object):

    def __init__(self, gridfile, native_coordinates='spherical', zone=None, verbose=False):
        """
        Load an unstructured grid.

        Parameters
        ----------
        gridfile : str, pathlib.Path
            The grid file to load.
        native_coordinates : str
            Defined the coordinate system used in the grid ('spherical' or 'cartesian'). Defaults to `spherical'.
        zone : str, optional
            If `native_coordinates' is 'cartesian', give the UTM zone as a string, formatted as, for example,
            '30N'. Ignored if `native_coordinates' is 'spherical'.
        verbose : bool, optional
            Set to True to enable verbose output. Defaults to False.

        """

        # TODO: Add the read_smesh_mesh reader in here too. What's the typical smeshing file extension for its files?

        self.filename = gridfile
        self.native_coordinates = native_coordinates
        self.zone = zone
        self._noisy = verbose
        self.open_boundary_nodes = None

        if self.native_coordinates.lower() == 'cartesian' and not self.zone:
            raise ValueError('For cartesian coordinates, a UTM Zone for the grid is required.')

        # Default to no node strings. Only the SMS read function can parse them as they're stored within that file.
        # We'll try and grab them from the FVCOM file assuming the standard FVCOM naming conventions.
        nodestrings = []
        types = None
        try:
            basedir = str(self.filename.parent)
            basename = self.filename.stem
            extension = self.filename.suffix
            self.filename = str(self.filename)  # most stuff will want a string later.
        except AttributeError:
            extension = os.path.splitext(self.filename)[-1]
            basedir, basename = os.path.split(self.filename)
            basename = basename.replace(extension, '')

        if extension == '.2dm':
            if self._noisy:
                print('Loading SMS grid: {}'.format(self.filename))
            triangle, nodes, x, y, z, types, nodestrings = read_sms_mesh(self.filename, nodestrings=True)
        elif extension == '.dat':
            if self._noisy:
                print('Loading FVCOM grid: {}'.format(self.filename))
            triangle, nodes, x, y, z = read_fvcom_mesh(self.filename)
            try:
                obcname = basename.replace('_grd', '_obc')
                obcfile = os.path.join(basedir, '{}.dat'.format(obcname))
                obc_node_array, types, count = read_fvcom_obc(obcfile)
                nodestrings = parse_obc_sections(obc_node_array, triangle)
                if self._noisy:
                    print('Found and parsed open boundary file: {}'.format(obcfile))
            except OSError:
                # File probably doesn't exist, so just carry on.
                pass
        elif extension == '.gmsh':
            if self._noisy:
                print('Loading GMSH grid: {}'.format(self.filename))
            triangle, nodes, x, y, z = read_gmsh_mesh(self.filename)
        elif extension == '.m21fm':
            if self._noisy:
                print('Loading MIKE21 grid: {}'.format(self.filename))
            triangle, nodes, x, y, z, grid_type = read_mike_mesh(self.filename)
        else:
            raise ValueError('Unknown file format ({}) for file: {}'.format(extension, self.filename))

        # Make open boundary objects from the nodestrings.
        self.open_boundary = []
        for nodestring in nodestrings:
            self.open_boundary.append(OpenBoundary(nodestring))

        if self.native_coordinates.lower() != 'spherical':
            # Convert from UTM.
            self.lon, self.lat = lonlat_from_utm(x, y, self.zone)
            self.x, self.y = x, y
        else:
            # Convert to UTM.
            self.lon, self.lat = x, y
            self.x, self.y, _ = utm_from_lonlat(x, y, zone=self.zone)

        self.triangles = triangle
        self.nv = triangle.T + 1  # for compatibility with FileReader
        self.h = z
        self.nodes = nodes
        self.types = types
        self.open_boundary_nodes = nodestrings
        # Make element-centred versions of everything.
        self.xc = nodes2elems(self.x, self.triangles)
        self.yc = nodes2elems(self.y, self.triangles)
        self.lonc = nodes2elems(self.lon, self.triangles)
        self.latc = nodes2elems(self.lat, self.triangles)
        self.h_center = nodes2elems(self.h, self.triangles)

        # Add the coordinate ranges too
        self.lon_range = np.ptp(self.lon)
        self.lat_range = np.ptp(self.lat)
        self.lonc_range = np.ptp(self.lonc)
        self.latc_range = np.ptp(self.latc)
        self.x_range = np.ptp(self.x)
        self.y_range = np.ptp(self.y)
        self.xc_range = np.ptp(self.xc)
        self.yc_range = np.ptp(self.yc)
        # Make a bounding box variable too (spherical coordinates): W/E/S/N
        self.bounding_box = (np.min(self.lon), np.max(self.lon), np.min(self.lat), np.max(self.lat))

    def __iter__(self):
        return (a for a in self.__dict__.keys() if not a.startswith('_'))


class _MakeDimensions(object):
    def __init__(self, grid_reader):
        """
        Calculate some dimensions from the given _GridReader object.

        Parameters
        ----------
        grid_reader : _GridReader
            The grid reader which contains the grid information from which we calculate the dimensions.

        """

        # Make the relevant dimensions.
        self.nele = len(grid_reader.xc)
        self.node = len(grid_reader.x)

    def __iter__(self):
        return (a for a in self.__dict__.keys() if not a.startswith('_'))


class Domain(object):
    """
    Class to hold information for unstructured grid from a range of file types. The aim is to abstract away the file
    format into a consistent object.

    Methods
    -------
    closest_node - find the closest node ID to the given position
    closest_element - find the closest element ID to the given position
    horizontal_transect_nodes - make a transect through nodes
    horizontal_transect_elements - make a transect through elements
    subset_domain - subset the model grid to the given region
    calculate_areas - calculate areas of elements
    calculate_control_area_and_volume - calculate control areas and volumes for the grid
    calculate_element_lengths - calculate element edge lengths
    gradient - compute the gradient of a field
    to_nodes - move values from elements to nodes
    to_elements - move values from nodes to elements
    in_element - check if a position in an element
    exterior - return the boundary of the grid
    info - print some information about the grid

    Attributes
    ----------
    dims - the grid dimensions (nodes, elements, number of open boundaries (if discernible) etc.
    grid - the grid data (lon, lat, x, y, triangles, open_boundaries, etc.).

    """

    def __init__(self, grid, native_coordinates, zone=None, noisy=False, debug=False, verbose=False):
        """
        Read in a grid and parse its structure into a standard format which we replicate throughout PyFVCOM.

        Parameters
        ----------
        grid : str, pathlib.Path
            Path to the model grid file. Supported formats includes:
                - SMS .2dm
                - FVCOM .dat
                - GMSH .gmsh
                - DHI MIKE21 .m21fm
                - FVCOM output file .nc
        native_coordinates : str
            Defined the coordinate system used in the grid ('spherical' or 'cartesian'). Defaults to `spherical'.
        zone : str, optional
            If `native_coordinates' is 'cartesian', give the UTM zone as a string, formatted as, for example,
            '30N'. Ignored if `native_coordinates' is 'spherical'.
        noisy : bool, optional
            Set to True to enable verbose output. Defaults to False.
        debug : bool, optional
            Set to True to enable debugging output. Defaults to False.

        """

        self._debug = debug
        self._noisy = noisy or verbose

        # Add some extra bits for the grid information.
        if Path(grid).suffix == '.nc':
            self.grid = GridReaderNetCDF(grid, zone=zone, verbose=verbose)
            # Add the open boundary list since we can't discern that from a netCDF file alone.
            self.grid.open_boundary_nodes = []
        else:
            self.grid = _GridReader(grid, native_coordinates, zone, verbose=verbose)
        self.dims = _MakeDimensions(self.grid)

        # Set two dimensions: number of open boundaries (obc) and number of open boundary nodes (open_boundary_nodes).
        if self.grid.open_boundary_nodes:
            # If we have more than one open boundary, we need to iterate through the list of lists to get the total
            # number, otherwise we can just len the list.
            try:
                self.dims.open_boundary_nodes = sum([len(i) for i in self.grid.open_boundary_nodes])
                self.dims.open_boundary = len(self.grid.open_boundary_nodes)
            except TypeError:
                self.dims.open_boundary_nodes = len(self.grid.open_boundary_nodes)
                self.dims.open_boundary = 1

    def __iter__(self):
        return (a for a in self.__dict__.keys() if not a.startswith('_'))

    @staticmethod
    def _closest_point(x, y, lon, lat, where, threshold=np.inf, vincenty=False, haversine=False):
        """
        Find the index of the closest node to the supplied position (x, y). Set `cartesian' to True for cartesian
        coordinates (defaults to spherical).

        Parameters
        ----------
        x, y : np.ndarray
            Grid coordinates within which to search. These are ignored if we have either of `vincenty' or `haversine'
            enabled.
        lon, lat : np.ndarray
            Spherical grid positions. These only used if we have either of `vincenty' or `haversine' enabled.
        where : list-like
            Arbitrary x, y position for which to find the closest model grid position.
        cartesian : bool, optional
            Set to True to use cartesian coordinates. Defaults to False.
        threshold : float, optional
            Give a threshold distance beyond which the closest grid is considered too far away. Units are the same as
            the coordinates in `where', unless using lat/lon and `vincenty' or `haversine' when it is in metres.
            Return None when beyond threshold.
        vincenty : bool, optional
            Use vincenty distance calculation. Allows specification of point in lat/lon but threshold in metres.
        haversine : bool, optional
            Use the simpler but much faster Haversine distance calculation. Allows specification of point in lon/lat
            but threshold in metres.

        Returns
        -------
        index : int, None
            Grid index which falls closest to the supplied position. If `threshold' is set and the distance from the
            supplied position to the nearest model node exceeds that threshold, `index' is None.

        """

        if vincenty and haversine:
            raise AttributeError("Please specify one of `haversine' or `vincenty', not both.")

        # We have to split this into two parts: if our threshold is in the same units as the grid (i.e. haversine and
        # vincenty are both False), then we can use the quick find_nearest_point function; if either of haversine or
        # vincenty have been given, we need to use the distance conversion functions, which are slower.
        if not vincenty and not haversine:
            _, _, _, index = find_nearest_point(x, y, *where, maxDistance=threshold)
            if np.any(np.isnan(index)):
                index[np.isnan(index)] = None

        # Note: this is really slow! Computing the distances between the entire grid and the point of interest is
        # very slow. There must be a faster way of doing this! which fall inside the `where' bounding box.
        # TODO: implement a faster conversion
        if vincenty:
            grid_pts = np.asarray([lon, lat]).T
            dist = np.asarray([vincenty_distance(pt_1, where) for pt_1 in grid_pts]) * 1000
        elif haversine:
            grid_pts = np.asarray([lon, lat]).T
            dist = np.asarray([haversine_distance(pt_1, where) for pt_1 in grid_pts]) * 1000

        if vincenty or haversine:
            index = np.argmin(dist)
            if threshold:
                if dist.min() < threshold:
                    index = np.argmin(dist)
                else:
                    index = None

        return index

    def closest_node(self, where, cartesian=False, threshold=np.inf, vincenty=False, haversine=False):
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
            Use the simpler but much faster Haversine distance calculation. Allows specification of points in lon/lat
            but threshold in metres.

        Returns
        -------
        index : int, None
            Grid index which falls closest to the supplied position. If `threshold' is set and the distance from the
            supplied position to the nearest model node exceeds that threshold, `index' is None.

        """
        if cartesian:
            x, y = self.grid.x, self.grid.y
        else:
            x, y = self.grid.lon, self.grid.lat

        return self._closest_point(x, y, self.grid.lon, self.grid.lat, where, threshold=threshold, vincenty=vincenty, haversine=haversine)

    def closest_element(self, where, cartesian=False, threshold=np.inf, vincenty=False, haversine=False):
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
        haversine : bool, optional
            Use the simpler but much faster Haversine distance calculation. Allows specification of points in lon/lat
            but threshold in metres.

        Returns
        -------
        index : int, None
            Grid index which falls closest to the supplied position. If `threshold' is set and the distance from the
            supplied position to the nearest model node exceeds that threshold, `index' is None.

        """
        if cartesian:
            x, y = self.grid.xc, self.grid.yc
        else:
            x, y = self.grid.lonc, self.grid.latc

        return self._closest_point(x, y, self.grid.lonc, self.grid.latc, where, threshold=threshold, vincenty=vincenty, haversine=haversine)

    def horizontal_transect_nodes(self, positions):
        """
        Extract node IDs along a line defined by `positions' [[x1, y1], [x2, y2], ..., [xn, yn]].

        Parameters
        ----------
        positions : np.ndarray
            Array of positions along which to sample the grid. Units are spherical decimal degrees.

        Returns
        -------
        indices : np.ndarray
            Indices of the grid node positions comprising the transect.
        distance : np.ndarray
            Distance (in metres) along the transect.

        """

        # Since we're letting the transect positions be specified in spherical coordinates and we want to return the
        # distance in metres, we need to do this in two steps: first, find the indices of the transect from the
        # spherical grid, and secondly, find the distance in metres from the cartesian grid.
        indices, _ = line_sample(self.grid.lon, self.grid.lat, positions)
        distance = np.cumsum([0] + [np.hypot(self.grid.x[i + 1] - self.grid.x[i], self.grid.y[i + 1] - self.grid.y[i]) for i in indices[:-1]])

        return indices, distance

    def horizontal_transect_elements(self, positions):
        """
        Extract element IDs along a line defined by `positions' [[x1, y1], [x2, y2], ..., [xn, yn]].

        Parameters
        ----------
        positions : np.ndarray
            Array of positions along which to sample the grid. Units are spherical decimal degrees.

        Returns
        -------
        indices : np.ndarray
            Indices of the grid element positions comprising the transect.
        distance : np.ndarray
            Distance (in metres) along the transect.

        """

        indices, distance = element_sample(self.grid.lonc, self.grid.latc, positions)

        return indices, distance

    def subset_domain(self, polygon=None):
        """
        Subset the current model grid interactively with a given polygon. Coordinates in `polygon' must be in the
        same system as `x' and `y'.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon, optional
            If given, use this polygon to use to clip the given domain. If omitted, subsetting is interactive.

        Returns
        -------
        nodes : np.ndarray
            List of the node IDs from the original `triangles' array within the given polygon.
        elements : np.ndarray
            List of the element IDs from the original `triangles' array within the given polygon.
        triangles : np.ndarray
            The reduced triangulation.

        """

        return subset_domain(self.grid.lon, self.grid.lat, self.grid.triangles, polygon)

    def calculate_areas(self):
        """
        Calculate the area of each element for the current grid.

        Provides
        --------
        self.grid.areas : np.ndarray
            The area of each triangular element in the grid.

        Notes
        -----
        This differs from self.calculate_control_area_and_volume which provides what's needed for integrating fields
        in the domain. This simply calculates the area of each triangle in the domain.

        """

        triangles = self.grid.triangles
        x = self.grid.x
        y = self.grid.y
        self.grid.areas = get_area(np.asarray((x[triangles[:, 0]], y[triangles[:, 0]])).T,
                                   np.asarray((x[triangles[:, 1]], y[triangles[:, 1]])).T,
                                   np.asarray((x[triangles[:, 2]], y[triangles[:, 2]])).T)

    def calculate_control_area_and_volume(self):
        """
        This calculates the surface area of individual control volumes consisted of triangles with a common node point.

        Parameters
        ----------
        x, y : ndarray
            Node positions
        tri : ndarray
            Triangulation table for the unstructured grid.
        node_control : bool
            Set to False to disable calculation of node control volumes. Defaults to True.
        element_control : bool
            Set to False to disable calculation of element control volumes. Defaults to True.
        noisy : bool
            Set to True to enable verbose output.

        Provides
        --------
        self.grid.art1 : ndarray
            Area of interior control volume (for node value integration)
        self.grid.art2 : ndarray
            Sum area of all cells around each node.

        Notes
        -----
        This is a python reimplementation of the FVCOM function CELL_AREA in cell_area.F. Whilst the reimplementation is
        coded with efficiency in mind (the calculations occur in parallel), this is still slow for large grids. Please be
        patient!

        """

        self.grid.art1, self.grid.art2 = control_volumes(self.x, self.y, self.triangles)

    def calculate_element_lengths(self):
        """
        Given a list of triangle nodes, calculate the length of each side of each
        triangle and return as an array of lengths. Units are in the original input
        units (no conversion from lat/long to metres, for example).

        The arrays triangles, x and y can be created by running read_sms_mesh(),
        read_fvcom_mesh() or read_mike_mesh() on a given SMS, FVCOM or MIKE grid
        file.

        Parameters
        ----------
        triangles : ndarray
            Integer array of shape (ntri, 3). Each triangle is composed of
            three points and this contains the three node numbers (stored in
            nodes) which refer to the coordinates in X and Y (see below).
        x, y : ndarray
            Coordinates of each grid node.

        Provides
        --------
        self.grid.lengths : np.ndarray
            The lengths of each vertex in each triangle in the grid.

        """

        self.grid.lengths = element_side_lengths(self.triangles, self.x, self.y)

    def gradient(self, field):
        """
        Returns the gradient of `z' defined on the irregular mesh with Delaunay
        triangulation `t'. `dx' corresponds to the partial derivative dZ/dX,
        and `dy' corresponds to the partial derivative dZ/dY.

        Parameters
        ----------
        x, y, z : array_like
            Horizontal (`x' and `y') positions and vertical position (`z').
        t : array_like, optional
            Connectivity table for the grid. If omitted, one will be calculated
            automatically.

        Returns
        -------
        dx, dy : ndarray
            `dx' corresponds to the partial derivative dZ/dX, and `dy'
            corresponds to the partial derivative dZ/dY.
        """

        dx, dy = trigradient(self.x, self.y, field, self.triangles)

        return dx, dy

    def to_nodes(self, field):
        """
        Calculate a nodal value based on the average value for the elements
        of which it is a part. This necessarily involves an average, so the
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

        return elems2nodes(field, self.triangles)

    def to_elements(self, field):
        """
        Calculate an element-centre value based on the average value for the
        nodes from which it is formed. This involves an average, so the
        conversion from nodes to elements cannot be reversed without smoothing.

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

        return nodes2elems(field, self.triangles)


    def info(self):
        """
        Print out some salient information on the currently loaded grid.

        """
        print(f'Nodes: {self.dims.node}')
        print(f'Elements: {self.dims.nele}')
        if hasattr(self.grid, 'open_boundary'):
            print(f'Open boundaries: {len(self.grid.open_boundary)}')
        print(f'Native grid coordinates: {self.grid.native_coordinates}')
        if self.grid.native_coordinates == 'cartesian':
            print(f'Native grid zone: {self.grid.zone}')
        print(f'West/east/south/north: {"/".join(map(str, self.grid.bounding_box))}')


def mp_interp_func(input):
    """
    Pass me to a multiprocessing.Pool().map() to quickly interpolate 2D data with LinearNDInterpolator.

    Parameters
    ----------
    input : tuple
        Input data to interpolate (lon, lat, data, x, y), where (x, y) are the positions onto which you would like to
        interpolate the regularly gridded data (lon, lat, data).

    Returns
    -------
    interp : np.ndarray
        The input data `data' interpolated onto the positions (x, y).

    """

    lon, lat, data, x, y = input
    interp = LinearNDInterpolator((lon, lat), data)
    return interp((x, y))


class OpenBoundary(object):
    """
    FVCOM grid open boundary object. Handles reading, writing and interpolating.

    Not sure this is the right place for this. Might be better placed in PyFVCOM.preproc. Also, this should probably
    be a superclass of Nest as an open boundary is just a special case of a PyFVCOM.preproc.Nest (one with 0 levels,
    essentially).

    """

    def __init__(self, ids, mode='nodes'):
        """
        Given a set of open boundary nodes, initialise a new open boundary object with relevant arrays.

        Parameters
        ----------
        ids : np.ndarray
            Array of unstructured grid IDs representing a model open boundary.
        mode : str, optional
            Specify whether the IDs given are node ('nodes') or element-centre ('elements') IDs. Defaults to 'nodes'.

        Provides
        --------
        add_sponge_layer : add a sponge layer to the open boundary.
        add_type : give the open boundary a type (see mod_obcs.F and Table 6.1 in the FVCOM manual)
        add_tpxo_tides : predict tidal elevations/currents along the open boundary from TPXO harmonics.
        add_fvcom_tides : predict tidal elevations/currents along the open boundary from FVCOM harmonics.
        add_nested_forcing : interpolate some regularly gridded data onto the open boundary.

        """

        self._debug = False

        self.nodes = None
        self.elements = None
        # Silently convert IDs from numpy arrays to lists. If the first and last node ID are the same, drop the last
        # one too to match the behaviour of the MATLAB preprocessing tools (and I'm sure there's an actual reason,
        # I just can't remember it at the moment).
        if ids[0] == ids[-1]:
            ids = ids[:-1]
        if mode == 'nodes':
            try:
                ids = ids.tolist()
            except AttributeError:
                pass
            self.nodes = ids
        else:
            try:
                ids = ids.tolist()
            except AttributeError:
                pass
            self.elements = ids
        self.sponge_coefficient = None
        self.sponge_radius = None
        self.type = None
        # Add fields which get populated if this open boundary is made a part of a nested region.
        self.weight_node = None
        self.weight_element = None
        # These get added to by PyFVCOM.preproc.Model and are used in the tide and nest functions below.
        self.tide = PassiveStore()
        self.grid = PassiveStore()
        self.sigma = PassiveStore()
        self.time = PassiveStore()
        self.data = PassiveStore()

    def __iter__(self):
        return (a for a in self.__dict__.keys() if not a.startswith('_'))

    def add_sponge_layer(self, radius, coefficient):
        """
        Add a sponge layer. If radius or coefficient are floats, apply the same value to all nodes.

        Parameters
        ----------
        radius : float, list, np.ndarray
            The sponge layer radius at the given nodes.
        coefficient : float, list, np.ndarray
            The sponge layer coefficient at the given nodes.

        Provides
        --------
        sponge_radius : np.ndarray
            Sponge radii for the nodes in the boundary.
        sponge_coefficient
            Sponge coefficients for the nodes in the boundary.

        """

        if isinstance(radius, (float, int)):
            radius = np.repeat(radius, np.shape(self.nodes))
        if isinstance(coefficient, (float, int)):
            coefficient = np.repeat(coefficient, np.shape(self.nodes))

        self.sponge_radius = radius

        self.sponge_coefficient = coefficient

    def add_type(self, obc_type=1):
        """
        Add an FVCOM open boundary type to the current boundary.

        Parameters
        ----------
        obc_type : int, optional
            The open boundary type. See the types listed in mod_obcs.F, lines 29 to 49, reproduced in the notes below
            for convenience. Defaults to 1 (prescribed surface elevation).

        Provides
        --------
        Populates the self.boundaries open boundary objects with the relevant `type' attribute.

        Notes
        -----
        These are the valid values as lifted from mod_obcs.F.

        TYPE_OBC =  1: Surface Elevation Specified (Tidal Forcing) (ASL)
        TYPE_OBC =  2: As TYPE_OBC=1 and non-linear flux for current at open boundary
        TYPE_OBC =  3: Zero Surface Elevation Boundary Condition (ASL_CLP)
        TYPE_OBC =  4: As TYPE_OBC=3 and non-linear flux for current at open boundary
        TYPE_OBC =  5: Gravity-wave radiation implicit open boundary condition (GWI)
        TYPE_OBC =  6: As TYPE_OBC=5 and non-linear flux for current at open boundary
        TYPE_OBC =  7: Blumberg and khanta implicit open boundary condition (BKI)
        TYPE_OBC =  8: As TYPE_OBC=7 and non-linear flux for current at open boundary
        TYPE_OBC =  9: Orlanski radiation explicit open boundary condition (ORE)
        TYPE_OBC = 10: As TYPE_OBC=9 and non-linear flux for current at open boundary

        """

        # Feels a bit ridiculous having a whole method for this...
        setattr(self, 'type', obc_type)

    def add_tpxo_tides(self, tpxo_harmonics, predict='zeta', interval=1 / 24, constituents=['M2'], serial=False,
                       interp_method='linear', pool_size=None, noisy=False):
        """
        Add TPXO tides at the open boundary nodes.

        Parameters
        ----------
        tpxo_harmonics : str, pathlib.Path
            Path to the TPXO harmonics netCDF file to use.
        predict : str, optional
            Type of data to predict. Select 'zeta' (default), 'u' or 'v'.
        interval : str, optional
            Time sampling interval in days. Defaults to 1 hour.
        constituents : list, optional
            List of constituent names to use in UTide.reconstruct. Defaults to ['M2'].
        serial : bool, optional
            Run in serial rather than parallel. Defaults to parallel.
        interp_method : str
            The interpolation method to use. Defaults to 'linear'. Passed to scipy.interp.RegularGridInterpolator,
            so choose any valid one for that.
        pool_size : int, optional
            Specify number of processes for parallel run. By default it uses all available.
        noisy : bool, optional
            Set to True to enable some sort of progress output. Defaults to False.

        """

        if not hasattr(self.time, 'start'):
            raise AttributeError('No time data have been added to this OpenBoundary object, so we cannot predict tides.')
        self.tide.time = date_range(self.time.start - relativedelta(days=1),
                                    self.time.end + relativedelta(days=1),
                                    inc=interval)

        constituent_name = 'con'
        if predict == 'zeta':
            amplitude_name, phase_name = 'ha', 'hp'
            x, y = copy.copy(self.grid.lon), self.grid.lat
            lon_name, lat_name = 'lon_z', 'lat_z'
        elif predict == 'u':
            amplitude_name, phase_name = 'ua', 'up'
            x, y = copy.copy(self.grid.lonc), self.grid.latc
            lon_name, lat_name = 'lon_u', 'lat_u'
        elif predict == 'v':
            amplitude_name, phase_name = 'va', 'vp'
            x, y = copy.copy(self.grid.lonc), self.grid.latc
            lon_name, lat_name = 'lon_v', 'lat_v'

        names = {'amplitude_name': amplitude_name,
                 'phase_name': phase_name,
                 'lon_name': lon_name,
                 'lat_name': lat_name,
                 'constituent_name': constituent_name}
        harmonics_lon, harmonics_lat, amplitudes, phases, available_constituents = self._load_harmonics(tpxo_harmonics,
                                                                                                        constituents,
                                                                                                        names)
        interpolated_amplitudes, interpolated_phases = self._interpolate_tpxo_harmonics(x, y,
                                                                                        amplitudes, phases,
                                                                                        harmonics_lon, harmonics_lat,
                                                                                        interp_method=interp_method)

        self.tide.constituents = available_constituents

        # Predict the tides
        results = self._prepare_tides(interpolated_amplitudes, interpolated_phases, y, serial, pool_size)

        # Dump the results into the object.
        setattr(self.tide, predict, np.asarray(results).T)  # put the time dimension first, space last.

    def add_fvcom_tides(self, fvcom_harmonics, predict='zeta', interval=1/24, constituents=['M2'], serial=False,
                        pool_size=None, noisy=False):
        """
        Add FVCOM-derived tides at the open boundary nodes.

        Parameters
        ----------
        fvcom_harmonics : str, pathlib.Path
            Path to the FVCOM harmonics netCDF file to use.
        predict : str, optional
            Type of data to predict. Select 'zeta' (default), 'u' or 'v'.
        interval : str, optional
            Time sampling interval in days. Defaults to 1 hour.
        constituents : list, optional
            List of constituent names to use in UTide.reconstruct. Defaults to ['M2'].
        serial : bool, optional
            Run in serial rather than parallel. Defaults to parallel.
        pool_size : int, optional
            Specify number of processes for parallel run. By default it uses all available.
        noisy : bool, optional
            Set to True to enable some sort of progress output. Defaults to False.

        """

        if not hasattr(self.time, 'start'):
            raise AttributeError('No time data have been added to this OpenBoundary object, so we cannot predict tides.')
        self.tide.time = date_range(self.time.start - relativedelta(days=1),
                                    self.time.end + relativedelta(days=1),
                                    inc=interval)

        constituent_name = 'z_const_names'

        if predict == 'zeta':
            amplitude_name, phase_name = 'z_amp', 'z_phase'
            lon_name, lat_name = 'lon', 'lat'
            x, y = copy.copy(self.grid.lon), self.grid.lat
        elif predict == 'u':
            amplitude_name, phase_name = 'u_amp', 'u_phase'
            lon_name, lat_name = 'lonc', 'latc'
            x, y = copy.copy(self.grid.lonc), self.grid.latc
        elif predict == 'v':
            amplitude_name, phase_name = 'v_amp', 'v_phase'
            lon_name, lat_name = 'lonc', 'latc'
            x, y = copy.copy(self.grid.lonc), self.grid.latc
        elif predict == 'ua':
            amplitude_name, phase_name = 'ua_amp', 'ua_phase'
            lon_name, lat_name = 'lonc', 'latc'
            x, y = copy.copy(self.grid.lonc), self.grid.latc
        elif predict == 'va':
            amplitude_name, phase_name = 'va_amp', 'va_phase'
            lon_name, lat_name = 'lonc', 'latc'
            x, y = copy.copy(self.grid.lonc), self.grid.latc

        names = {'amplitude_name': amplitude_name,
                 'phase_name': phase_name,
                 'lon_name': lon_name,
                 'lat_name': lat_name,
                 'constituent_name': constituent_name}
        harmonics_lon, harmonics_lat, amplitudes, phases, available_constituents = self._load_harmonics(fvcom_harmonics,
                                                                                                        constituents,
                                                                                                        names)
        if predict in ['zeta', 'ua', 'va']:
            amplitudes = amplitudes[:, np.newaxis, :]
            phases = phases[:, np.newaxis, :]

        results = []
        for i in np.arange(amplitudes.shape[1]):
            locations_match, match_indices = self._match_coords(np.asarray([x, y]).T, np.asarray([harmonics_lon, harmonics_lat]).T)
            if locations_match:
                if noisy:
                    print('Coords match, skipping interpolation')
                interpolated_amplitudes = amplitudes[:, i, match_indices].T
                interpolated_phases = phases[:, i, match_indices].T
            else:
                interpolated_amplitudes, interpolated_phases = self._interpolate_fvcom_harmonics(x, y,
                                                                                                 amplitudes[:, i, :],
                                                                                                 phases[:, i, :],
                                                                                                 harmonics_lon,
                                                                                                 harmonics_lat,
                                                                                                 pool_size)
            self.tide.constituents = available_constituents

            # Predict the tides
            results.append(np.asarray(self._prepare_tides(interpolated_amplitudes, interpolated_phases,
                                                          y, serial, pool_size, noisy)).T)

        # Dump the results into the object. Put the time dimension first, space last.
        setattr(self.tide, predict, np.squeeze(np.transpose(np.asarray(results), (1, 0, 2))))

    def _prepare_tides(self, amplitudes, phases, latitudes, serial=False, pool_size=None, noisy=False):
        # Prepare the UTide inputs for the constituents we've loaded.
        const_idx = np.asarray([ut_constants['const']['name'].tolist().index(i) for i in self.tide.constituents])
        frq = ut_constants['const']['freq'][const_idx]

        coef = Bunch(name=self.tide.constituents, mean=0, slope=0)
        coef['aux'] = Bunch(reftime=729572.47916666674, lind=const_idx, frq=frq)
        coef['aux']['opt'] = Bunch(twodim=False, nodsatlint=False, nodsatnone=False,
                                   gwchlint=False, gwchnone=False, notrend=True, prefilt=[])

        # Prepare the time data for predicting the time series. UTide needs MATLAB times.
        times = mtime(self.tide.time)
        args = [(latitudes[i], times, coef, amplitudes[i], phases[i], noisy) for i in range(len(latitudes))]
        if serial:
            results = []
            for arg in args:
                results.append(self._predict_tide(arg))
        else:
            if pool_size is None:
                pool = multiprocessing.Pool()
            else:
                pool = multiprocessing.Pool(pool_size)
            results = pool.map(self._predict_tide, args)
            pool.close()

        return results

    @staticmethod
    def _load_harmonics(harmonics, constituents, names):
        """
        Load the given variables from the given file.

        Parameters
        ----------
        harmonics : str
            The file to load.
        constituents : list
            The list of tidal constituents to load.
        names : dict
            Dictionary with the variables names:
                amplitude_name - amplitude data
                phase_name - phase data
                lon_name - longitude data
                lat_name - latitude data
                constituent_name - constituent names

        Returns
        -------
        amplitudes : np.ndarray
            The amplitudes for the given constituents.
        phases : np.darray
            The amplitudes for the given constituents.
        fvcom_constituents : list
            The constituents which have been requested that actually exist in the harmonics netCDF file.

        """

        with Dataset(str(harmonics), 'r') as tides:
            const = [''.join(i).upper().strip() for i in tides.variables[names['constituent_name']][:].astype(str)]
            # If we've been given constituents that aren't in the harmonics data, just find the indices we do have.
            cidx = [const.index(i) for i in constituents if i in const]
            # Save the names of the constituents we've actually used.
            available_constituents = [constituents[i] for i in cidx]

            harmonics_lon = tides.variables[names['lon_name']][:]
            harmonics_lat = tides.variables[names['lat_name']][:]

            amplitude_shape = tides.variables[names['amplitude_name']][:].shape
            if amplitude_shape[0] == len(const):
                amplitudes = tides.variables[names['amplitude_name']][cidx, ...]
                phases = tides.variables[names['phase_name']][cidx, ...]
            elif amplitude_shape[-1] == len(const):
                amplitudes = tides.variables[names['amplitude_name']][..., cidx].T
                phases = tides.variables[names['phase_name']][..., cidx].T

        return harmonics_lon, harmonics_lat, amplitudes, phases, available_constituents

    @staticmethod
    def _match_coords(pts_1, pts_2, epsilon=10):
        """
        Check whether the lat lon points in the array pts_1 all within an epsilon of some point in pts_2

        Parameters
        ----------
        pts_1 : Nx2 array
            The lon/lat of points to check
        pts_2 : Mx2 array
            The lon/lat of points to check against
        epsilon : float, optional
            The distance within which a point in pts_1 has to be to a point in pts_2 to count as matching.
            Given that the positions are in lon/lat, the method haversine distance, and the resolution of FVCOM
            meshes 10m should be reasonable in most cases.

        Returns
        -------
        is_matched : bool
            True if every point in pts_1 lies within an epsilon of some point in pts_2
        indices : N array
            The indices of the matching point in pts_2 for each point in pts_1
        """

        is_matched = np.zeros(len(pts_1), dtype=bool)
        match_indices = np.ones(len(pts_1))*-1
        for i, this_pt in enumerate(pts_1):
            dists_m = haversine_distance(this_pt, pts_2.T) * 1000
            if np.min(dists_m) < epsilon:
                is_matched[i] = True
                match_indices[i] = np.argmin(dists_m)
        return np.all(is_matched), np.asarray(match_indices, dtype=int)

    @staticmethod
    def _interpolate_tpxo_harmonics(x, y, amp_data, phase_data, harmonics_lon, harmonics_lat, interp_method):
        """
        Interpolate from the harmonics data onto the current open boundary positions.

        Parameters
        ----------
        x, y : np.ndarray
            The positions at which to interpolate the harmonics data.
        amp_data, phase_data : np.ndarray
            The tidal constituent amplitude and phase data.
        harmonics_lon, harmonics_lat : np.ndarray
            The positions of the harmonics data (1D).

        Returns
        -------
        interpolated_amplitude, interpolated_phase : np.ndarray
            The interpolated data.

        """

        if np.ndim(harmonics_lon) != 1 and np.ndim(harmonics_lat) != 1:
            # Creating the RegularGridInterpolator object will fail if we've got 2D position arrays with a
            # ValueError. So, this is fragile, but try first assuming we've got the right shape arrays and try again
            # if that fails with the unique coordinates in the relevant position arrays.
            warn('Harmonics are given as 2D arrays: trying to convert to 1D for the interpolation.')
            harmonics_lon = np.unique(harmonics_lon)
            harmonics_lat = np.unique(harmonics_lat)

        # Since interpolating phase directly is a bad idea (cos of the 360 -> 0 degree thing) convert to vectors first
        harmonics_u, harmonics_v = pol2cart(amp_data, phase_data, degrees=True) 

        # Make a dummy first dimension since we need it for the RegularGridInterpolator but don't actually
        # interpolated along it.
        c_data = np.arange(amp_data.shape[0])
        u_interp = RegularGridInterpolator((c_data, harmonics_lon, harmonics_lat), harmonics_u, method=interp_method, fill_value=None)
        v_interp = RegularGridInterpolator((c_data, harmonics_lon, harmonics_lat), harmonics_v, method=interp_method, fill_value=None)

        # Fix our input position longitudes to be in the 0-360 range to match the harmonics data range,
        # if necessary.
        if harmonics_lon.min() >= 0:
            x[x < 0] += 360

        # Since everything should be in the 0-360 range, stuff which is between the Greenwich meridian and the
        # first harmonics data point is now outside the interpolation domain, which yields an error since
        # RegularGridInterpolator won't tolerate data outside the defined bounding box. As such, we need to
        # squeeze the interpolation locations to the range of the open boundary positions.
        if x.min() < harmonics_lon.min():
            harmonics_lon[harmonics_lon == harmonics_lon.min()] = x.min()

        # Make our boundary positions suitable for interpolation with a RegularGridInterpolator.
        xx = np.tile(x, [amp_data.shape[0], 1])
        yy = np.tile(y, [amp_data.shape[0], 1])
        ccidx = np.tile(c_data, [len(x), 1]).T
        u = u_interp((ccidx, xx, yy)).T
        v = v_interp((ccidx, xx, yy)).T

        # And convert back to amplitude and phase
        amplitudes, phases = cart2pol(u, v, degrees=True)        

        return amplitudes, phases

    @staticmethod
    def _interpolate_fvcom_harmonics(x, y, amp_data, phase_data, harmonics_lon, harmonics_lat, pool_size=None):
        """
        Interpolate from the harmonics data onto the current open boundary positions.

        Parameters
        ----------
        x, y : np.ndarray
            The positions at which to interpolate the harmonics data.
        amp_data, phase_data : np.ndarray
            The tidal constituent amplitude and phase data.
        harmonics_lon, harmonics_lat : np.ndarray
            The positions of the harmonics data.

        Returns
        -------
        interpolated_amplitude, interpolated_phase : np.ndarray
            The interpolated data.

        """

        # Fix our input position longitudes to be in the 0-360 range to match the harmonics data range, if necessary.
        if harmonics_lon.min() >= 0:
            x[x < 0] += 360

        # Since everything should be in the 0-360 range, stuff which is between the Greenwich meridian and the first
        # harmonics data point is now outside the interpolation domain, which yields an error since
        # RegularGridInterpolator won't tolerate data outside the defined bounding box. As such, we need to squeeze
        # the interpolation locations to the range of the open boundary positions.
        if x.min() < harmonics_lon.min():
            harmonics_lon[harmonics_lon == harmonics_lon.min()] = x.min()

        # I can't wrap my head around the n-dimensional unstructured interpolation tools (Rdf, griddata etc.),
        # so just loop through each constituent and do a 2D interpolation.

        if pool_size is None:
            pool = multiprocessing.Pool()
        else:
            pool = multiprocessing.Pool(pool_size)

        # Use pol2cart/cart2pol like we do for the TPXO components.
        harmonics_u, harmonics_v = pol2cart(amp_data, phase_data, degrees=True)
        inputs = [(harmonics_lon, harmonics_lat, harmonics_u[i], x, y) for i in range(harmonics_u.shape[0])]
        harmonics_u = np.asarray(pool.map(mp_interp_func, inputs))
        inputs = [(harmonics_lon, harmonics_lat, harmonics_v[i], x, y) for i in range(harmonics_v.shape[0])]
        harmonics_v = np.asarray(pool.map(mp_interp_func, inputs))
        pool.close()
        # Map back to amplitude and phase.
        amplitudes, phases = cart2pol(harmonics_u, harmonics_v, degrees=True)

        # Transpose so space is first, constituents second (expected by self._predict_tide).
        return amplitudes.T, phases.T

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
        times : np.ndarray
            Array of matplotlib datenums (see `matplotlib.dates.num2date').
        coef : utide.utilities.Bunch
            Configuration options for utide.
        amplitudes : np.ndarray
            Amplitude of the relevant constituents shaped [nconst].
        phases : np.ndarray
            Array of the phase of the relevant constituents shaped [nconst].
        noisy : bool
            Set to true to enable verbose output. Defaults to False (no output).

        Returns
        -------
        zeta : np.ndarray
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
        pred = reconstruct(times, coef, verbose=noisy)
        zeta = pred['h']

        return zeta

    def add_nested_forcing(self, fvcom_name, coarse_name, coarse, interval=1, constrain_coordinates=False,
                           mode='nodes', tide_adjust=False, verbose=False):
        """
        Interpolate the given data onto the open boundary nodes for the period from `self.time.start' to
        `self.time.end'.

        Parameters
        ----------
        fvcom_name : str
            The data field name to add to the nest object which will be written to netCDF for FVCOM.
        coarse_name : str
            The data field name to use from the coarse object.
        coarse : RegularReader
            The regularly gridded data to interpolate onto the open boundary nodes. This must include time, lon,
            lat and depth data as well as the time series to interpolate (4D volume [time, depth, lat, lon]).
        interval : float, optional
            Time sampling interval in days. Defaults to 1 day.
        constrain_coordinates : bool, optional
            Set to True to constrain the open boundary coordinates (lon, lat, depth) to the supplied coarse data.
            This essentially squashes the open boundary to fit inside the coarse data and is, therefore, a bit of a
            fudge! Defaults to False.
        mode : bool, optional
            Set to 'nodes' to interpolate onto the open boundary node positions or 'elements' for the elements for
            z-level data. For 2D data, set to 'surface' (interpolates to the node positions ignoring depth
            coordinates). Also supported are 'sigma_nodes' and `sigma_elements' which means we have spatially (and
            optionally temporally) varying water depths (i.e. sigma layers rather than z-levels). Defaults to 'nodes'.
        tide_adjust : bool, optional
            Some nested forcing doesn't include tidal components and these have to be added from predictions using
            harmonics. With this set to true the interpolated forcing has the tidal component (required to already
            exist in self.tide) added to the final data.
        verbose : bool, optional
            Set to True to enable verbose output. Defaults to False (no verbose output).

        """

        # Check we have what we need.
        raise_error = False
        if mode == 'nodes':
            if not np.any(self.nodes):
                if verbose:
                    print(f'No {mode} on which to interpolate on this boundary')
                return
            if not hasattr(self.sigma, 'layers'):
                raise_error = True
        elif mode == 'elements':
            if not hasattr(self.sigma, 'layers_center'):
                raise_error = True
            if not np.any(self.elements):
                if verbose:
                    print(f'No {mode} on which to interpolate on this boundary')
                return

        if raise_error:
            raise AttributeError('Add vertical sigma coordinates in order to interpolate forcing along this boundary.')

        # Populate the time data. Why did I put the time data in here rather than self.time? This is annoying.
        self.data.time = PassiveStore()
        self.data.time.interval = interval
        self.data.time.datetime = date_range(self.time.start, self.time.end, inc=interval)
        self.data.time.time = date2num(getattr(self.data.time, 'datetime'), units='days since 1858-11-17 00:00:00')
        self.data.time.Itime = np.floor(getattr(self.data.time, 'time'))  # integer Modified Julian Days
        self.data.time.Itime2 = (getattr(self.data.time, 'time') - getattr(self.data.time, 'Itime')) * 24 * 60 * 60 * 1000  # milliseconds since midnight
        self.data.time.Times = [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in getattr(self.data.time, 'datetime')]

        if 'elements' in mode:
            boundary_points = self.elements
            x = self.grid.lonc
            y = self.grid.latc
            # Keep positive down depths.
            z = -self.sigma.layers_center_z
        else:
            boundary_points = self.nodes
            x = self.grid.lon
            y = self.grid.lat
            # Keep positive down depths.
            z = -self.sigma.layers_z

        if constrain_coordinates:
            x[x < coarse.grid.lon.min()] = coarse.grid.lon.min()
            x[x > coarse.grid.lon.max()] = coarse.grid.lon.max()
            y[y < coarse.grid.lat.min()] = coarse.grid.lat.min()
            y[y > coarse.grid.lat.max()] = coarse.grid.lat.max()

            # Internal landmasses also need to be dealt with, so test if a point lies within the mask of the grid and
            # move it to the nearest in grid point if so.
            if not mode == 'surface':
                land_mask = getattr(coarse.data, coarse_name)[0, ...].mask[0, :, :]
            else:
                land_mask = getattr(coarse.data, coarse_name)[0, ...].mask

            sea_points = np.ones(land_mask.shape)
            sea_points[land_mask] = np.nan

            ft_sea = RegularGridInterpolator((coarse.grid.lat, coarse.grid.lon), sea_points, method='linear', fill_value=np.nan)
            internal_points = np.isnan(ft_sea(np.asarray([y, x]).T))

            if np.any(internal_points):
                xv, yv = np.meshgrid(coarse.grid.lon, coarse.grid.lat)
                valid_ll = np.asarray([x[~internal_points], y[~internal_points]]).T
                for this_ind in np.where(internal_points)[0]:
                    nearest_valid_ind = np.argmin((valid_ll[:, 0] - x[this_ind])**2 + (valid_ll[:, 1] - y[this_ind])**2)
                    x[this_ind] = valid_ll[nearest_valid_ind, 0]
                    y[this_ind] = valid_ll[nearest_valid_ind, 1]

            # The depth data work differently as we need to squeeze each FVCOM water column into the available coarse
            # data. The only way to do this is to adjust each FVCOM water column in turn by comparing with the
            # closest coarse depth.
            if mode != 'surface':
                coarse_depths = np.tile(coarse.grid.depth, [coarse.dims.lat, coarse.dims.lon, 1]).transpose(2, 0, 1)
                coarse_depths = np.ma.masked_array(coarse_depths, mask=getattr(coarse.data, coarse_name)[0, ...].mask)
                coarse_depths = np.max(coarse_depths, axis=0)
            
                # Find any places where only the zero depth layer exists and copy down 
                zero_depth_water = np.where(np.logical_and(coarse_depths == 0, ~coarse_depths.mask))
                if zero_depth_water[0].size:
                    data_mod = getattr(coarse.data, coarse_name)
                    data_mod[:, 1, zero_depth_water[0], zero_depth_water[1]] = data_mod[:, 0, zero_depth_water[0], zero_depth_water[1]]
                    data_mod.mask[:, 1, zero_depth_water[0], zero_depth_water[1]] = False
                    setattr(coarse.data, coarse_name, data_mod) # Probably isn't needed cos pointers but for clarity

                coarse_depths = np.ma.filled(coarse_depths, 0)

                # Go through each open boundary position and if its depth is deeper than the closest coarse data,
                # squash the open boundary water column into the coarse water column.
                for idx, node in enumerate(zip(x, y, z)):
                    nearest_lon_ind = np.argmin((coarse.grid.lon - node[0])**2)
                    nearest_lat_ind = np.argmin((coarse.grid.lat - node[1])**2)

                    if node[0] < coarse.grid.lon[nearest_lon_ind]:
                        nearest_lon_ind = [nearest_lon_ind -1, nearest_lon_ind, nearest_lon_ind -1, nearest_lon_ind]
                    else:
                        nearest_lon_ind = [nearest_lon_ind, nearest_lon_ind + 1, nearest_lon_ind, nearest_lon_ind + 1]

                    if node[1] < coarse.grid.lat[nearest_lat_ind]:
                        nearest_lat_ind = [nearest_lat_ind -1, nearest_lat_ind -1, nearest_lat_ind, nearest_lat_ind]
                    else:
                        nearest_lat_ind = [nearest_lat_ind, nearest_lat_ind, nearest_lat_ind + 1, nearest_lat_ind + 1]

                    grid_depth = np.min(coarse_depths[nearest_lat_ind, nearest_lon_ind])

                    if grid_depth < node[2].max():
                        # Squash the FVCOM water column into the coarse water column.
                        z[idx, :] = (node[2] / node[2].max()) * grid_depth
                # Fix all depths which are shallower than the shallowest coarse depth. This is more straightforward as
                # it's a single minimum across all the open boundary positions.
                z[z < coarse.grid.depth.min()] = coarse.grid.depth.min()

        nt = len(self.data.time.time)
        nx = len(boundary_points)
        nz = z.shape[-1]

        if verbose:
            print(f'Interpolating {nt} times, {nz} vertical layers and {nx} points')

        # Make arrays of lon, lat, depth and time for non-sigma interpolation. Need to make the coordinates match the
        # coarse data shape and then flatten the lot. We should be able to do the interpolation in one shot this way,
        # but we have to be careful our coarse data covers our model domain (space and time).
        if mode == 'surface':
            if verbose:
                print('Interpolating surface data...', end=' ')

            # We should use np.meshgrid here instead of all this tiling business.
            boundary_grid = np.array((np.tile(self.data.time.time, [nx, 1]).T.ravel(),
                                      np.tile(y, [nt, 1]).transpose(0, 1).ravel(),
                                      np.tile(x, [nt, 1]).transpose(0, 1).ravel())).T
            ft = RegularGridInterpolator((coarse.time.time, coarse.grid.lat, coarse.grid.lon),
                                         getattr(coarse.data, coarse_name), method='linear', fill_value=np.nan)
            # Reshape the results to match the un-ravelled boundary_grid array.
            interpolated_coarse_data = ft(boundary_grid).reshape([nt, -1])
        elif 'sigma' in mode:
            if verbose:
                print('Interpolating sigma data...', end=' ')

            nt = coarse.dims.time  # rename!
            interp_args = [(boundary_points, x, y, self.sigma.layers_z, coarse, coarse_name, self._debug, t) for t in np.arange(nt)]
            if hasattr(coarse, 'ds'):
                coarse.ds.close()
                delattr(coarse, 'ds')
            pool = multiprocessing.Pool()
            results = pool.map(self._brute_force_interpolator, interp_args)

            # Now we have those data interpolated in space (horizontal and vertical), interpolate to match in time.
            interp_args = [(coarse.time.time, j, self.data.time.time) for i in np.asarray(results).T for j in i]
            results = pool.map(self._interpolate_in_time, interp_args)
            pool.close()

            # Reshape and transpose to be the correct size for writing to netCDF (time, depth, node).
            interpolated_coarse_data = np.asarray(results).reshape(nz, nx, -1).transpose(2, 0, 1)
        else:
            if verbose:
                print('Interpolating z-level data...', end=' ')
            # Assume it's z-level data (e.g. HYCOM, CMEMS). We should use np.meshgrid here instead of all this tiling
            # business.
            boundary_grid = np.array((np.tile(self.data.time.time, [nx, nz, 1]).T.ravel(),
                                      np.tile(z.T, [nt, 1, 1]).ravel(),
                                      np.tile(y, [nz, nt, 1]).transpose(1, 0, 2).ravel(),
                                      np.tile(x, [nz, nt, 1]).transpose(1, 0, 2).ravel())).T
            ft = RegularGridInterpolator((coarse.time.time, coarse.grid.depth, coarse.grid.lat, coarse.grid.lon),
                                         np.ma.filled(getattr(coarse.data, coarse_name), np.nan), method='linear',
                                         fill_value=np.nan)
            # Reshape the results to match the un-ravelled boundary_grid array (time, depth, node).
            interpolated_coarse_data = ft(boundary_grid).reshape([nt, nz, -1])

        if tide_adjust and fvcom_name in ['u', 'v', 'ua', 'va']:
            interpolated_coarse_data = interpolated_coarse_data + getattr(self.tide, fvcom_name)

        # Drop the interpolated data into the data object.
        setattr(self.data, fvcom_name, interpolated_coarse_data)

        if verbose:
            print('done.')

    @staticmethod
    def _brute_force_interpolator(args):
        """
        Interpolate a given time of coarse data into the current open boundary node positions and times.

        The name stems from the fact we simply iterate through all the points (horizontal and vertical) in the
        current boundary rather than using LinearNDInterpolator. This is because the creation of a
        LinearNDInterpolator object for a 4D set of points is hugely expensive (even compared with this brute force
        approach). Plus, this is easy to parallelise. It may be more sensible for use RegularGridInterpolators for
        each position in a loop since the vertical and time are regularly spaced.

        Parameters
        ----------
        args : tuple
            The input arguments as a tuple of:
            boundary_points : np.ndarray
                The open boundary point indices (self.nodes or self.elements).
            x : np.ndarray
                The source data x positions (spherical coordinates).
            y : np.ndarray
                The source data y positions (spherical coordinates).
            fvcom_layer_depth : np.ndarray
                The vertical grid layer depths in metres (nx, nz) or (nx, nz, time).
            coarse : PyFVCOM.preproc.RegularReader
                The coarse data from which we're interpolating.
            coarse_name : str
                The name of the data from which we're interpolating.
            verbose : bool
                True for verbose output, False for none. Only really useful in serial.
            t : int
                The time index for the coarse data.

        Returns
        -------
        The interpolated boundary data at `x', `y', `fvcom_layer_depth' for coarse.data.coarse_name at time index `t'.

        """

        # MATLAB interp_coarse_to_obc.m reimplementation in Python with some tweaks. The only difference is I renamed
        # the variables as the MATLAB ones were horrible.
        #
        # This gets slower the more variables you interpolate (i.e. each successive variable being interpolated
        # increases the time it takes to interpolate). This is probably a memory overhead from using
        # multiprocessing.Pool.map().
        boundary_points, x, y, fvcom_layer_depth, coarse, coarse_name, verbose, t = args

        num_fvcom_z = fvcom_layer_depth.shape[-1]
        num_fvcom_points = len(boundary_points)

        num_coarse_z = coarse.grid.siglay_z.shape[0]  # rename!

        if verbose:
            print(f'Interpolating time {t} of {coarse.dims.time}')
        # Get this time's data from the coarse model.
        coarse_data_volume = np.squeeze(getattr(coarse.data, coarse_name)[t, ...]).reshape(num_coarse_z, -1).T

        interp_fvcom_data = np.full((num_fvcom_points, num_coarse_z), np.nan)
        interp_fvcom_depth = np.full((num_fvcom_points, num_coarse_z), np.nan)
        if np.ndim(coarse.grid.siglay_z) == 4:
            coarse_layer_depth = np.squeeze(coarse.grid.siglay_z[..., t].reshape((num_coarse_z, -1))).T
        else:
            coarse_layer_depth = np.squeeze(coarse.grid.siglay_z.reshape((num_coarse_z, -1))).T

        # Go through each coarse model vertical level, interpolate the coarse model depth and data to each position
        # in the current open boundary. Make sure we remove all NaNs.
        for z_index in np.arange(num_coarse_z):
            if verbose:
                print(f'Interpolating layer {z_index} of {num_coarse_z}')
            coarse_data_layer = coarse_data_volume[:, z_index]
            coarse_depth_layer = coarse_layer_depth[:, z_index]

            coarse_lon, coarse_lat = np.meshgrid(coarse.grid.lon, coarse.grid.lat)
            coarse_lon, coarse_lat = coarse_lon.ravel(), coarse_lat.ravel()

            interpolator = LinearNDInterpolator((coarse_lon, coarse_lat), coarse_data_layer)
            interp_fvcom_data_layer = interpolator((x, y))
            # Update values in the triangulation so we don't have to remake it (which can be expensive).
            interpolator.values = coarse_depth_layer[:, np.newaxis].astype(np.float64)
            interp_fvcom_depth_layer = interpolator((x, y))

            # If we're interpolating NEMO data (and we are when we're using this method), the bottom layer will
            # always return NaNs, so this message will always be triggered, which is a bit annoying. It'd be nice to
            # use the tmask option when loading the NEMOReader to omit these values properly (rather than just
            # setting them to NaN) so we could stop spitting out these messages for each interpolation. Maybe another
            # day, eh?
            if np.any(np.isnan(interp_fvcom_data_layer)) or np.any(np.isnan(interp_fvcom_depth_layer)):
                bad_indices = np.argwhere(np.isnan(interp_fvcom_data_layer))
                if len(bad_indices) == 1:
                    singular_plural = ''
                else:
                    singular_plural = 's'
                warn(f'{len(bad_indices)} FVCOM boundary node{singular_plural} returned NaN after interpolation. Using '
                     f'inverse distance interpolation instead.')
                for bad_index in bad_indices:
                    weight = 1 / np.hypot(coarse_lon - x[bad_index], coarse_lat - y[bad_index])
                    weight = weight / weight.max()
                    interp_fvcom_data_layer[bad_index] = (coarse_data_layer * weight).sum() / weight.sum()
                    interp_fvcom_depth_layer[bad_index] = (coarse_depth_layer * weight).sum() / weight.sum()

            interp_fvcom_data[:, z_index] = interp_fvcom_data_layer
            interp_fvcom_depth[:, z_index] = interp_fvcom_depth_layer

        # Now for each point in the current open boundary points (x, y), interpolate the interpolated (in the
        # horizontal) coarse model data onto the FVCOM vertical grid.
        interp_fvcom_boundary = np.full((num_fvcom_points, num_fvcom_z), np.nan)
        for p_index in np.arange(num_fvcom_points):
            if verbose:
                print(f'Interpolating point {p_index} of {num_fvcom_points}')
            fvcom_point_depths = fvcom_layer_depth[p_index, :]
            coarse_point_depths = interp_fvcom_depth[p_index, :]

            # Drop the NaN values from the coarse depths and data (where we're at the bottom of the water column).
            nan_depth = np.isnan(coarse_point_depths)
            coarse_point_depths = coarse_point_depths[~nan_depth]
            interp_vertical_profile = interp_fvcom_data[p_index, ~nan_depth]

            # Squeeze the coarse model water column into the FVCOM one.
            norm_coarse_point_depths = fix_range(coarse_point_depths,
                                                 fvcom_point_depths.min(),
                                                 fvcom_point_depths.max())

            if not np.any(np.isnan(coarse_point_depths)):
                interp_fvcom_boundary[p_index, :] = np.interp(fvcom_point_depths,
                                                              norm_coarse_point_depths,
                                                              interp_vertical_profile)

        # Make sure we remove any NaNs from the vertical profiles by replacing with the interpolated data from the
        # non-NaN data in the vicinity.
        for p_index in np.arange(num_fvcom_z):
            horizontal_slice = interp_fvcom_boundary[:, p_index]
            if np.any(np.isnan(horizontal_slice)):
                good_indices = ~np.isnan(horizontal_slice)
                interpolator = LinearNDInterpolator((x[good_indices], y[good_indices]), horizontal_slice[good_indices])
                interp_fvcom_boundary[:, p_index] = interpolator((x, y))

        return interp_fvcom_boundary

    @staticmethod
    def _interpolate_in_time(args):
        """
        Worker function to interpolate the given time series in time.

        Parameters
        ----------
        args : tuple
            A tuple containing:
            time_coarse : np.ndarray
                The coarse time data.
            data_coarse : np.ndarray
                The coarse data.
            time_fine : np.ndarray
                The fine time data onto which to interpolate [time_coarse, data_coarse].

        Returns
        -------
        data_fine : np.ndarray
            The interpolate data time series.

        """

        time_coarse, data_coarse, time_fine = args

        return np.interp(time_fine, time_coarse, data_coarse)

    @staticmethod
    def _nested_forcing_interpolator(data, lon, lat, depth, points):
        """
        Worker function to interpolate the regularly gridded [depth, lat, lon] data onto the supplied `points' [lon,
        lat, depth].

        Parameters
        ----------
        data : np.ndarray
            Coarse data to interpolate [depth, lat, lon].
        lon : np.ndarray
            Coarse data longitude array.
        lat : np.ndarray
            Coarse data latitude array.
        depth : np.ndarray
            Coarse data depth array.
        points : np.ndarray
            Points onto which the coarse data should be interpolated.

        Returns
        -------
        interpolated_data : np.ndarray
            Coarse data interpolated onto the supplied points.

        """

        # Make a RegularGridInterpolator from the supplied 4D data.
        ft = RegularGridInterpolator((depth, lat, lon), data, method='linear', fill_value=None)
        interpolated_data = ft(points)

        return interpolated_data

    def avg_nest_force_vel(self):
        """
        Create depth-averaged velocities (`ua', `va') in the current self.data data.

        """
        layer_thickness = self.sigma.levels_center.T[0:-1, :] - self.sigma.levels_center.T[1:, :]
        self.data.ua = zbar(self.data.u, layer_thickness)
        self.data.va = zbar(self.data.v, layer_thickness)


def read_sms_mesh(mesh, nodestrings=False):
    """
    Reads in the SMS unstructured grid format. Also creates IDs for output to
    MIKE unstructured grid format.

    Parameters
    ----------
    mesh : str
        Full path to an SMS unstructured grid (.2dm) file.
    nodestrings : bool, optional
        Set to True to return the IDs of the node strings as a dictionary.

    Returns
    -------
    triangle : np.ndarray
        Integer array of shape (nele, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in `x' and `y' (see below). Values
        are python-indexed.
    nodes : np.ndarray
        Integer number assigned to each node.
    X, Y, Z : np.ndarray
        Coordinates of each grid node and any associated Z value.
    types : np.ndarray
        Classification for each node string based on the number of node
        strings + 2. This is mainly for use if converting from SMS .2dm
        grid format to DHI MIKE21 .mesh format since the latter requires
        unique IDs for each boundary (with 0 and 1 reserved for land and
        sea nodes).
    nodestrings : list, optional (see nodestrings above)
        Optional list of lists containing the node IDs (python-indexed) of the
        node strings in the SMS grid.

    """

    fileRead = open(mesh, 'r')
    lines = fileRead.readlines()
    fileRead.close()

    triangles = []
    nodes = []
    types = []
    nodeStrings = []
    nstring = []
    x = []
    y = []
    z = []

    # MIKE unstructured grids allocate their boundaries with a type ID flag.
    # Although this function is not necessarily always the precursor to writing
    # a MIKE unstructured grid, we can create IDs based on the number of node
    # strings in the SMS grid. MIKE starts counting open boundaries from 2 (1
    # and 0 are land and sea nodes, respectively).
    typeCount = 2

    for line in lines:
        line = line.strip()
        if line.startswith('E3T'):
            ttt = line.split()
            t1 = int(ttt[2]) - 1
            t2 = int(ttt[3]) - 1
            t3 = int(ttt[4]) - 1
            triangles.append([t1, t2, t3])
        elif line.startswith('ND '):
            xy = line.split()
            x.append(float(xy[2]))
            y.append(float(xy[3]))
            z.append(float(xy[4]))
            nodes.append(int(xy[1]))
            # Although MIKE keeps zero and one reserved for normal nodes and
            # land nodes, SMS doesn't. This means it's not straightforward
            # to determine this information from the SMS file alone. It would
            # require finding nodes which are edge nodes and assigning their
            # ID to one. All other nodes would be zero until they were
            # overwritten when examining the node strings below.
            types.append(0)
        elif line.startswith('NS '):
            allTypes = line.split(' ')

            for nodeID in allTypes[2:]:
                types[np.abs(int(nodeID)) - 1] = typeCount
                if int(nodeID) > 0:
                    nstring.append(int(nodeID) - 1)
                else:
                    nstring.append(np.abs(int(nodeID)) - 1)
                    nodeStrings.append(nstring)
                    nstring = []

                # Count the number of node strings, and output that to types.
                # Nodes in the node strings are stored in nodeStrings.
                if int(nodeID) < 0:
                    typeCount += 1

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    types = np.asarray(types)
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)
    if nodestrings:
        return triangle, nodes, X, Y, Z, types, nodeStrings
    else:
        return triangle, nodes, X, Y, Z, types


def read_sms_map(map_file, merge_lines=False):
    """
    Reads in the SMS map file format.

    Parameters
    ----------
    map_file : str
        Full path to an SMS map (.map) file.
    merge_lines : bool
        Set to True to merge distinct arcs into contiguous ones to make sensible polygons. Defaults to False.

    Returns
    -------
    arcs : dict
        Dictionary of each map (set of arcs) in the map file name as per SMS names. Each is a list of the coordinate
        pairs and elevation for the polygons defined in the map file.

    Notes
    -----
    This is inspired by and partially based on the MATLAB fvcom-toolbox function read_sms_map.m.

    This could also (additionally) return a shapely.MultiPolygon, but I've not done that for now as it's trivial to
    convert these arcs into a MultiPolygon after the fact.

    """

    x, y, z, arc_id, node_id = [], [], [], [], []
    arcs = {}
    with open(map_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('COVNAME'):
                arc_name = line.split()[1].replace('"', '')
                arcs[arc_name] = []
            elif line == 'NODE':
                line = next(f).strip()
                _, node_x, node_y, node_z = line.split()
                x.append(float(node_x))
                y.append(float(node_y))
                z.append(float(node_z))
                line = next(f).strip()
                node_id.append(float(line.split()[-1]))
            elif line == 'ARC':
                # Skip the arc ID.
                next(f)
                # Skip the arc elevation.
                next(f)
                # Grab the start and end IDs for the arc nodes.
                line = next(f).strip()
                start_id, end_id = [int(i) for i in line.split()[1:]]
                start_index = node_id.index(start_id)
                end_index = node_id.index(end_id)
                # The number of nodes in the current arc.
                line = next(f).strip()
                number_of_nodes = int(line.split()[-1])
                # Read in the coordinates and elevation for the current arc. Start with the appropriate coordinate
                # from `x', `y' and its `z' value too (the 'node' in SMS parlance) before appending the arc vertices
                # (in SMS parlance).
                arcs[arc_name].append([[x[start_index], y[start_index], z[start_index]]])
                for next_line in range(number_of_nodes):
                    line = next(f).strip()
                    arcs[arc_name][-1].append([float(i) for i in line.split()])
                # Add the end node
                arcs[arc_name][-1].append([x[end_index], y[end_index], z[end_index]])

    if merge_lines:
        # Get all the z values for the arcs so we can put them back after merging.
        for arc in arcs:
            separate_lines = [shapely.geometry.LineString(a) for a in arcs[arc]]
            all_lines = shapely.geometry.MultiLineString(separate_lines)
            merged = shapely.ops.linemerge(all_lines)
            arcs[arc] = [np.asarray(line.coords) for line in merged]
    else:
        # Just make the arcs a list of numpy arrays.
        arcs = {key: [np.asarray(i) for i in arcs[key]] for key in arcs}

    return arcs


def read_fvcom_mesh(mesh):
    """
    Reads in the FVCOM unstructured grid format.

    Parameters
    ----------
    mesh : str
        Full path to the FVCOM unstructured grid file (.dat usually).

    Returns
    -------
    triangle : np.ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in `x' and `y' (see below).
    nodes : np.ndarray
        Integer number assigned to each node.
    X, Y, Z : np.ndarray
        Coordinates of each grid node and any associated Z value.

    """

    fileRead = open(mesh, 'r')
    # Skip the file header (two lines)
    lines = fileRead.readlines()[2:]
    fileRead.close()

    triangles = []
    nodes = []
    x = []
    y = []
    z = []

    for line in lines:
        ttt = line.strip().split()
        if len(ttt) == 5:
            t1 = int(ttt[1]) - 1
            t2 = int(ttt[2]) - 1
            t3 = int(ttt[3]) - 1
            triangles.append([t1, t2, t3])
        elif len(ttt) == 4:
            x.append(float(ttt[1]))
            y.append(float(ttt[2]))
            z.append(float(ttt[3]))
            nodes.append(int(ttt[0]))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)

    return triangle, nodes, X, Y, Z


def read_smesh_mesh(mesh):
    """
    Reads output of the smeshing tool. This is (close) to the fort.14 file format.

    Parameters
    ----------
    mesh : str
        Full path to the smesh output file

    Returns
    -------
    triangle : np.ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of three points and this contains the three node
        numbers which refer to the coordinates in `x' and `y' (see below).
    x, y : np.ndarray
        Coordinates of each grid node

    """

    with open(mesh, 'r') as file_read:
        # Skip the file header line
        lines = file_read.readlines()[1:]

    triangles = []
    x = []
    y = []

    for line in lines:
        ttt = line.strip().split()
        if len(ttt) == 3:
            t1 = int(ttt[0])
            t2 = int(ttt[1])
            t3 = int(ttt[2])
            triangles.append([t1, t2, t3])
        elif len(ttt) == 2:
            x.append(float(ttt[0]))
            y.append(float(ttt[1]))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    x = np.asarray(x)
    y = np.asarray(y)

    return triangle, x, y


def read_mike_mesh(mesh, flipZ=True):
    """
    Reads in the MIKE unstructured grid format.

    Depth sign is typically reversed (i.e. z*-1) but can be disabled by
    passing flipZ=False.

    Parameters
    ----------
    mesh : str
        Full path to the DHI MIKE21 unstructured grid file (.mesh usually).
    flipZ : bool, optional
        DHI MIKE21 unstructured grids store the z value as positive down
        whereas FVCOM wants negative down. The conversion is
        automatically applied unless flipZ is set to False.

    Returns
    -------
    triangle : np.ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
    nodes) which refer to the coordinates in `x' and `y' (see below). Give as
        a zero-indexed array.
    nodes : np.ndarray
        Integer number assigned to each node.
    X, Y, Z : np.ndarray
        Coordinates of each grid node and any associated Z value.
    types : np.ndarray
        Classification for each open boundary. DHI MIKE21 .mesh format
        requires unique IDs for each open boundary (with 0 and 1
        reserved for land and sea nodes).

    """

    fileRead = open(mesh, 'r')
    # Skip the file header (one line)
    lines = fileRead.readlines()[1:]
    fileRead.close()

    triangles = []
    nodes = []
    types = []
    x = []
    y = []
    z = []

    for line in lines:
        ttt = line.strip().split()
        if len(ttt) == 4:
            t1 = int(ttt[1]) - 1
            t2 = int(ttt[2]) - 1
            t3 = int(ttt[3]) - 1
            triangles.append([t1, t2, t3])
        elif len(ttt) == 5:
            x.append(float(ttt[1]))
            y.append(float(ttt[2]))
            z.append(float(ttt[3]))
            types.append(int(ttt[4]))
            nodes.append(int(ttt[0]))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    types = np.asarray(types)
    X = np.asarray(x)
    Y = np.asarray(y)
    # N.B. Depths should be negative for FVCOM
    if flipZ:
        Z = -np.asarray(z)
    else:
        Z = np.asarray(z)

    return triangle, nodes, X, Y, Z, types


def read_gmsh_mesh(mesh):
    """
    Reads in the GMSH unstructured grid format (version 2.2).

    Parameters
    ----------
    mesh : str
        Full path to the DHI MIKE21 unstructured grid file (.mesh usually).

    Returns
    -------
    triangle : np.ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of three
        points and this contains the three node numbers (stored in nodes) which
        refer to the coordinates in `x' and `y' (see below).
    nodes : np.ndarray
        Integer number assigned to each node.
    X, Y, Z : np.ndarray
        Coordinates of each grid node and any associated Z value.

    """

    fileRead = open(mesh, 'r')
    lines = fileRead.readlines()
    fileRead.close()

    _header = False
    _nodes = False
    _elements = False

    # Counters for the nodes and elements.
    n = 0
    e = 0

    for line in lines:
        line = line.strip()

        # If we've been told we've got to the header, read in the mesh version
        # here.
        if _header:
            _header = False
            continue

        # Grab the number of nodes.
        if _nodes:
            nn = int(line.strip())
            x, y, z, nodes = np.zeros(nn) - 1, np.zeros(nn) - 1, np.zeros(nn) - 1, np.zeros(nn).astype(int) - 1
            _nodes = False
            continue

        # Grab the number of elements.
        if _elements:
            ne = int(line.strip())
            triangles = np.zeros((ne, 3)).astype(int) - 1
            _elements = False
            continue

        if line == r'$MeshFormat':
            # Header information on the next line
            _header = True
            continue

        elif line == r'$EndMeshFormat':
            continue

        elif line == r'$Nodes':
            _nodes = True
            continue

        elif line == r'$EndNodes':
            continue

        elif line == r'$Elements':
            _elements = True
            continue

        elif line == r'$EndElements':
            continue

        else:
            # Some non-info line, so either nodes or elements. Discern that
            # based on the number of fields.
            s = line.split(' ')
            if len(s) == 4:
                # Nodes
                nodes[n] = int(s[0])
                x[n] = float(s[1])
                y[n] = float(s[2])
                z[n] = float(s[3])
                n += 1

            # Only keep the triangulation for the 2D mesh (ditch the 1D stuff).
            elif len(s) > 4 and int(s[1]) == 2:
                # Offset indices by one for Python indexing.
                triangles[e, :] = [int(i) - 1 for i in s[-3:]]
                e += 1

            else:
                continue

    # Tidy up the triangles array to remove the empty rows due to the number
    # of elements specified in the mesh file including the 1D triangulation.
    # triangles = triangles[triangles[:, 0] != -1, :]
    triangles = triangles[:e, :]

    return triangles, nodes, x, y, z


def read_fvcom_obc(obc):
    """
    Read in an FVCOM open boundary file.

    Parameters
    ----------
    obc : str
        Path to the casename_obc.dat file from FVCOM.

    Returns
    -------
    nodes : np.ndarray
        Node IDs (zero-indexed) for the open boundary.
    types : np.ndarray
        Open boundary node types (see the FVCOM manual for more information on
        what these values mean).
    count : np.ndarray
        Open boundary node number.


    """

    obcs = np.genfromtxt(obc, skip_header=1).astype(int)
    count = obcs[:, 0]
    nodes = obcs[:, 1] - 1
    types = obcs[:, 2]

    return nodes, types, count


def parse_obc_sections(obc_node_array, triangle):
    """
    Separates the open boundary nodes of a mesh into the separate contiguous open boundary segments

    Parameters
    ----------
    obc_node_array : array
        Array of the nodes which are open boundary nodes, as nodes returned by read_fvcom_obc
    triangle : 3xn array
        Triangulation array of nodes, as triangle returned by read_fvcom_mesh

    Returns
    -------
    nodestrings : list of arrays
        A list of arrays, each of which is one contiguous section of open boundary

    """

    all_edges = np.vstack([triangle[:, 0:2], triangle[:, 1:], triangle[:, [0, 2]]])
    boundary_edges = all_edges[np.all(np.isin(all_edges, obc_node_array), axis=1), :]
    u_nodes, bdry_counts = np.unique(boundary_edges, return_counts=True)
    start_end_nodes = list(u_nodes[bdry_counts == 1])

    nodestrings = []

    while len(start_end_nodes) > 0:
        this_obc_section_nodes = [start_end_nodes[0]]
        start_end_nodes.remove(start_end_nodes[0])

        nodes_to_add = True

        while nodes_to_add:
            possible_nodes = np.unique(boundary_edges[np.any(np.isin(boundary_edges, this_obc_section_nodes), axis=1), :])
            nodes_to_add = list(possible_nodes[~np.isin(possible_nodes, this_obc_section_nodes)])
            if nodes_to_add:
                this_obc_section_nodes.append(nodes_to_add[0])

        nodestrings.append(np.asarray(this_obc_section_nodes))
        start_end_nodes.remove(list(set(start_end_nodes).intersection(this_obc_section_nodes)))

    return nodestrings


def read_sms_cst(cst):
    """
    Read a CST file and store the vertices in a dict.

    Parameters
    ----------
    cst : str
        Path to the CST file to load in.

    Returns
    -------
    vert : dict
        Dictionary with the coordinates of the vertices of the arcs defined in
        the CST file.

    """

    f = open(cst, 'r')
    lines = f.readlines()
    f.close()

    vert = {}
    c = 0
    for line in lines:
        line = line.strip()
        if line.startswith('COAST'):
            pass
        else:
            # Split the line on tabs and work based on that output.
            line = line.split('\t')
            if len(line) == 1:
                # Number of arcs. We don't especially need to know this.
                pass

            elif len(line) == 2:
                # Number of nodes within a single arc. Store the current index
                # and use as the key for the dict.
                nv = int(line[0])
                id = str(c)    # dict key
                vert[id] = []  # initialise the vert list
                c += 1         # arc counter

            elif len(line) == 3:
                coords = [float(x) for x in line[:-1]]
                # Skip the last position if we've already got some data in the
                # dict for this arc.
                if vert[id]:
                    if len(vert[id]) != nv - 1:
                        vert[id].append(coords)
                    else:
                        # We're at the end of this arc, so convert the
                        # coordinates we've got to a numpy array for easier
                        # handling later on.
                        vert[id] = np.asarray(vert[id])
                else:
                    vert[id].append(coords)

    return vert


def write_sms_mesh(triangles, nodes, x, y, z, types, mesh):
    """
    Takes appropriate triangle, node, boundary type and coordinate data and
    writes out an SMS formatted grid file (mesh). The footer is largely static,
    but the elements, nodes and node strings are parsed from the input data.

    Input data is probably best obtained from one of:

        grid.read_sms_mesh()
        grid.read_fvcom_mesh()
        grid.read_mike_mesh()

    which read in the relevant grids and output the required information for
    this function.

    The footer contains meta data and additional information. See page 18 in
    http://smstutorials-11.0.aquaveo.com/SMS_Gen2DM.pdf.

    In essence, four bits are critical:
        1. The header/footer MESH2D/BEGPARAMDEF
        2. E3T prefix for the connectivity:
            (elementID, node1, node2, node3, material_type)
        3. ND prefix for the node information:
            (nodeID, x, y, z)
        4. NS prefix for the node strings which indicate the open boundaries.

    As far as I can tell, the footer is largely irrelevant for FVCOM purposes.

    Parameters
    ----------
    triangles : np.ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in `x' and `y' (see below). Give as
        a zero-indexed array.
    nodes : np.ndarray
        Integer number assigned to each node.
    x, y, z : np.ndarray
        Coordinates of each grid node and any associated Z value.
    types : np.ndarray
        Classification for each open boundary. DHI MIKE21 .mesh format
        requires unique IDs for each open boundary (with 0 and 1
        reserved for land and sea nodes). Similar values can be used in
        SMS grid files too.
    mesh : str
        Full path to the output file name.

    Notes
    -----

    The footer contains some information which is largely ignored here, but
    which is included in case it's critical.

    BEGPARAMDEF = Marks end of mesh data/beginning of mesh model definition
    GM = Mesh name (enclosed in "")
    SI = use SI units y/n = 1/0
    DY = Dynamic model y/n = 1/0
    TU = Time units
    TD = Dynamic time data (?)
    NUME = Number of entities available (nodes, node strings, elements)
    BGPGC = Boundary group parameter group correlation y/n = 1/0
    BEDISP/BEFONT = Format controls on display of boundary labels.
    ENDPARAMDEF = End of the mesh model definition
    BEG2DMBC = Beginning of the model assignments
    MAT = Material assignment
    END2DMBC = End of the model assignments

    """

    file_write = open(mesh, 'w')
    # Add a header
    file_write.write('MESH2D\n')

    # Write out the connectivity table (triangles)
    current_node = 0
    for line in triangles.copy():
        # Bump the numbers by one to correct for Python indexing from zero
        line += 1
        line_string = []
        # Convert the numpy array to a string array
        for value in line:
            line_string.append(str(value))

        current_node += 1
        # Build the output string for the connectivity table
        output = f'E3T {current_node} {" ".join(line_string)} 1\n'
        file_write.write(output)

    for count, line in enumerate(nodes):
        output = f'ND {line} {x[count]:.8e} {y[count]:.8e} {z[count]:.8e}\n'

        file_write.write(output)

    # Convert MIKE boundary types to node strings. The format requires a prefix
    # NS, and then a maximum of 10 node IDs per line. The node string tail is
    # indicated by a negative node ID.

    # Iterate through the unique boundary types to get a new node string for
    # each boundary type (ignore types of less than 2 which are not open
    # boundaries in MIKE).
    for boundary_type in np.unique(types[types > 1]):

        # Find the nodes for the boundary type which are greater than 1 (i.e.
        # not 0 or 1).
        node_boundaries = nodes[types == boundary_type].astype(int)

        node_strings = [node_boundaries[i:i+10] for i in range(0, len(node_boundaries), 10)]
        node_strings[-1][-1] = -node_strings[-1][-1]  # flip sign of the last node
        joined_string = [f"NS  {' '.join(section.astype(str))}\n" for section in node_strings]
        for section in joined_string:
            file_write.write(section)

    # Add all the blurb at the end of the file.
    #
    # BEGPARAMDEF = Marks end of mesh data/beginning of mesh model definition
    # GM = Mesh name (enclosed in "")
    # SI = use SI units y/n = 1/0
    # DY = Dynamic model y/n = 1/0
    # TU = Time units
    # TD = Dynamic time data (?)
    # NUME = Number of entities available (nodes, node strings, elements)
    # BGPGC = Boundary group parameter group correlation y/n = 1/0
    # BEDISP/BEFONT = Format controls on display of boundary labels.
    # ENDPARAMDEF = End of the mesh model definition
    # BEG2DMBC = Beginning of the model assignments
    # MAT = Material assignment
    # END2DMBC = End of the model assignments
    footer_parts = ('BEGPARAMDEF', 'GM  "Mesh"', 'SI  0', 'DY  0', 'TU  ""', 'TD  0  0', 'NUME  3', 'BCPGC  0',
                    'BEDISP  0 0 0 0 1 0 1 0 0 0 0 1', 'BEFONT  0 2', 'BEDISP  1 0 0 0 1 0 1 0 0 0 0 1',
                    'BEFONT  1 2', 'BEDISP  2 0 0 0 1 0 1 0 0 0 0 1', 'BEFONT  2 2', 'ENDPARAMDEF',
                    'BEG2DMBC', 'MAT  1 "material 01"', 'END2DMBC')
    footer = '\n'.join(footer_parts)

    file_write.write(f'{footer}\n')

    file_write.close()


def write_sms_bathy(triangles, nodes, z, filename):
    """
    Writes out the additional bathymetry file sometimes output by SMS. Not sure
    why this is necessary as it's possible to put the depths in the other file,
    but hey ho, it is obviously sometimes necessary.

    Parameters
    ----------
    triangles : np.ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
    nodes) which refer to the coordinates in `x' and `y' (see below). Give as
        a zero-indexed array.
    nodes : np.ndarray
        Integer number assigned to each node.
    z : np.ndarray
        Depth values at each node location.
    filename : str
        Full path of the output file name.

    """

    fname = open(filename, 'w')

    # Get some information needed for the metadata side of things
    node_number = len(nodes)
    element_number = len(triangles[:, 0])

    # Header format (see:
    #     http://wikis.aquaveo.com/xms/index.php?title=GMS:Data_Set_Files)
    # DATASET = indicates data
    # OBJTYPE = type of object (i.e. mesh 3d, mesh 2d) data is associated with
    # BEGSCL = Start of the scalar data set
    # ND = Number of data values
    # NC = Number of elements
    # NAME = Freeform data set name
    # TS = Time step of the data
    header = 'DATASET\nOBJTYEP = "mesh2d"\nBEGSCL\nND  {:<6d}\nNC  {:<6d}\nNAME "Z_interp"\nTS 0 0\n'.format(int(node_number), int(element_number))
    fname.write(header)

    # Now just iterate through all the z values. This process assumes the z
    # values are in the same order as the nodes. If they're not, this will
    # make a mess of your data.
    for depth in z:
        fname.write('{:.5f}\n'.format(float(depth)))

    # Close the file with the footer
    fname.write('ENDDS\n')
    fname.close()


def write_mike_mesh(triangles, nodes, x, y, z, types, mesh):
    """
    Write out a DHI MIKE unstructured grid (mesh) format file. This
    assumes the input coordinates are in longitude and latitude. If they
    are not, the header will need to be modified with the appropriate
    string (which is complicated and of which I don't have a full list).

    If types is empty, then zeros will be written out for all nodes.

    Parameters
    ----------
    triangles : np.ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
    nodes) which refer to the coordinates in `x' and `y' (see below). Give as
        a zero-indexed array.
    nodes : np.ndarray
        Integer number assigned to each node.
    x, y, z : np.ndarray
        Coordinates of each grid node and any associated Z value.
    types : np.ndarray
        Classification for each open boundary. DHI MIKE21 .mesh format
        requires unique IDs for each open boundary (with 0 and 1
        reserved for land and sea nodes).
    mesh : str
        Full path to the output mesh file.

    """
    file_write = open(mesh, 'w')
    # Add a header
    output = '{}  LONG/LAT'.format(int(len(nodes)))
    file_write.write(output + '\n')

    if len(types) == 0:
        types = np.zeros(shape=(len(nodes), 1))

    # Write out the node information
    for count, line in enumerate(nodes):

        # Convert the numpy array to a string array
        line_string = str(line)

        output = \
            [line_string] + \
            ['{}'.format(x[count])] + \
            ['{}'.format(y[count])] + \
            ['{}'.format(z[count])] + \
            ['{}'.format(int(types[count]))]
        output = ' '.join(output)

        file_write.write(output + '\n')

    # Now for the connectivity

    # Little header. No idea what the 3 and 21 are all about (version perhaps?)
    output = '{} {} {}'.format(int(len(triangles)), '3', '21')
    file_write.write(output + '\n')

    for count, line in enumerate(triangles):

        # Bump the numbers by one to correct for Python indexing from zero
        line = line + 1
        line_string = []
        # Convert the numpy array to a string array
        for value in line:
            line_string.append(str(value))

        # Build the output string for the connectivity table
        output = [str(count + 1)] + line_string
        output = ' '.join(output)

        file_write.write(output + '\n')

    file_write.close()


def write_fvcom_mesh(triangles, nodes, x, y, z, mesh, extra_depth=None):
    """
    Write out an FVCOM unstructured grid (mesh) format file.

    Parameters
    ----------
    triangles : np.ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in `x' and `y' (see below). Give as
        a zero-indexed array.
    nodes : np.ndarray
        Integer number assigned to each node.
    x, y, z : np.ndarray
        Coordinates of each grid node and any associated Z value.
    mesh : str
        Full path to the output mesh file.
    extra_depth : str, optional
        If given, output depths to a separate FVCOM compatible file.

    """

    with open(mesh, 'w') as f:
        f.write('Node Number = {:d}\n'.format(len(x)))
        f.write('Cell Number = {:d}\n'.format(np.shape(triangles)[0]))
        for i, triangle in enumerate(triangles, 1):
            f.write('{node:d} {:d} {:d} {:d} {node:d}\n'.format(node=i, *triangle + 1))
        for node in zip(nodes, x, y, z):
            f.write('{:d} {:.6f} {:.6f} {:.6f}\n'.format(*node))

    if extra_depth:
        with open(extra_depth, 'w') as f:
            f.write('Node Number = {:d}\n'.format(len(x)))
            for node in zip(x, y, z):
                f.write('{:.6f} {:.6f} {:.6f}\n'.format(*node))


def write_sms_cst(obc, file, sort=False):
    """
    Read a CST file and store the vertices in a dict.

    Parameters
    ----------
    obc : dict
        Dict with each entry as a NumPy array of coordinates (x, y).
    file : str
        Path to the CST file to which to write (overwrites existing files).
    sort : bool, optional
        Optionally sort the output coordinates (by x then y). This might break
        things with complicated open boundary geometries.

    """

    nb = len(obc)

    with open(file, 'w') as f:
        # Header
        f.write('COAST\n')
        f.write('{:d}\n'.format(nb))

        for _, bb in obc:  # each boundary
            nn = len(bb)

            # The current arc's header
            f.write('{:d}\t0.0\n'.format(nn))

            if sort:
                idx = np.lexsort(bb.transpose())
                bb = bb[idx, :]

            for xy in bb:
                f.write('\t{:.6f}\t{:.6f}\t0.0\n'.format(xy[0], xy[1]))

        f.close


def MIKEarc2cst(file, output):
    """
    Read in a set of MIKE arc files and export to CST format compatible with
    SMS.

    MIKE format is:

        x, y, position, z(?), ID

    where position is 1 = along arc and 0 = end of arc.

    In the CST format, the depth is typically zero, but we'll read it from the
    MIKE z value and add it to the output file nevertheless. For the
    conversion, we don't need the ID, so we can ignore that.

    Parameters
    ----------
    file : str
        Full path to the DHI MIKE21 arc files.
    output : str
        Full path to the output file.

    """

    with open(file, 'r') as file_in:
        lines = file_in.readlines()

    with open(output, 'w') as file_out:
        # Add the easy header
        file_out.write('COAST\n')
        # This isn't pretty, but assuming you're coastline isn't millions of
        # points, it should be ok...
        num_arcs = 0
        for line in lines:
            x, y, pos, z, arc_id = line.strip().split(' ')
            if int(pos) == 0:
                num_arcs += 1

        file_out.write('{}\n'.format(int(num_arcs)))

        arc = []
        n = 1

        for line in lines:

            x, y, pos, z, arc_id = line.strip().split(' ')
            if int(pos) == 1:
                arc.append([x, y])
                n += 1
            elif int(pos) == 0:
                arc.append([x, y])
                # We're at the end of an arc, so write out to file. Start with
                # number of nodes and z
                file_out.write('{}\t{}\n'.format(int(n), float(z)))
                for arc_position in arc:
                    file_out.write('\t{}\t{}\t{}\n'.format(float(arc_position[0]), float(arc_position[1]), float(z)))
                # Reset n and arc for new arc
                n = 1
                arc = []


def shp2cst(file, output_file):
    """
    Convert ESRI ShapeFiles to SMS-compatible CST files.

    Parameters
    ----------
    file : str
        Full path to the ESRI ShapeFile to convert.
    output_file : str
        Full path to the output file.

    Notes
    -----
    There's no particular reason this function should exist as SMS can read shapefiles natively. I imagine I didn't
    know that when I wrote this.

    """

    sf = shapefile.Reader(file)
    shapes = sf.shapes()

    nArcs = sf.numRecords

    # Set up the output file
    with open(output_file, 'w') as file_out:
        file_out.write('COAST\n')
        file_out.write('{}\n'.format(int(nArcs)))

        z = 0

        for arc in range(nArcs):
            # Write the current arc out to file. Start with number of nodes and z
            arcLength = len(shapes[arc].points)
            file_out.write('{}\t{}\n'.format(arcLength, float(z)))
            # Add the actual arc
            for arcPos in shapes[arc].points:
                file_out.write('\t{}\t{}\t{}\n'.format(float(arcPos[0]), float(arcPos[1]), float(z)))


def find_nearest_point(grid_x, grid_y, x, y, maxDistance=np.inf):
    """
    Given some point(s) `x' and `y', find the nearest grid node in `grid_x' and `grid_y'.

    Returns the nearest coordinate(s), distance(s) from the point(s) and the index(ices) in the respective array.

    Optionally specify a maximum distance (in the same units as the input) to only return grid positions which are
    within that distance. This means if your point lies outside the grid, for example, you can use maxDistance to
    filter it out. Positions and indices which cannot be found within maxDistance are returned as NaN; distance is
    always returned, even if the maxDistance threshold has been exceeded.

    Parameters
    ----------
    grid_x, grid_y : np.ndarray
        Coordinates within which to search for the nearest point given in `x' and `y'.
    x, y : np.ndarray
        List of coordinates to find the closest value in `grid_x' and `grid_y'. Upper threshold of distance is given
        by maxDistance (see below).
    maxDistance : float, optional
        Unless given, there is no upper limit on the distance away from the source for which a result is deemed
        valid. Any other value specified here limits the upper threshold.

    Returns
    -------
    nearest_x, nearest_y : np.ndarray
        Coordinates from `grid_x' and `grid_y' which are within maxDistance (if given) and closest to the
        corresponding point in `x' and `y'.
    distance : np.ndarray
        Distance between each point in `x' and `y' and the closest value in `grid_x' and `grid_y'. Even if
        maxDistance is given (and exceeded), the distance is reported here.
    index : np.ndarray
        List of indices of `grid_x' and `grid_y' for the closest positions to those given in `x', `y'.

    """

    if np.ndim(x) != np.ndim(y):
        raise Exception("Number of points in `x' and `y' do not match")

    grid_xy = np.array((grid_x, grid_y)).T

    if np.ndim(x) == 0:
        search_xy = np.column_stack([x, y])
    else:
        search_xy = np.array((x, y)).T

    kdtree = scipy.spatial.cKDTree(grid_xy)
    dist, indices = kdtree.query(search_xy)
    # Replace positions outside the grid with NaNs. Should these simply be removed?
    if np.any(indices == len(grid_xy)):
        indices = indices.astype(float)
        indices[indices == len(grid_xy)] = np.nan
    # Replace positions beyond the given distance threshold with NaNs.
    if np.any(dist > maxDistance):
        indices = indices.astype(float)
        indices[dist > maxDistance] = np.nan

    # To maintain backwards compatibility, we need to return a value for every input position. We return NaN for
    # values outside the threshold distance (if given) or domain.
    nearest_x, nearest_y = np.empty(len(indices)) * np.nan, np.empty(len(indices)) * np.nan
    nearest_x[~np.isnan(indices)] = grid_x[indices[~np.isnan(indices)].astype(int)]
    nearest_y[~np.isnan(indices)] = grid_y[indices[~np.isnan(indices)].astype(int)]

    return nearest_x, nearest_y, dist, indices


def element_side_lengths(triangles, x, y):
    """
    Given a list of triangle nodes, calculate the length of each side of each
    triangle and return as an array of lengths. Units are in the original input
    units (no conversion from lat/long to metres, for example).

    The arrays triangles, x and y can be created by running read_sms_mesh(),
    read_fvcom_mesh() or read_mike_mesh() on a given SMS, FVCOM or MIKE grid
    file.

    Parameters
    ----------
    triangles : np.ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in `x' and `y' (see below). Give as
        a zero-indexed array.
    x, y : np.ndarray
        Coordinates of each grid node.

    Returns
    -------
    element_sides : np.ndarray
        Length of each element described by triangles and x, y.

    """

    element_sides = np.zeros([np.shape(triangles)[0], 3])
    for it, tri in enumerate(triangles):
        pos1x, pos2x, pos3x = x[tri]
        pos1y, pos2y, pos3y = y[tri]

        element_sides[it, 0] = np.sqrt((pos1x - pos2x)**2 + (pos1y - pos2y)**2)
        element_sides[it, 1] = np.sqrt((pos2x - pos3x)**2 + (pos2y - pos3y)**2)
        element_sides[it, 2] = np.sqrt((pos3x - pos1x)**2 + (pos3y - pos1y)**2)

    return element_sides


def mesh2grid(mesh_x, mesh_y, mesh_z, nx, ny, thresh=None, noisy=False):
    """
    Resample the unstructured grid in mesh_x and mesh_y onto a regular grid whose
    size is nx by ny or which is specified by the arrays nx, ny. Optionally
    specify dist to control the proximity of a value considered valid.

    Parameters
    ----------
    mesh_x, mesh_y : np.ndarray
        Arrays of the unstructured grid (mesh) node positions.
    mesh_z : np.ndarray
        Array of values to be resampled onto the regular grid. The shape of the
        array should have the nodes as the first dimension. All subsequent
        dimensions will be propagated automatically.
    nx, ny : int, np.ndarray
        Number of samples in x and y onto which to sample the unstructured
        grid. If given as a list or array, the values within the arrays are
        assumed to be positions in x and y.
    thresh : float, optional
        Distance beyond which a sample is considered too far from the current
        node to be included.
    noisy : bool, optional
        Set to True to enable verbose messages.

    Returns
    -------
    xx, yy : np.ndarray
        New position arrays (1D). Can be used with numpy.meshgrid to plot the
        resampled variables with matplotlib.pyplot.pcolor.
    zz : np.ndarray
        Array of the resampled data from mesh_z. The first dimension from the
        input is now replaced with two dimensions (x, y). All other input
        dimensions follow.

    """

    if not thresh:
        thresh = np.inf

    # Get the extents of the input data.
    xmin, xmax, ymin, ymax = mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()

    if isinstance(nx, int) and isinstance(ny, int):
        xx = np.linspace(xmin, xmax, nx)
        yy = np.linspace(ymin, ymax, ny)
    else:
        xx = nx
        yy = ny

    # We need to check the input we're resampling for its number of dimensions
    # so we can create an output array of the right shape. We can just take the
    # shape of the input, omitting the first value (the nodes). That should
    # leave us with the right shape. This even works for 1D inputs (i.e. a
    # single value at each unstructured grid location).
    if isinstance(nx, int) and isinstance(ny, int):
        zz = np.empty((nx, ny) + mesh_z.shape[1:]) * np.nan
    else:
        zz = np.empty((nx.shape) + mesh_z.shape[1:]) * np.nan

    if noisy:
        if isinstance(nx, int) and isinstance(ny, int):
            print('Resampling unstructured to regular ({} by {}). '.format(nx, ny), end='')
            print('Be patient...')
        else:
            _nx, _ny = len(nx[:, 1]), len(ny[0, :])
            print('Resampling unstructured to regular ({} by {}). '.format(_nx, _ny), end='')
            print('Be patient...')

        sys.stdout.flush()

    if isinstance(nx, int) and isinstance(ny, int):
        for xi, xpos in enumerate(xx):
            # Do all the y-positions with findNearestPoint
            for yi, ypos in enumerate(yy):
                # Find the nearest node in the unstructured grid data and grab
                # its u and v values. If it's beyond some threshold distance,
                # leave the z value as NaN.
                dist = np.sqrt((mesh_x - xpos)**2 + (mesh_y - ypos)**2)

                # Get the index of the minimum and extract the values only if
                # the nearest point is within the threshold distance (thresh).
                if dist.min() < thresh:
                    idx = dist.argmin()

                    # The ... means "and all the other dimensions". Since we've
                    # asked for our input array to have the nodes as the first
                    # dimension, this means we can just get all the others when
                    # using the node index.
                    zz[xi, yi, ...] = mesh_z[idx, ...]
    else:
        # We've been given positions, so run through those instead of our
        # regularly sampled grid.
        c = 0
        for ci, _ in enumerate(xx[0, :]):
            for ri, _ in enumerate(yy[:, 0]):
                if noisy:
                    if np.mod(c, 1000) == 0 or c == 0:
                        print('{} of {}'.format(
                            c,
                            len(xx[0, :]) * len(yy[:, 0])
                            ))
                c += 1

                dist = np.sqrt(
                    (mesh_x - xx[ri, ci])**2 + (mesh_y - yy[ri, ci])**2
                )
                if dist.min() < thresh:
                    idx = dist.argmin()
                    zz[ri, ci, ...] = mesh_z[idx, ...]

    if noisy:
        print('done.')

    return xx, yy, zz


def line_sample(x, y, positions, num=0, return_distance=False, noisy=False):
    """
    Function to take an unstructured grid of positions x and y and find the
    points which fall closest to a line defined by the coordinate pairs
    `positions'.

    If num=0 (default), then the line will be sampled at each nearest node; if
    num is greater than 1, then the line will be subdivided into num segments
    (at num + 1 points), and the closest point to that line used as the sample.

    Returns a list of array indices.

    N.B. Most of the calculations assume we're using cartesian coordinates.
    Things might get a bit funky if you're using spherical coordinates. Might
    be easier to convert before using this.

    Parameters
    ----------
    x, y : np.ndarray
        Position arrays for the unstructured grid.
    positions : np.ndarray
        Coordinate pairs of the sample line coordinates [[xpos, ypos], ...,
        [xpos, ypos]].  Units must match those in (x, y).
    num : int, optional
        Optionally specify a number of points to sample along the line
        described in `positions'. If no number is given, then the sampling of
        the line is based on the closest nodes to that line.
    return_distance : bool, optional
        Set to True to return the distance along the sampling line. Defaults
        to False.
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    idx : list
        List of indices for the nodes used in the line sample.
    line : np.ndarray
        List of positions which fall along the line described by `positions'.
        These are the projected positions of the nodes which fall closest to
        the line (not the positions of the nodes themselves).
    distance : np.ndarray, optional
        If `return_distance' is True, return the distance along the line
        described by the nodes in idx.

    """

    if not isinstance(num, int):
        raise TypeError('num must be an int')

    def __nodes_on_line__(xs, ys, start, end, pdist, noisy=False):
        """
        Child function to find all the points within the coordinates in xs and
        ys which fall along the line described by the coordinate pairs start
        and end. Uses pdist (distance of all coordinates from the line [start,
        end]) to select nodes.

        Parameters
        ----------
        xs, ys : np.ndarray
            Node position arrays.
        start, end : np.ndarray
            Coordinate pairs for the start and end of the sample line.
        pdist : np.ndarray
            Distance of the nodes in xs and ys from the line defined by
            `start' and `end'.

        Returns
        -------
        idx : list
            List of indices for the nodes used in the line sample.
        line : np.ndarray
            List of positions which fall along the line described by (start,
            end). These are the projected positions of the nodes which fall
            closest to the line (not the positions of the nodes themselves).

        """

        # Create empty lists for the indices and positions.
        sidx = []
        line = []

        beg = start  # seed the position with the start of the line.

        while True:

            # Find the nearest point to the start which hasn't already been
            # used (if this is the first iteration, we need to use all the
            # values).
            sx = np.ma.array(xs, mask=False)
            sy = np.ma.array(ys, mask=False)
            sx.mask[sidx] = True
            sy.mask[sidx] = True
            ndist = np.sqrt((sx - beg[0])**2 + (sy - beg[1])**2)

            # Create an array summing the distance from the current node with
            # the distance of all nodes to the line.
            sdist = ndist + pdist

            # Add the closest index to the list of indices. In some unusual
            # circumstances, the algorithm ends up going back along the line.
            # As such, add a check for the remaining distance to the end, and
            # if we're going backwards, find the next best point and use that.

            # Closest node index.
            tidx = sdist.argmin().astype(int)
            # Distance from the start point.
            fdist = np.sqrt((start[0] - xx[tidx])**2 + (start[1] - yy[tidx])**2).min()
            # Distance to the end point.
            tdist = np.sqrt((end[0] - xx[tidx])**2 + (end[1] - yy[tidx])**2).min()
            # Last node's distance to the end point.
            if len(sidx) >= 1:
                oldtdist = np.sqrt((end[0] - xx[sidx[-1]])**2 + (end[1] - yy[sidx[-1]])**2).min()
            else:
                # Haven't found any points yet.
                oldtdist = tdist

            if fdist >= length:
                # We've gone beyond the end of the line, so don't bother trying
                # to find another node.  Leave the if block so we actually add
                # the current index and position to sidx and line. We'll break
                # out of the main while loop a bit later.
                break

            elif tdist > oldtdist:
                # We're moving away from the end point. Find the closest point
                # in the direction of the end point.
                c = 0
                sdistidx = np.argsort(sdist)

                while True:
                    try:
                        tidx = sdistidx[c]
                        tdist = np.sqrt((end[0] - xx[tidx])**2 + (end[1] - yy[tidx])**2).min()
                        c += 1
                    except IndexError:
                        # Eh, we've run out of indices for some reason. Let's
                        # just go with whatever we had as the last set of
                        # values.
                        break

                    if tdist < oldtdist:
                        break

            sidx.append(tidx)

            line.append([xx[tidx], yy[tidx]])

            if noisy:
                done = 100 - ((tdist / length) * 100)
                if len(sidx) == 1:
                    print('Found {} node ({:.2f}%)'.format(len(sidx), done))
                else:
                    print('Found {} nodes ({:.2f}%)'.format(len(sidx), done))

            # Check if we've gone beyond the end of the line (by checking the
            # length of the sampled line), and if so, break out of the loop.
            # Otherwise, carry on.
            if beg.tolist() == start.tolist() or fdist <= length:
                # Reset the beginning point for the next iteration if we're at
                # the start or within the line extent.
                beg = np.array(([xx[tidx], yy[tidx]]))
            else:
                if noisy:
                    print('Reached the end of the line segment')

                break

        # Convert the list to an array before we leave.
        line = np.asarray(line)

        return sidx, line

    # To do multi-segment lines, we'll break each one down into a separate
    # line and do those sequentially. This means I don't have to rewrite
    # masses of the existing code and it's still pretty easy to understand (for
    # me at least!).
    nlocations = len(positions)

    idx = []
    line = []
    if return_distance:
        dist = []

    for xy in range(1, nlocations):
        # Make the first segment.
        start = positions[xy - 1]
        end = positions[xy]

        # Get the lower left and upper right coordinates of this section of the
        # line.
        lowerx = min(start[0], end[0])
        lowery = min(start[1], end[1])
        upperx = max(start[0], end[0])
        uppery = max(start[1], end[1])
        ll = [lowerx, lowery]
        ur = [upperx, uppery]

        lx = float(end[0] - start[0])
        ly = float(end[1] - start[1])
        length = np.sqrt(lx**2 + ly**2)
        dcn = np.degrees(np.arctan2(lx, ly))

        if num > 1:
            # This is easy: decimate the line between the start and end and
            # find the grid nodes which fall closest to each point in the line.

            # Create the line segments
            inc = length / num
            xx = start[0] + (np.cumsum(np.hstack((0, np.repeat(inc, num)))) *
                             np.sin(np.radians(dcn)))
            yy = start[1] + (np.cumsum(np.hstack((0, np.repeat(inc, num)))) *
                             np.cos(np.radians(dcn)))
            [line.append(xy) for xy in zip([xx, yy])]

            # For each position in the line array, find the nearest indices in
            # the supplied unstructured grid. We'll use our existing function
            # findNearestPoint for this.
            _, _, _, tidx = find_nearest_point(x, y, xx, yy)
            [idx.append(i) for i in tidx.tolist()]

        else:
            # So really, this shouldn't be that difficult, all we're doing is
            # finding the intersection of two lines which are orthogonal to one
            # another. We basically need to find the equations of both lines
            # and then solve for the intersection.

            # First things first, clip the coordinates to a rectangle defined
            # by the start and end coordinates. We'll use a buffer based on the
            # size of the elements which surround the first and last nodes.
            # This ensures we'll get relatively sensible results if the profile
            # is relatively flat or vertical. Use the six closest nodes as the
            # definition of surrounding elements.
            bstart = np.mean(np.sort(np.sqrt((x - start[0])**2 +
                                             (y - start[1])**2))[:6])
            bend = np.mean(np.sort(np.sqrt((x - end[0])**2 +
                                           (y - end[1])**2))[:6])
            # Use the larger of the two lengths to be on the safe side.
            bb = 2 * np.max((bstart, bend))
            ss = np.where((x >= (ll[0] - bb)) * (x <= (ur[0] + bb)) * (y >= (ll[1] - bb)) * (y <= (ur[1] + bb)))[0]
            xs = x[ss]
            ys = y[ss]

            # Sampling line equation.
            if lx == 0:
                # Vertical line.
                yy = ys
                xx = np.repeat(start[0], len(yy))

            elif ly == 0:
                # Horizontal line.
                xx = xs
                yy = np.repeat(start[1], len(xx))

            else:
                m1 = ly / lx  # sample line gradient
                c1 = start[1] - (m1 * start[0])  # sample line intercept

                # Find the equation of the line through all nodes in the domain
                # normal to the original line (gradient = -1 / m).
                m2 = -1 / m1
                c2 = ys - (m2 * xs)

                # Now find the intersection of the sample line and then all the
                # lines which go through the nodes.
                #   1a. y1 = (m1 * x1) + c1  # sample line
                #   2a. y2 = (m2 * x2) + c2  # line normal to it
                # Rearrange 1a for x.
                #   1b. x1 = (y1 - c1) / m1

                # Substitute equation 1a (y1) into 2a and solve for x.
                xx = (c2 - c1) / (m1 - m2)
                # Substitute xx into 2a to solve for y.
                yy = (m2 * xx) + c2

            # Find the distance from the original nodes to their corresponding
            # projected node.
            pdist = np.sqrt((xx - xs)**2 + (yy - ys)**2)

            # Now we need to start our loop until we get beyond the end of the
            # line.
            tidx, tline = __nodes_on_line__(xs, ys,
                                            start, end,
                                            pdist,
                                            noisy=noisy)

            # Now, if we're being asked to return the distance along the
            # profile line (rather than the distance along the line
            # connecting the positions in xs and ys together), generate that
            # for this segment here.
            if return_distance:
                # Make the distances relative to the first node we've found.
                # Doing this, instead of using the coordinates given in start
                # means we don't end up with negative distances, which means
                # we don't have to worry about signed distance functions and
                # other fun things to get proper distance along the transect.
                xdist = np.diff(xx[tidx])
                ydist = np.diff(xx[tidx])
                tdist = np.hstack((0, np.cumsum(np.sqrt(xdist**2 + ydist**2))))
                # Make distances relative to the end of the last segment,
                # if we have one.
                if not dist:
                    distmax = 0
                else:
                    distmax = np.max(dist)
                [dist.append(i + distmax) for i in tdist.tolist()]

            [line.append(i) for i in tline.tolist()]
            # Return the indices in the context of the original input arrays so
            # we can more easily extract them from the main data arrays.
            [idx.append(i) for i in ss[tidx]]

    # Return the distance as a numpy array rather than a list.
    if return_distance:
        dist = np.asarray(dist)

    # Make the line list an array for easier plotting.
    line = np.asarray(line)

    if return_distance:
        return idx, line, dist
    else:
        return idx, line


def element_sample(xc, yc, positions):
    """
    Find the shortest path between the sets of positions using the unstructured grid triangulation.

    Returns element indices and a distance along the line (in metres).

    Parameters
    ----------
    xc, yc : np.ndarray
        Position arrays for the unstructured grid element centres (decimal degrees).
    positions : np.ndarray
        Coordinate pairs of the sample line coordinates np.array([[x1, y1], ..., [xn, yn]] in decimal degrees.

    Returns
    -------
    indices : np.ndarray
        List of indices for the elements used in the transect.
    distance : np.ndarray, optional
        The distance along the line in metres described by the elements in indices.

    Notes
    -----
    This is lifted and adjusted for use with PyFVCOM from PySeidon.utilities.shortest_element_path.

    """

    grid = np.array((xc, yc)).T

    triangulation = scipy.spatial.Delaunay(grid)

    # Create a set for edges that are indices of the points.
    edges = []
    for vertex in triangulation.vertices:
        # For each edge of the triangle, sort the vertices (sorting avoids duplicated edges being added to the set)
        # and add to the edges set.
        edge = sorted([vertex[0], vertex[1]])
        a = grid[edge[0]]
        b = grid[edge[1]]
        weight = (np.hypot(a[0] - b[0], a[1] - b[1]))
        edges.append((edge[0], edge[1], {'weight': weight}))

        edge = sorted([vertex[0], vertex[2]])
        a = grid[edge[0]]
        b = grid[edge[1]]
        weight = (np.hypot(a[0] - b[0], a[1] - b[1]))
        edges.append((edge[0], edge[1], {'weight': weight}))

        edge = sorted([vertex[1], vertex[2]])
        a = grid[edge[0]]
        b = grid[edge[1]]
        weight = (np.hypot(a[0] - b[0], a[1] - b[1]))
        edges.append((edge[0], edge[1], {'weight': weight}))

    # Make a graph based on the Delaunay triangulation edges.
    graph = networkx.Graph(edges)

    # List of elements forming the shortest path.
    elements = []
    for position in zip(positions[:-1], positions[1:]):
        # We need grid indices for networkx.shortest_path rather than positions, so for the current pair of positions,
        # find the closest element IDs.
        source = np.argmin(np.hypot(xc - position[0][0], yc - position[0][1]))
        target = np.argmin(np.hypot(xc - position[1][0], yc - position[1][1]))
        elements += networkx.shortest_path(graph, source=source, target=target, weight='weight')

    # Calculate the distance along the transect in metres (use the fast-but-less-accurate Haversine function rather
    # than the slow-but-more-accurate Vincenty distance function).
    distance = np.cumsum([0] + [haversine_distance((xc[i], yc[i]), (xc[i + 1], yc[i + 1])) for i in elements[:-1]])

    return np.asarray(elements), distance


def connectivity(p, t):
    """
    Assemble connectivity data for a triangular mesh.

    The edge based connectivity is built for a triangular mesh and the boundary
    nodes identified. This data should be useful when implementing FE/FV
    methods using triangular meshes.

    Parameters
    ----------
    p : np.ndarray
        Nx2 array of nodes coordinates, [[x1, y1], [x2, y2], etc.]
    t : np.ndarray
        Mx3 array of triangles as indices, [[n11, n12, n13], [n21, n22, n23],
        etc.]

    Returns
    -------
    e : np.ndarray
        Kx2 array of unique mesh edges - [[n11, n12], [n21, n22], etc.]
    te : np.ndarray
        Mx3 array of triangles as indices into e, [[e11, e12, e13], [e21, e22,
        e23], etc.]
    e2t : np.ndarray
        Kx2 array of triangle neighbours for unique mesh edges - [[t11, t12],
        [t21, t22], etc]. Each row has two entries corresponding to the
        triangle numbers associated with each edge in e. Boundary edges have
        e2t[i, 1] = -1.
    bnd : np.ndarray, bool
        Nx1 logical array identifying boundary nodes. p[i, :] is a boundary
        node if bnd[i] = True.

    Notes
    -----
    Python translation of the MATLAB MESH2D connectivity function by Darren
    Engwirda.

    """

    def _unique_rows(A, return_index=False, return_inverse=False):
        """
        Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
        where B is the unique rows of A and I and J satisfy
        A = B[J, :] and B = A[I, :]

        Returns I if return_index is True
        Returns J if return_inverse is True

        Taken from https://github.com/numpy/numpy/issues/2871

        """
        A = np.require(A, requirements='C')
        assert A.ndim == 2, "array must be 2-dim'l"

        B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
                      return_index=return_index,
                      return_inverse=return_inverse)

        if return_index or return_inverse:
            return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
                + B[1:]
        else:
            return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

    if p.shape[-1] != 2:
        raise Exception('p must be an Nx2 array')
    if t.shape[-1] != 3:
        raise Exception('t must be an Mx3 array')
    if np.any(t.ravel() < 0) or t.max() > p.shape[0] - 1:
        raise Exception('Invalid t')

    # Unique mesh edges as indices into p
    numt = t.shape[0]
    # Triangle indices
    vect = np.arange(numt)
    # Edges - not unique
    e = np.vstack(([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]))
    # Unique edges
    e, j = _unique_rows(np.sort(e, axis=1), return_inverse=True)
    # Unique edges in each triangle
    te = np.column_stack((j[vect], j[vect + numt], j[vect + (2 * numt)]))

    # Edge-to-triangle connectivity
    # Each row has two entries corresponding to the triangle numbers
    # associated with each edge. Boundary edges have e2t[i, 1] = -1.
    nume = e.shape[0]
    e2t = np.zeros((nume, 2)).astype(int) - 1
    for k in range(numt):
        for j in range(3):
            ce = te[k, j]
            if e2t[ce, 0] == -1:
                e2t[ce, 0] = k
            else:
                e2t[ce, 1] = k

    # Flag boundary nodes
    bnd = np.zeros((p.shape[0],)).astype(bool)
    # True for bnd nodes
    bnd[e[e2t[:, 1] == -1, :]] = True

    return e, te, e2t, bnd


def find_connected_nodes(n, triangles):
    """
    Return the IDs of the nodes surrounding node number `n'.

    Parameters
    ----------
    n : int
        Node ID around which to find the connected nodes.
    triangles : np.ndarray
        Triangulation matrix to find the connected nodes. Shape is [nele,
        3].

    Returns
    -------
    surroundingidx : np.ndarray
        Indices of the surrounding nodes.

    See Also
    --------
    PyFVCOM.grid.find_connected_elements().

    Notes
    -----

    Check it works with:
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy.spatial import Delaunay
    >>> x, y = np.meshgrid(np.arange(25), np.arange(100, 125))
    >>> x = x.flatten() + np.random.randn(x.size) * 0.1
    >>> y = y.flatten() + np.random.randn(y.size) * 0.1
    >>> tri = Delaunay(np.array((x, y)).transpose())
    >>> for n in np.linspace(1, len(x) - 1, 5).astype(int):
    ...     aa = surrounders(n, tri.vertices)
    ...     plt.figure()
    ...     plt.triplot(x, y, tri.vertices, zorder=20, alpha=0.5)
    ...     plt.plot(x[n], y[n], 'ro', label='central node')
    ...     plt.plot(x[aa], y[aa], 'ko', label='connected nodes')
    ...     plt.xlim(x[aa].min() - 1, x[aa].max() + 1)
    ...     plt.ylim(y[aa].min() - 1, y[aa].max() + 1)
    ...     plt.legend(numpoints=1)

    """

    eidx = np.max((np.abs(triangles - n) == 0), axis=1)
    surroundingidx = np.unique(triangles[eidx][triangles[eidx] != n])

    return surroundingidx


def find_connected_elements(n, triangles):
    """
    Return the IDs of the elements connected to node number `n'.

    Parameters
    ----------
    n : int or iterable
        Node ID(s) around which to find the connected elements. If more than
        one node is given, the unique elements for all nodes are returned.
        Order of results is not maintained.
    triangles : np.ndarray
        Triangulation matrix to find the connected elements. Shape is [nele,
        3].

    Returns
    -------
    surroundingidx : np.ndarray
        Indices of the surrounding elements.

    See Also
    --------
    PyFVCOM.grid.find_connected_nodes().

    """

    try:
        surroundingidx = []
        for ni in n:
            idx = np.argwhere(triangles == ni)[:, 0]
            surroundingidx.append(idx)
        surroundingidx = np.asarray([item for sublist in surroundingidx for item in sublist])
        surroundingidx = np.unique(surroundingidx)
    except TypeError:
        surroundingidx = np.argwhere(triangles == n)[:, 0]

    return surroundingidx


def get_area(v1, v2, v3):
    """ Calculate the area of a triangle/set of triangles.

    Parameters
    ----------
    v1, v2, v3 : tuple, list (float, float)
        Coordinate pairs (x, y) of the three vertices of a triangle. Can be 1D
        arrays of positions or lists of positions.

    Returns
    -------
    area : tuple, np.ndarray
        Area of the triangle(s). Units of v0, v1 and v2.

    Examples
    --------
    >>> v1 = ((4, 0), (0, 0))
    >>> v2 = ((10, -3), (2, 6))
    >>> v3 = ((7, 9), (10, -5))
    >>> a = get_area(v1, v2, v3)
    >>> print(a)
    [ 31.5  35. ]

    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v3 = np.asarray(v3)

    if np.size(v1) == 2:
        # Single position
        area = 0.5 * (v1[0] * (v2[1] - v3[1]) + v2[0] * (v3[1] - v1[1]) + v3[0] * (v1[1] - v2[1]))
    else:
        # Array of positions
        area = 0.5 * (v1[:, 0] * (v2[:, 1] - v3[:, 1]) + v2[:, 0] * (v3[:, 1] - v1[:, 1]) + v3[:, 0] * (v1[:, 1] - v2[:, 1]))

    return abs(area)


def get_area_heron(s1, s2, s3):
    """
    Calculate the area of a triangle/set of triangles based on side length (Herons formula). Could tidy by combining
    with get_area.

    Parameters
    ----------
    s1, s2, s3 : tuple, list (float, float)
        Side lengths of the three sides of a triangle. Can be 1D arrays of lengths or lists of lengths.

    Returns
    -------
    area : tuple, np.ndarray
        Area of the triangle(s). Units of v0, v1 and v2.

    """

    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    s3 = np.asarray(s3)

    p = 0.5 * (s1 + s2 + s3)

    area = np.sqrt(p * (p - s1) * (p - s2) * (p - s3))

    return abs(area)


def find_bad_node(nv, node_id):
    """
    Check nodes on the boundary of a grid for nodes connected to a single
    element only. These elements will always have zero velocities,
    which means rivers input here will cause the model crash as the water
    never advects away. It is also computationally wasteful to include these
    elements.

    Return True if it was bad, False otherwise.

    Parameters
    ----------
    nv : np.ndarray
        Connectivity table for the grid.
    node_id : int
        Node ID to check.

    Returns
    -------
    bad : bool
        Status of the supplied node ID: True is connected to a single element
        only (bad), False is connected to multiple elements (good).

    Examples
    --------
    >>> from PyFVCOM.grid import read_sms_mesh, connectivity
    >>> nv, nodes, x, y, z, _ = read_sms_mesh('test_mesh.2dm')
    >>> _, _, _, bnd = connectivity(np.asarray((x, y)).transpose(), nv)
    >>> coast_nodes = nodes[bnd]
    >>> bad_nodes = np.empty(coast_nodes.shape).astype(bool)
    >>> for i, node_id in enumerate(coast_nodes):
    >>>     bad_nodes[i] = find_bad_node(node_id, nv)

    """

    was = False
    if len(np.argwhere(nv == node_id)) == 1:
        was = True

    return was


def trigradient(x, y, z, t=None):
    """
    Returns the gradient of `z' defined on the irregular mesh with Delaunay
    triangulation `t'. `dx' corresponds to the partial derivative dZ/dX,
    and `dy' corresponds to the partial derivative dZ/dY.

    Parameters
    ----------
    x, y, z : array_like
        Horizontal (`x' and `y') positions and vertical position (`z').
    t : array_like, optional
        Connectivity table for the grid. If omitted, one will be calculated
        automatically.

    Returns
    -------
    dx, dy : np.ndarray
        `dx' corresponds to the partial derivative dZ/dX, and `dy'
        corresponds to the partial derivative dZ/dY.

    Example
    -------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from PyFVCOM.grid import trigradient
    >>> from matplotlib.tri.triangulation import Triangulation
    >>> x, y = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
    >>> x[1:2:-1, :] = x[1:2:-1, :] + 0.1 / 2
    >>> tt = Triangulation(x.ravel(), y.ravel())
    >>> z = x * np.exp(-x**2 - y**2)
    >>> dx, dy = trigradient(x.ravel(), y.ravel(), z.ravel())
    >>> dzdx = (1 / x - 2 * x) * z
    >>> dzdy = -2 * y * z
    >>> plt.figure(1)
    >>> plt.quiver(x.ravel(), y.ravel(), dzdx.ravel(), dzdy.ravel(),
    >>>            color='r', label='Exact')
    >>> plt.quiver(x.ravel(), y.ravel(), dx, dy,
    >>>            color='k', label='trigradient', alpha=0.5)
    >>> tp = plt.tripcolor(x.ravel(), y.ravel(), tt.triangles, z.ravel(),
    >>>                    zorder=0)
    >>> plt.colorbar(tp)
    >>> plt.legend()

    Notes
    -----
    Lifted from:
        http://matplotlib.org/examples/pylab_examples/trigradient_demo.html

    """

    if np.any(t):
        tt = Triangulation(x.ravel(), y.ravel(), t)
    else:
        tt = Triangulation(x.ravel(), y.ravel())

    tci = CubicTriInterpolator(tt, z.ravel())
    # Gradient requested here at the mesh nodes but could be anywhere else:
    dx, dy = tci.gradient(tt.x, tt.y)

    return dx, dy


def rotate_points(x, y, origin, angle):
    """
    Rotate the points in `x' and `y' around the point `origin' by `angle'
    degrees.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinates to rotate.
    origin : list, np.ndarray
        Point about which to rotate the grid (x, y).
    angle : float
        Angle (in degrees) by which to rotate the grid. Positive clockwise.

    Returns
    -------
    xr, yr : np.ndarray
        Rotated coordinates.

    """

    # Make the x and y values relative to the origin.
    x -= origin[0]
    y -= origin[1]

    # Rotate clockwise by some angle `rot'. See
    # http://stackoverflow.com/questions/29708840 for a faster version on 2D
    # arrays with np.einsum if necessary in future.
    xr = np.cos(np.deg2rad(angle)) * x + np.sin(np.deg2rad(angle)) * y
    yr = -np.sin(np.deg2rad(angle)) * x + np.cos(np.deg2rad(angle)) * y

    # Add the origin back to restore the right coordinates.
    xr += origin[0]
    yr += origin[1]

    return xr, yr


def get_boundary_polygons(triangle, noisy=False, nodes=None):
    """
    Gets a list of the grid boundary nodes ordered correctly.

    ASSUMPTIONS: This function assumes a 'clean' FVCOM grid, i.e. no
    elements with 3 boundary nodes and no single element width channels.

    Parameters
    ----------
    triangle : np.ndarray
        The triangle connectivity matrix as produced by the read_fvcom_mesh
        function.

    nodes : optional, np.ndarray
        Optionally a Nx2 array of coordinates for nodes in the grid, if passed the function will
        additionally return a boolean of whether the polygons are boundaries (domain on interior)
        or islands (domain on the exterior)

    Returns
    -------
    boundary_polygon_list : list
        List of integer arrays. Each array is one closed boundary polygon with
        the integers referring to node number.

    islands_list : list
        Optional, only returned if an array is passed for nodes. A boolean list of whether the
        polygons are boundaries (domain on interior) or islands (domain on the exterior)

    """

    u, c = np.unique(triangle, return_counts=True)
    uc = np.asarray([u, c]).T

    nodes_lt_4 = np.asarray(uc[uc[:, 1] < 4, 0], dtype=int)
    boundary_polygon_list = []

    # Pretty certain we can use `while np.any(nodes_lt_4)` below instead.
    while len(nodes_lt_4) > 0:

        start_node = nodes_lt_4[0]

        boundary_node_list = [start_node, get_attached_unique_nodes(start_node, triangle)[-1]]

        full_loop = True
        while full_loop:
            next_nodes = get_attached_unique_nodes(boundary_node_list[-1], triangle)
            node_ind = 0
            len_bl = len(boundary_node_list)
            if noisy:
                print(len_bl)
            while len(boundary_node_list) == len_bl:
                try:
                    if next_nodes[node_ind] not in boundary_node_list:
                        boundary_node_list.append(next_nodes[node_ind])
                    else:
                        node_ind += 1
                except:
                    full_loop = False
                    len_bl += 1

        boundary_polygon_list.append(np.asarray(boundary_node_list))
        nodes_lt_4 = np.asarray(list(set(nodes_lt_4) - set(boundary_node_list)), dtype=int)

    if nodes is None:
        return boundary_polygon_list

    else:
        all_poly_nodes = np.asarray([y for x in boundary_polygon_list for y in x])
        reduce_nodes =  nodes[~all_poly_nodes, :]
        reduce_nodes_pts = [shapely.geometry.Point(this_ll) for this_ll in reduce_nodes]

        islands_list = []
        for this_poly_nodes in boundary_polygon_list:
            this_poly = shapely.geometry.Polygon(nodes[this_poly_nodes, :])
            this_poly_contain = np.asarray([this_poly.contains(this_pt) for this_pt in reduce_nodes_pts])

            if np.any(this_poly_contain):
                islands_list.append(False)
            else:
                islands_list.append(True)

        return [boundary_polygon_list, islands_list]


def get_attached_unique_nodes(this_node, trinodes):
    """
    Find the nodes on the boundary connected to `this_node'.

    Parameters
    ----------
    this_node : int
        Node ID.
    trinodes : np.ndarray
        Triangulation table for an unstructured grid.

    Returns
    -------
    connected_nodes : np.ndarray
        IDs of the nodes connected to `this_node' on the boundary. If `this_node' is not on the boundary,
        `connected_nodes' is empty.

    """

    all_trinodes = trinodes[(trinodes[:, 0] == this_node) | (trinodes[:, 1] == this_node) | (trinodes[:, 2] == this_node), :]
    u, c = np.unique(all_trinodes, return_counts=True)

    return u[c == 1]


def grid_metrics(tri, noisy=False):
    """
    Calculate unstructured grid metrics (most of FVCOM's tge.F).

    Parameters
    ----------
    tri : np.ndarray
        Triangulation table for the grid.
    noisy : bool
        Set to True to enable verbose output (default = False)

    Returns
    -------
    ntve : np.ndarray
        The number of neighboring elements of each grid node
    nbve : np.ndarray
        nbve(i, 1->ntve(i)) = ntve elements containing node i
    nbe : np.ndarray
        Indices of tri for the elements connected to each element in the domain. To visualise:
            plt.plot(x[tri[1000, :], y[tri[1000, :], 'ro')
            plt.plot(x[tri[nbe[1000], :]] and y[tri[nbe[1000], :]], 'k.')
        plots the 999th element nodes with the nodes of the surrounding elements too.
    isbce : np.ndarray
        Flag if element is on the boundary (True = yes, False = no)
    isonb : np.ndarray
        Flag if node is on the boundary (True = yes, False = no)

    Notes
    -----
    This is more or less a direct translation from FORTRAN (FVCOM's tge.F).

    """

    m = len(np.unique(tri.ravel()))

    # Allocate all our arrays. Use masked by default arrays so we only use valid indices.
    isonb = np.zeros(m).astype(bool)
    ntve = np.zeros(m, dtype=int)
    nbe = np.ma.array(np.zeros(tri.shape, dtype=int), mask=True)
    nbve = np.ma.array(np.zeros((m, 10), dtype=int), mask=True)
    # Number of elements connected to each node (ntve) and the IDs of the elements connected to each node (nbve).
    if noisy:
        print('Counting neighbouring nodes and elements')
    for i, (n1, n2, n3) in enumerate(tri):
        nbve[tri[i, 0], ntve[n1]] = i
        nbve[tri[i, 1], ntve[n2]] = i
        nbve[tri[i, 2], ntve[n3]] = i
        # Only increment the counters afterwards as Python indexes from 0.
        ntve[n1] += 1
        ntve[n2] += 1
        ntve[n3] += 1

    if noisy:
        print('Getting neighbouring elements for each element')
    # Get the element IDs connected to each element.
    for i, (n1, n2, n3) in enumerate(tri):
        for j1 in range(ntve[n1]):
            for j2 in range(ntve[n2]):
                if nbve[n1, j1] == nbve[n2, j2] and nbve[n1, j1] != i:
                    nbe[i, 2] = nbve[n1, j1]
        for j2 in range(ntve[n2]):
            for j3 in range(ntve[n3]):
                if nbve[n2, j2] == nbve[n3, j3] and nbve[n2, j2] != i:
                    nbe[i, 0] = nbve[n2, j2]
        for j1 in range(ntve[n1]):
            for j3 in range(ntve[n3]):
                if nbve[n1, j1] == nbve[n3, j3] and nbve[n1, j1] != i:
                    nbe[i, 1] = nbve[n3, j3]

    if noisy:
        print('Getting boundary element IDs')
    isbce = np.max(nbe.mask, axis=1)

    if noisy:
        print('Getting boundary node IDs')

    # Get the boundary node IDs.
    boundary_element_node_ids = np.unique(tri[isbce, :]).ravel()
    boundary_nodes = []
    for i in boundary_element_node_ids:
        current_nodes = get_attached_unique_nodes(i, tri)
        if np.any(current_nodes):
            boundary_nodes += current_nodes.tolist()
    boundary_nodes = np.unique(boundary_nodes)
    # Make a boolean of that.
    isonb[boundary_nodes] = True

    return ntve, nbve, nbe, isbce, isonb


def control_volumes(x, y, tri, node_control=True, element_control=True, noisy=False, poolsize=None, **kwargs):
    """
    This calculates the surface area of individual control volumes consisted of triangles with a common node point.

    Parameters
    ----------
    x, y : np.ndarray
        Node positions
    tri : np.ndarray
        Triangulation table for the unstructured grid.
    node_control : bool
        Set to False to disable calculation of node control volumes. Defaults to True.
    element_control : bool
        Set to False to disable calculation of element control volumes. Defaults to True.
    noisy : bool
        Set to True to enable verbose output.

    Additional kwargs are passed to `PyFVCOM.grid.node_control_area`.

    Returns
    -------
    art1 : np.ndarray, optional
        Area of interior control volume (for node value integration). Omitted if node_control is False.
    art2 : np.ndarray, optional
        Sum area of all cells around each node. Omitted if element_control is False.
    art1_points : np.ndarray, optional
        If return_points is set to True (an argument to `PyFVCOM.grid.node_control_area`), then those are returned as
        the second or third argument (with element_control = False, with element_control = True respectively).

    Notes
    -----
    This is a python reimplementation of the FVCOM function CELL_AREA in cell_area.F. Whilst the reimplementation is
    coded with efficiency in mind (the calculations occur in parallel), this is still slow for large grids. Please be
    patient!

    """

    if not node_control and not element_control:
        raise ValueError("Set either `node_control' or `element_control' to `True'")

    if poolsize is None:
        pool = multiprocessing.Pool()
    elif poolsize == 'serial':
        pool = False
    else:
        pool = multiprocessing.Pool(poolsize)

    m = len(x)  # number of nodes

    # Calculate art1 (control volume for fluxes of node-based values). I do this differently from how it's done in
    # FVCOM as I can't wrap my head around the seemingly needlessly complicated approach they've taken. Here,
    # my approach is:
    #   1. For each node, find all the elements connected to it (find_connected_elements).
    #   2. Identify the nodes in all those elements.
    #   3. Find the position of the halfway point along each vertex between our current node and all the other nodes
    #   connected to it.
    #   4. Using those positions and the positions of the centre of each element, create a polygon (ordered clockwise).
    if node_control:
        if noisy:
            print('Compute control volume for fluxes at nodes (art1)')
        xc = nodes2elems(x, tri)
        yc = nodes2elems(y, tri)

        if not pool:
            art1 = []
            for this_node in range(m):
                art1.append(node_control_area(this_node, x, y, xc, yc, tri, **kwargs))
            art1 = np.asarray(art1)
        else:
            results = pool.map(partial(node_control_area, x=x, y=y, xc=xc, yc=yc, tri=tri, **kwargs), range(m))
            # Unpack the results in case we've been asked for the points too.
            if 'return_points' in kwargs:
                art1 = np.asarray([i[0] for i in results])
                art1_points = np.asarray([i[1] for i in results])
            else:
                art1 = np.asarray(results)

    # Compute area of control volume art2(i) = sum(all tris surrounding node i)
    if element_control:
        if noisy:
            print('Compute control volume for fluxes over elements (art2)')
        art = get_area(np.asarray((x[tri[:, 0]], y[tri[:, 0]])).T, np.asarray((x[tri[:, 1]], y[tri[:, 1]])).T, np.asarray((x[tri[:, 2]], y[tri[:, 2]])).T)
        if not pool:
            art2 = []
            for this_node in range(m):
                art2.append(element_control_area(this_node, triangles=tri, art=art))
            art2 = np.asarray(art2)
        else:
            art2 = np.asarray(pool.map(partial(element_control_area, triangles=tri, art=art), range(m)))

    if pool:
        pool.close()

    if node_control and element_control:
        if 'return_points' in kwargs:
            return art1, art2, art1_points
        else:
            return art1, art2
    elif node_control and not element_control:
        if 'return_points' in kwargs:
            return art1, art1_points
        else:
            return art1
    elif not node_control and element_control:
        return art2


def node_control_area(n, x, y, xc, yc, tri, return_points=False):
    """
    Worker function to calculate the control volume for fluxes of node-based values for a given node.

    Parameters
    ----------
    n : int
        Current node ID.
    x, y : list-like
        Node positions
    xc, yc : list-like
        Element centre positions
    tri : list-like
        Unstructured grid triangulation table.
    return_points : bool
        Return the coordinates of the points which form the node control area(s).

    Returns
    -------
    node_area : float
        Node control volume area in x or y length units squared.
    node_points : np.ndarray, optional
        Array of the (x, y) positions of the points which form the node control area. Only returned if
        `return_points' is set to True.

    """

    connected_elements = find_connected_elements(n, tri)
    area = 0
    # Create two triangles which are from the mid-point of each vertex with the current node and the element centre.
    # Sum the areas as we go.
    if return_points:
        control_area_points_x = []
        control_area_points_y = []
    for element in connected_elements:
        # Find the nodes in this element.
        connected_nodes = np.unique(tri[element, :])
        other_nodes = connected_nodes[connected_nodes != n]
        centre_x = xc[element]
        centre_y = yc[element]
        mid_x, mid_y = [], []
        for node in other_nodes:
            mid_x.append(x[n] - ((x[n] - x[node]) / 2))
            mid_y.append(y[n] - ((y[n] - y[node]) / 2))
        # Now calculate the area of the triangles formed by [(x[n], y[n]), (centre_x, centre_y), (mid_x, mid_y)].
        for mid_xy in zip(mid_x, mid_y):
            area += get_area((x[n], y[n]), mid_xy, (centre_x, centre_y))
        if return_points:
            control_area_points_x += [mid_x[0], centre_x, mid_x[1]]
            control_area_points_y += [mid_y[0], centre_y, mid_y[1]]

    if return_points:
        if area != 0:
            centre = (x[n], y[n])
            # If our current node is on the boundary of the grid, we need to add it to the polygon. Use
            # get_attached_unique_nodes to find the nodes on the model boundary connected to the current one. If we
            # get one, we're on the boundary, otherwise, we're in the domain. We could also use `connectivity',
            # but I think this is quicker since connectivity does a bunch of other stuff too.
            on_coast = get_attached_unique_nodes(n, tri)
            if np.any(on_coast):
                # Add the centre to the list of nodes we're using and also make a new centre for the ordering as it
                # breaks if you try to order clockwise around a point that's included in the list of points you're
                # ordering. We need to first sort what we've got then add the centre so we can get a valid Polygon.
                tmp_control_points = np.column_stack((control_area_points_x, control_area_points_y))
                control_area_points = clockwise(np.unique(tmp_control_points, axis=0), relative_to=centre)
                # Use a shapely polygon centroid.
                centre = np.asarray(shapely.geometry.Polygon(control_area_points).centroid.xy)

                control_area_points_x.append(x[n])
                control_area_points_y.append(y[n])
            control_area_points = np.column_stack((control_area_points_x, control_area_points_y))
            control_area_points = clockwise(np.unique(control_area_points, axis=0), relative_to=centre)
        else:
            control_area_points = None

        return area, control_area_points
    else:
        return area


def clockwise(points, relative_to=None):
    """
    Function to order points clockwise from north.

    Parameters
    ----------
    points : np.ndarray
        Array of positions (n, 2).
    relative_to : list-like, optional
        The point relative to which to rotate the points. If omitted, the centre of the convex hull of the points is
        used.

    Returns
    -------
    ordered_points : np.ndarray
        The points in clockwise order (n, 2).

    Notes
    -----

    This uses numpy.arctan2 meaning it might be slow for large numbers of points. There are faster algorithms out
    there (e.g. https://stackoverflow.com/a/6989383), but this is fairly easy to understand.

    """

    if relative_to is None:
        try:
            boundary = scipy.spatial.ConvexHull(points)
            # Use that to create a shapely polygon with lots of useful bits in it. Make sure it's closed by
            # repeating the first vertex.
            vertices = np.append(boundary.vertices, boundary.vertices[0, :])
            coverage_polygon = shapely.geometry.Polygon(points[vertices])
            relative_to = coverage_polygon.centroid()
        except (QhullError, IndexError):
            # We've probably got fewer than three points or we've got points in a straight line, which means
            # the coverage here is zero.
            raise ValueError('Valid points in sample are colinear or too few in number for a valid polygon to be created.')

    ordered_points = np.asarray(sorted(points, key=lambda x: np.arctan2(x[1] - relative_to[1], x[0] - relative_to[0])))

    return ordered_points


def element_control_area(node, triangles, art):
    """
    Worker function to calculate the control volume for fluxes of element-based values for a given node.

    Parameters
    ----------
    node : int
        Node ID.
    triangles : list-like
        Unstructured grid triangulation table.
    art : list-like
        Element areas.

    Returns
    -------
    element_area : float
        Element control volume area in x or y length units squared.

    Notes
    -----

    """

    connected_elements = find_connected_elements(node, triangles)

    return np.sum(art[connected_elements])


def unstructured_grid_volume(area, depth, surface_elevation, thickness, depth_integrated=False):
    """
    Calculate the volume for every cell in the unstructured grid.

    Parameters
    ----------
    area : np.ndarray
        Element area
    depth : np.ndarray
        Static water depth
    surface_elevation : np.ndarray
        Time-varying surface elevation
    thickness : np.ndarray
        Level (i.e. between layer) position (range 0-1). In FVCOM, this is siglev.
    depth_intergrated : bool, optional
        Set to True to return the depth-integrated volume in addition to the depth-resolved volume. Defaults to False.

    Returns
    -------
    depth_volume : np.ndarray
        Depth-resolved volume of all the elements with time.
    volume : np.ndarray, optional
        Depth-integrated volume of all the elements with time.

    """

    # Convert thickness to actual thickness rather than position in water column of the layer.
    dz = np.abs(np.diff(thickness, axis=0))
    volume = (area * (surface_elevation + depth))
    depth_volume = volume[:, np.newaxis, :] * dz[np.newaxis, ...]

    if depth_integrated:
        return depth_volume, volume
    else:
        return depth_volume


def unstructured_grid_depths(h, zeta, sigma, nan_invalid=False):
    """
    Calculate the depth time series for cells in an unstructured grid.

    Parameters
    ----------
    h : np.ndarray
        Water depth
    zeta : np.ndarray
        Surface elevation time series
    sigma : np.ndarray
        Sigma vertical distribution, range 0-1 (`siglev' or `siglay' in FVCOM)
    nan_invalid : bool, optional
        Set values shallower than the mean sea level (`h') to NaN. Defaults to not doing that.

    Returns
    -------
    depths : np.ndarray
        Time series of model depths.

    """

    if nan_invalid:
        invalid = -zeta > h
        zeta[invalid] = np.nan

    abs_water_depth = zeta + h
    # Add zeta again so the range is surface elevation (`zeta') to mean water depth rather (`h') than zero to water
    # depth (`h' + `zeta') which is much more useful for plotting.
    depths = abs_water_depth[:, np.newaxis, :] * sigma[np.newaxis, ...] + zeta[:, np.newaxis, :]

    return depths


def elems2nodes(elems, tri, nvert=None):
    """
    Calculate a nodal value based on the average value for the elements
    of which it is a part. This necessarily involves an average, so the
    conversion from nodes2elems and elems2nodes is not reversible.

    Parameters
    ----------
    elems : np.ndarray
        Array of unstructured grid element values to move to the element
        nodes.
    tri : np.ndarray
        Array of shape (nelem, 3) comprising the list of connectivity
        for each element.
    nvert : int, optional
        Number of nodes (vertices) in the unstructured grid.

    Returns
    -------
    nodes : np.ndarray
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
    Calculate an element-centre value based on the average value for the
    nodes from which it is formed. This involves an average, so the
    conversion from nodes to elements cannot be reversed without smoothing.

    Parameters
    ----------
    nodes : np.ndarray
        Array of unstructured grid node values to move to the element
        centres.
    tri : np.ndarray
        Array of shape (nelem, 3) comprising the list of connectivity
        for each element.

    Returns
    -------
    elems : np.ndarray
        Array of values at the grid nodes.

    """

    if np.ndim(nodes) == 1:
        elems = nodes[tri].mean(axis=-1)
    else:
        elems = nodes[..., tri].mean(axis=-1)

    return elems


def vincenty_distance(point1, point2, miles=False):
    """
    Vincenty's formula (inverse method) to calculate the distance (in
    kilometres or miles) between two points on the surface of a spheroid

    Parameters
    ----------
    point1 : list, tuple, np.ndarray
        Decimal degree longitude and latitude for the start.
    point2 : list, tuple, np.ndarray
        Decimal degree longitude and latitude for the end.
    miles : bool
        Set to True to return the distance in miles. Defaults to False (kilometres).

    Returns
    -------
    distance : float
        Distance between point1 and point2 in kilometres.

    Notes
    -----
    Author Maurycy Pietrzak (https://github.com/maurycyp/vincenty)

    """

    a = 6378137  # metres
    f = 1 / 298.257223563
    b = 6356752.314245  # metres; b = (1 - f)a

    miles_per_kilometre = 0.621371

    max_iterations = 200
    convergence_threshold = 1e-12  # .000,000,000,001

    # Short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0

    u1 = math.atan((1 - f) * math.tan(math.radians(point1[0])))
    u2 = math.atan((1 - f) * math.tan(math.radians(point2[0])))
    lambda_initial = math.radians(point2[1] - point1[1])
    lambda_current = copy.copy(lambda_initial)

    sin_u1 = math.sin(u1)
    cos_u1 = math.cos(u1)
    sin_u2 = math.sin(u2)
    cos_u2 = math.cos(u2)

    for iteration in range(max_iterations):
        sin_lambda = math.sin(lambda_current)
        cos_lambda = math.cos(lambda_current)
        sin_sigma = math.sqrt((cos_u2 * sin_lambda) ** 2 + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda) ** 2)
        if sin_sigma == 0:
            return 0.0  # coincident points
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2
        try:
            cos2sigma_m = cos_sigma - 2 * sin_u1 * sin_u2 / cos_sq_alpha
        except ZeroDivisionError:
            cos2sigma_m = 0
        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        lambda_prev = copy.copy(lambda_current)
        lambda_current = lambda_initial + (1 - C) * f * sin_alpha * (sigma + C * sin_sigma * (cos2sigma_m + C * cos_sigma * (-1 + 2 * cos2sigma_m**2)))
        if abs(lambda_current - lambda_prev) < convergence_threshold:
            break  # successful convergence
    else:
        return None  # failure to converge

    u_sq = cos_sq_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = B * sin_sigma * (cos2sigma_m + B / 4 * (cos_sigma * (-1 + 2 * cos2sigma_m ** 2) - B / 6 *
                                                          cos2sigma_m * (-3 + 4 * sin_sigma ** 2) *
                                                          (-3 + 4 * cos2sigma_m ** 2)))
    s = b * A * (sigma - delta_sigma)

    s /= 1000  # metres to kilometres
    if miles:
        s *= miles_per_kilometre  # kilometres to miles

    return round(s, 6)


def haversine_distance(point1, point2, miles=False):
    """
    Haversine function to calculate first order distance measurement. Assumes
    spherical Earth surface. Converted from MATLAB function:

    http://www.mathworks.com/matlabcentral/fileexchange/27785

    Parameters
    ----------
    point1 : list, tuple, np.ndarray
        Decimal degree longitude and latitude for the start.
    point2 : list, tuple, np.ndarray
        Decimal degree longitude and latitude for the end.
    miles : bool, optional
        Set to True to return the distance in miles. Defaults to False (kilometres).

    Returns
    -------
    distance : np.ndarray
        Distance between point1 and point2 in kilometres.

    """

    # Convert all decimal degree inputs to radians.
    point1 = np.deg2rad(point1)
    point2 = np.deg2rad(point2)

    R = 6371                           # Earth's mean radius in kilometres
    delta_lat = point2[1] - point1[1]  # difference in latitude
    delta_lon = point2[0] - point1[0]  # difference in longitude
    # Magic follows
    a = np.sin(delta_lat / 2)**2 + np.cos(point1[1]) * np.cos(point2[1]) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c                   # distance in kilometres

    if miles:
        distance *= 0.621371

    return distance


def shape_coefficients(xc, yc, nbe, isbce):
    """
    This function is used to calculate the coefficients for a linear function on the x-y plane:
        r(x, y; phai) = phai_c + cofa1 * x + cofa2 * y

    Unlike the FVCOM implementation, this returns NaNs for boundary elements (FVCOM returns zeros).

    Parameters
    ----------
    xc, yc : np.ndarray, list-like
        Element centre coordinates (cartesian coordinates)
    nbe : np.ndarray
        The three element IDs which surround each element.
    isbce : np.ndarray
        Flag for whether a given element is on the grid boundary.

    Returns
    -------
    a1u : np.ndarray
        Interpolant in the x direction.
    a2u : np.ndarray
        Interpolant in the y direction.

    Notes
    -----
    This is a more or less direct translation of the FVCOM subroutine SHAPE_COEFFICIENTS_GCN from shape_coef_gcn.F.
    There is probably a lot of optimisation to be done, but it seems to run in a not totally unreasonable amount of
    time as is, so I'm leaving it for now.

    """

    a1u = np.empty((len(xc), 4))
    a2u = np.empty((len(xc), 4))
    a1u[:] = np.nan
    a2u[:] = np.nan
    non_boundary_indices = np.arange(len(xc))[~isbce]
    for i in non_boundary_indices:
        y1 = (yc[nbe[i, 0]] - yc[i]) / 1000
        y2 = (yc[nbe[i, 1]] - yc[i]) / 1000
        y3 = (yc[nbe[i, 2]] - yc[i]) / 1000
        x1 = (xc[nbe[i, 0]] - xc[i]) / 1000
        x2 = (xc[nbe[i, 1]] - xc[i]) / 1000
        x3 = (xc[nbe[i, 2]] - xc[i]) / 1000

        delt = ((x1 * y2 - x2 * y1)**2 + (x1 * y3 - x3 * y1)**2 + (x2 * y3 - x3 * y2)**2) * 1000

        a1u[i, 0] = (y1 + y2 + y3) * (x1 * y1 + x2 * y2 + x3 * y3)- (x1 + x2 + x3) * (y1**2 + y2**2 + y3**2)
        a1u[i, 0] = a1u[i, 0] / delt
        a1u[i, 1] = (y1**2 + y2**2 + y3**2) * x1 - (x1 * y1 + x2 * y2 + x3 * y3) * y1
        a1u[i, 1] = a1u[i, 1] / delt
        a1u[i, 2] = (y1**2 + y2**2 + y3**2) * x2 - (x1 * y1 + x2 * y2 + x3 * y3) * y2
        a1u[i, 2] = a1u[i, 2] / delt
        a1u[i, 3] = (y1**2 + y2**2 + y3**2) * x3 - (x1 * y1 + x2 * y2 + x3 * y3) * y3
        a1u[i, 3] = a1u[i, 3] / delt

        a2u[i, 0] = (x1 + x2 + x3) * (x1 * y1 + x2 * y2 + x3 * y3)- (y1 + y2 + y3) * (x1**2 + x2**2 + x3**2)
        a2u[i, 0] = a2u[i, 0] / delt
        a2u[i, 1] = (x1**2 + x2**2 + x3**2) * y1 - (x1 * y1 + x2 * y2 + x3 * y3) * x1
        a2u[i, 1] = a2u[i, 1] / delt
        a2u[i, 2] = (x1**2 + x2**2 + x3**2) * y2 - (x1 * y1 + x2 * y2 + x3 * y3) * x2
        a2u[i, 2] = a2u[i, 2] / delt
        a2u[i, 3] = (x1**2 + x2**2 + x3**2) * y3 - (x1 * y1 + x2 * y2 + x3 * y3) * x3
        a2u[i, 3] = a2u[i, 3] / delt

    # Return transposed arrays to match what gets read in from a netCDF file.
    return a1u.T, a2u.T


def reduce_triangulation(tri, nodes, return_elements=False):
    """
    Returns the triangulation for a subset of grid nodes.

    Parameters
    ----------
    tri : np.ndarray Nx3
        Grid triangulation (e.g. triangle as returned from read_fvcom_mesh)
    nodes : np.ndarray M
        Selected subset of nodes for re-triangulating
    return_elements : bool, optional
        Return the index (integer array) of cells chosen from the original triangulation

    Returns
    -------
    reduced_tri : np.ndarray Mx3
        Triangulation for just the nodes listed in nodes.
    reduced_tri_elements : np.ndarrya M, optional
        If return_elements is specified then it returns additionally an array of element indices used by the new
        triangulation

    Notes
    -----
    Assumes the nodes selected are a contiguous part of the grid without any checking

    """

    # The node IDs must be sorted otherwise you can end up changing a given ID multiple times, which makes for quite
    # the mess, I can assure you.
    nodes = np.sort(nodes)

    reduced_tri = tri[np.all(np.isin(tri, nodes), axis=1), :]

    # Remap nodes to a new index. Use a copy of the reduced triangulation for the lookup so we avoid potentially
    # remapping a node twice.
    original_tri = reduced_tri.copy()
    new_index = np.arange(0, len(nodes))
    for this_old, this_new in zip(nodes, new_index):
        reduced_tri[original_tri == this_old] = this_new

    if return_elements:
        ele_ind = np.where(np.all(np.isin(tri, nodes), axis=1))[0]
        reduced_tri = [reduced_tri, ele_ind]

    return reduced_tri


def getcrossectiontriangles(cross_section_pnts, trinodes, X, Y, dist_res):
    """
    Subsamples the line defined by cross_section_pnts at the resolution dist_res on the grid defined by
    the triangulation trinodes, X, Y. Returns the location of the sub sampled points (sub_samp), which
    triangle they are in (sample_cells) and their nearest nodes (sample_nodes).

    Parameters
    ----------
    cross_section_pnts : 2x2 list_like
        The two ends of the cross section line.
    trinodes : list-like
        Unstructured grid triangulation table
    X, Y : list-like
        Node positions
    dist_res : float
        Approximate distance at which to sample the line

    Returns
    -------
    sub_samp : 2xN list
        Positions of sample points
    sample_cells : N list
        The cells within which the subsample points fall. -1 indicates that the point is outside the grid.
    sample_nodes : N list
        The nodes nearest the subsample points. -1 indicates that the point is outside the grid.

    Example
    -------

    TO DO
    -----

    Messy code. There definitely should be a more elegant version of this...
    Set up example and tests.

    """
    cross_section_x = [cross_section_pnts[0][0], cross_section_pnts[1][0]]
    cross_section_y = [cross_section_pnts[0][1], cross_section_pnts[1][1]]

    cross_section_dist = np.sqrt((cross_section_x[1] - cross_section_x[0])**2 + (cross_section_y[1] - cross_section_y[0])**2)
    res = np.ceil(cross_section_dist/dist_res)

    # first reduce the number of points to consider by only including triangles which cross the line through the two points
    tri_X = X[trinodes]
    tri_Y = Y[trinodes]

    # This section needs tidying up and making easier to understand!
    tri_cross_log_1_1 = np.logical_or(np.logical_and(tri_X.min(1) < min(cross_section_x), tri_X.max(1) > max(cross_section_x)),
                                      np.logical_and(tri_Y.min(1) < min(cross_section_y), tri_Y.max(1) > max(cross_section_y)))

    tri_cross_log_1_2 = np.any(np.logical_and(np.logical_and(tri_X < max(cross_section_x), tri_X > min(cross_section_x)), np.logical_and(tri_Y < max(cross_section_y), tri_Y > min(cross_section_y))), axis=1)
    tri_cross_log_1 = np.logical_or(tri_cross_log_1_1, tri_cross_log_1_2)

    tri_cross_log_1_2 = np.any(np.logical_and(np.logical_and(tri_X < max(cross_section_x), tri_X > min(cross_section_x)), np.logical_and(tri_Y < max(cross_section_y), tri_Y > min(cross_section_y))), axis=1)
    tri_cross_log_1 = np.logical_or(tri_cross_log_1_1, tri_cross_log_1_2)

    # and add a buffer of one attached triangle
    tri_cross_log_1 = np.any(np.isin(trinodes, np.unique(trinodes[tri_cross_log_1, :])), axis=1)

    # and reduce further by requiring every node to be within 1 line length + 10%
    line_len = np.sqrt((cross_section_x[0] - cross_section_x[1])**2 + (cross_section_y[0] - cross_section_y[1])**2)
    line_len_plus = line_len * 1.1

    tri_dist_1 = np.sqrt((tri_X - cross_section_x[0])**2 + (tri_Y - cross_section_y[0])**2)
    tri_dist_2 = np.sqrt((tri_X - cross_section_x[1])**2 + (tri_Y - cross_section_y[1])**2)

    tri_cross_log_2 = np.logical_and(tri_dist_1.min(1) < line_len_plus, tri_dist_2.min(1) < line_len_plus)
    tri_cross_log = np.logical_and(tri_cross_log_1, tri_cross_log_2)

    # but as a fall back for short lines add back in triangles within a threshold of 100m
    tri_cross_log_3 = np.logical_or(tri_dist_1.min(1) < 100, tri_dist_2.min(1) < 100)
    tri_cross_log = np.logical_or(tri_cross_log, tri_cross_log_3)

    # and add a buffer of one attached triangle
    tri_cross_log_1 = np.any(np.isin(trinodes, np.unique(trinodes[tri_cross_log, :])), axis=1)
    tri_cross_log = np.logical_or(tri_cross_log, tri_cross_log_1)


    # then subsample the line at a given resolution and find which triangle each sample falls in (if at all)
    sub_samp = np.asarray([np.linspace(cross_section_x[0], cross_section_x[1], res), np.linspace(cross_section_y[0], cross_section_y[1], res)]).T
    red_tri_list_ind = np.arange(0, len(trinodes))[tri_cross_log]
    sample_cells = np.zeros(len(sub_samp))

    for this_ind, this_point in enumerate(sub_samp):
        in_this_tri = False
        this_tri_ind = 0
        while in_this_tri is False:
            this_tri = red_tri_list_ind[this_tri_ind]
            is_in = isintriangle(tri_X[this_tri, :], tri_Y[this_tri, :], this_point[0], this_point[1])

            if is_in:
                sample_cells[this_ind] = this_tri
                in_this_tri = True
            elif this_tri_ind == len(red_tri_list_ind)-1:
                sample_cells[this_ind] = -1
                in_this_tri = True
            else:
                this_tri_ind += 1

    # for node properties now need the weight the nearest nodes to the sample point
    sample_nodes = np.zeros(len(sub_samp))
    red_node_ind = np.unique(trinodes[red_tri_list_ind, :])

    for this_ind, this_point in enumerate(sub_samp):
        if sample_cells[this_ind] == -1:
            sample_nodes[this_ind] = -1
        else:
            all_dist = np.sqrt((X[red_node_ind] - this_point[0])**2 + (Y[red_node_ind] - this_point[1])**2)
            sample_nodes[this_ind] = red_node_ind[np.where(all_dist == all_dist.min())[0][0]]

    return sub_samp, sample_cells, sample_nodes


def isintriangle(tri_x, tri_y, point_x, point_y):
    """
    Returns a boolean as to whether the point (point_x, point_y) is within the triangle (tri_x, tri_y)

    Parameters
    ----------
    tri_x, tri_y : np.ndarray
        Coordinates of the triangle vertices.
    point_x, point_y : float
        Position to test.

    Returns
    -------
    isin : bool
        True if the position (point_x, point_y) is in the triangle defined by (tri_x, tri_y).

    Notes
    -----
    Method from http://totologic.blogspot.co.uk/2014/01/accurate-point-in-triangle-test.html without edge test used
    in getcrossectiontriangles.

    """

    x1 = tri_x[0]
    x2 = tri_x[1]
    x3 = tri_x[2]

    y1 = tri_y[0]
    y2 = tri_y[1]
    y3 = tri_y[2]

    a = ((y2 - y3) * (point_x - x3) + (x3 - x2) * (point_y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    b = ((y3 - y1) * (point_x - x3) + (x1 - x3) * (point_y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    c = 1 - a - b

    is_in = 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1

    return is_in


def subset_domain(x, y, triangles, polygon=None):
    """
    Subset the current model grid, either interactively or with a given polygon. Coordinates in `polygon' must be in
    the same system as `x' and `y'.

    Parameters
    ----------
    x, y : np.ndarray
        The x and y coordinates of the whole model grid.
    triangles : np.ndarray
        The grid triangulation (shape = [elements, 3])
    polygon : shapely.geometry.Polygon, optional
        If given, the domain will be clipped by this polygon. If omitted, the polygon is defined interactively.

    Returns
    -------
    nodes : np.ndarray
        List of the node IDs from the original `triangles' array within the given polygon.
    elements : np.ndarray
        List of the element IDs from the original `triangles' array within the given polygon.
    triangles : np.ndarray
        The reduced triangulation of the new subdomain.

    """

    if polygon is not None:
        bounding_poly = np.asarray(polygon.exterior.coords)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, c='lightgray')
        plt.show()

        keep_picking = True
        while keep_picking:
            n_pts = int(input('How many polygon points? '))
            bounding_poly = np.full((n_pts, 2), np.nan)
            poly_lin = []
            for point in range(n_pts):
                bounding_poly[point, :] = plt.ginput(1)[0]
                poly_lin.append(ax.plot(np.hstack([bounding_poly[:, 0], bounding_poly[0, 0]]),
                                        np.hstack([bounding_poly[:, 1], bounding_poly[0, 1]]),
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
    nodes = np.squeeze(np.argwhere(np.asarray(poly_path.contains_points(np.asarray([x, y]).T))))
    elements = np.squeeze(np.argwhere(np.all(np.isin(triangles, nodes), axis=1)))

    sub_triangles = reduce_triangulation(triangles, nodes)

    return nodes, elements, sub_triangles


def model_exterior(lon, lat, triangles):
    """
    For a given unstructured grid, return a shapely.geometry.Polygon of the exterior boundary.

    Parameters
    ----------
    lon, lat : np.ndarray
        The x and y coordinates for the domain. Can be spherical or cartesian.
    triangles : np.ndarray
        The grid triangulation table (n, 3).

    Returns
    -------
    boundary : shapely.geometry.Polygon
        The grid exterior boundary (interior holes are ignores).

    """

    # `connectivity' doesn't return the boundary nodes in the right order, so we can't use it to get the exterior
    # boundary.
    boundary_nodes = get_boundary_polygons(triangles)
    polygons = [shapely.geometry.Polygon(np.asarray((lon[i], lat[i])).T) for i in boundary_nodes]
    areas = [i.area for i in polygons]
    boundary = polygons[areas.index(max(areas))]

    return boundary


def fvcom2ugrid(fvcom):
    """
    Add the necessary information to convert an FVCOM output file to one which is compatible with the UGRID format.

    Parameters
    ----------
    fvcom : str
        Path to an FVCOM netCDF file (can be a remote URL).

    """

    with Dataset(fvcom, 'a') as ds:
        fvcom_mesh = ds.createVariable('fvcom_mesh', np.int32)
        setattr(fvcom_mesh, 'cf_role', 'mesh_topology')
        setattr(fvcom_mesh, 'topology_dimension', 2)
        setattr(fvcom_mesh, 'node_coordinates', 'lon lat')
        setattr(fvcom_mesh, 'face_coordinates', 'lonc latc')
        setattr(fvcom_mesh, 'face_node_connectivity', 'nv')

        # Add the global convention.
        setattr(ds, 'Convention', 'UGRID-1.0')
        setattr(ds, 'CoordinateProjection', 'none')


def point_in_pixel(x, y, point):
    """
    Return the corner coordinate indices (x_min, x_max) and (y_min, y_max) for the pixel from (`x', `y') in which
    the given `point' lies.

    Parameters
    ----------
    x, y : np.ndarray
        The coordinates of the pixels (as vectors).
    point : tuple, list
        The target coordinate.

    Returns
    -------
    x_indices, y_indices : list
        The indices of `x' and `y' for the position in `point'.

    Notes
    -----

    No special attention is paid to points which lie exactly on a boundary. In that situation, the returned pixel
    will fall either to the left of or above the point.

    """

    x_diff = x - point[0]
    y_diff = y - point[1]

    closest_x = np.argmin(np.abs(x_diff))
    closest_y = np.argmin(np.abs(y_diff))

    if x_diff[closest_x] >= 0:
        # Containing pixel is right of point
        x_bound = closest_x - 1
    else:
        # Containing pixel is left of point
        x_bound = closest_x + 1

    if y_diff[closest_y] >= 0:
        # Containing pixel is above point
        y_bound = closest_y - 1
    else:
        # Containing pixel is below point
        y_bound = closest_y + 1

    x_indices = sorted((closest_x, x_bound))
    y_indices = sorted((closest_y, y_bound))

    return x_indices, y_indices


def node_to_centre(field, filereader):
    """
    TODO: docstring

    """
    tt = Triangulation(filereader.grid.x, filereader.grid.y, filereader.grid.triangles)
    interped_out = []
    if len(field.shape) == 1:
        field = field[np.newaxis, :]

    for this_t in field:
        ct = CubicTriInterpolator(tt, this_t)
        interped_out.append(ct(filereader.grid.xc, filereader.grid.yc))

    return np.asarray(interped_out)


class Graph(object):
    """
    Base class for graph theoretic functions.

    """

    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance

    def dijkstra(self, initial):
        visited = {initial: 0}
        path = {}

        nodes = set(self.nodes)

        while nodes:
            min_node = None
            for node in nodes:
                if node in visited:
                    if min_node is None:
                        min_node = node
                    elif visited[node] < visited[min_node]:
                        min_node = node
            if min_node is None:
                break

            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in self.edges[min_node]:
                try:
                    weight = current_weight + self.distances[(min_node, edge)]
                except:
                    continue
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node

        return visited, path

    def shortest_path(self, origin, destination):
        visited, paths = self.dijkstra(origin)
        full_path = deque()
        _destination = paths[destination]

        while _destination != origin:
            full_path.appendleft(_destination)
            _destination = paths[_destination]

        full_path.appendleft(origin)
        full_path.append(destination)

        return visited[destination], list(full_path)


class ReducedFVCOMdist(Graph):
    """
    Supporter class for refining paths

    The graph is defined using the triangulation from FVCOM but with only a subset of nodes.

    Note - assumes nodes are numbered 0 - len(triangle) and thus correspond to numbers in triangle

    """

    def __init__(self, nodes_sel, triangle, edge_weights):
        super(ReducedFVCOMdist, self).__init__()

        for this_node in nodes_sel:
            self.add_node(this_node)

        self.node_index = nodes_sel

        tri_inds = [[0, 1], [1, 2], [2, 0]]

        for this_tri, this_sides in zip(triangle, edge_weights):
            for these_inds in tri_inds:
                if np.all(np.isin(this_tri[these_inds], nodes_sel)):
                    self.add_edge(this_tri[these_inds[0]], this_tri[these_inds[1]], this_sides[these_inds[0]])
                    self.add_edge(this_tri[these_inds[1]], this_tri[these_inds[0]], this_sides[these_inds[0]])


class GraphFVCOMdepth(Graph):
    """
    A class for setting up a graph of an FVCOM grid, weighting the edges by the depth. This allows automatic
    identification of channel nodes between two points.

    Example use:

    import PyFVCOM as pf
    import matplotlib.pyplot as plt

    bounding_box = [[408975.302, 421555.45], [5576821.85, 5598995.61]]
    test_graph = pf.grid.GraphFVCOMdepth.('tamar_v2_grd.dat', depth_weight=200, depth_power=8, bounding_box=bounding_box)
    channel_nodes = test_graph.get_channel_between_points(([414409.23, 5596831.30], [415268.13, 5580529.97])

    channel_nodes_bool = np.isin(test_graph.node_index, channel_nodes)
    plt.scatter(test_graph.X, test_graph.Y, c='lightgray')
    plt.scatter(test_graph.X[channel_nodes_bool], test_graph.Y[channel_nodes_bool], c='red')

    Play with the weighting factors if the channel isn't looking right. depth_weight should adjust for the difference
    between the horizontal and vertical length scales. depth_power allows the difference in depths to be exaggerated
    so for channels with shallower cross-channel gradients it needs to be higher

    """

    def __init__(self, fvcom_grid_file, depth_weight=200, depth_power=8, bounding_box=None):
        """

        Parameters
        ----------
        fvcom_grid_file : str
            Location of the .grd file as read by the read_fvcom_mesh function.
        depth_weight : float
            Weighting by which the depths are multiplied.
        depth_power : int
            Power by which the depths are raised, this helps exaggerate differences between depths.
        bounding_box : list 2x2, optional
            To reduce computation times can subset the grid, the bounding box expects [[x_min, x_max], [y_min, y_max]]
            format.

        """

        super(GraphFVCOMdepth, self).__init__()
        triangle, nodes, x, y, z = read_fvcom_mesh(fvcom_grid_file)
        # Only offset the nodes by 1 if we've got 1-based indexing.
        if np.min(nodes) == 1:
            nodes -= 1
        elem_sides = element_side_lengths(triangle, x, y)

        if bounding_box is not None:
            x_lim = bounding_box[0]
            y_lim = bounding_box[1]
            nodes_in_box = np.logical_and(np.logical_and(x > x_lim[0], x < x_lim[1]),
                                          np.logical_and(y > y_lim[0], y < y_lim[1]))

            nodes = nodes[nodes_in_box]
            x = x[nodes_in_box]
            y = y[nodes_in_box]
            z = z[nodes_in_box]

            elem_sides = elem_sides[np.all(np.isin(triangle, nodes), axis=1), :]
            triangle = reduce_triangulation(triangle, nodes)

        z = np.mean(z[triangle], axis=1)  # adjust to cell depths rather than nodes, could use FVCOM output depths instead
        z = z - np.min(z)  # make it so depths are all >= 0
        z = np.max(z) - z  # and flip so deeper areas have lower numbers

        depth_weighted = (z * depth_weight)**depth_power
        edge_weights = elem_sides * np.tile(depth_weighted, [3, 1]).T

        self.elem_sides = elem_sides
        self.x = x
        self.y = y
        self.z = z
        self.node_index = nodes
        self.triangle = triangle
        self.edge_weights = edge_weights

        for this_node in nodes:
            self.add_node(this_node)

        tri_inds = [[0, 1], [1, 2], [2, 0]]

        for this_tri, this_sides in zip(self.triangle, edge_weights):
            for these_inds in tri_inds:
                self.add_edge(self.node_index[this_tri[these_inds[0]]], self.node_index[this_tri[these_inds[1]]], this_sides[these_inds[0]])
                self.add_edge(self.node_index[this_tri[these_inds[1]]], self.node_index[this_tri[these_inds[0]]], this_sides[these_inds[0]])

    def get_nearest_node_ind(self, near_xy):
        """
        Find the nearest graph node to a given xy point

        Parameters
        ----------
        near_xy : list-like
            x and y coordinates of the point

        Returns
        -------
        node_number : int
            the node in the grid closes to xy

        """

        dists = (self.x - near_xy[0])**2 + (self.y - near_xy[1])**2
        return self.node_index[dists.argmin()]

    def get_channel_between_points(self, start_xy, end_xy, refine_channel=False):
        """
        Find the shortest path between two points according to depth weighted distance (hopefully the channel...)

        Parameters
        ----------
        near_xy : list-like
            x and y coordinates of the start point

        end_xy : list-like
            x and y coordinates of the end point

        refine_channel : bool, optional
            Apply a refinement step, this might help in cases if extreme depth scalings cause the path to take two
            sides of a triangle when it should take just one

        Returns
        -------
        node_list : np.ndarray
            list of fvcom node numbers forming the predicted channel between the start and end points

        """
        start_node_ind = self.get_nearest_node_ind(start_xy)
        end_node_ind = self.get_nearest_node_ind(end_xy)

        _, node_list = self.shortest_path(start_node_ind, end_node_ind)

        if refine_channel:
            node_list = self._refine_channel(np.asarray(node_list))

        return np.asarray(node_list) + 1  # to get fvcom nodeReducedFVCOMdist number not python indices

    def _refine_channel(self, node_list):
        """
        Refines the channel nodes by re running the shortest path algorthim without depth weighting but only on the
        chosen nodes

        """
        nodes_sel = np.where(np.isin(self.node_index, node_list))[0]
        start_node = np.where(self.node_index == node_list[0])[0][0]
        end_node = np.where(self.node_index == node_list[-1])[0][0]

        red_graph = ReducedFVCOMdist(nodes_sel, self.triangle, self.elem_sides)
        _, red_node_list = red_graph.shortest_path(start_node, end_node)

        return np.asarray(self.node_index[np.asarray(red_node_list)])

