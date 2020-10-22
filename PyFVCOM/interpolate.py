from PyFVCOM.read import FileReader
from PyFVCOM.grid import unstructured_grid_depths, node_to_centre, get_boundary_polygons, reduce_triangulation
from PyFVCOM.utilities.general import PassiveStore, warn
import shapely.geometry as sg
import scipy.interpolate as si
import numpy as np


def mask_to_fvcom(fvcom_ll, fvcom_tri, lons, lats, split_domain_check=False):
    """
    Constructs a mask for a list of points which is true for points lying outside the specified fvcom domain and false for those
    within. The split_domain_check is more intensive to calculate but can handle cases where the FVCOM domain is non-contiguous or node/element 0
    is not in the exterior boundary


    Parameters
    ----------
    fvcom_ll : array
        Px2 array of the lon lat positions of the FVCOM grid nodes
    fvcom_tri : array
        Qx3 python indexed triangulation array for the FVCOM grid
    lons : array
        1d array of the longitudes to mask
    grid_mesh_lats : array
        1d array of latitudes to mask
    split_domain_check : boolean, optional
        The default version of the algorithm makes two assumptions, 1) That the first polygon returned by pf.grid.get_boundary_polygons
        is the exterior of the FVCOM domain (certainly true if the OBC nodes start at 0) and 2) The FVCOM domain is one contiguous piece
        of water. These are normally the case but sometimes, e.g, when slicing by a depth level which makes the domain non-contiguos then
        this should be set to True to do a more thorough job of checking each point

    Returns
    -------
    out_of_domain_mask : array
        A 1d boolean array true for points outside the FVCOM domain and false for those within


    """
    out_of_domain_mask = np.zeros(lons.size, dtype=bool)
    grid_point_list = [sg.Point(this_ll) for this_ll in np.asarray([lons, lats]).T]

    if split_domain_check:
        boundary_polygon_list, islands_list = get_boundary_polygons(fvcom_tri, nodes=fvcom_ll)

        loc_code = np.zeros(len(grid_point_list))

        for is_island, this_poly_pts in zip(islands_list, boundary_polygon_list):
            this_poly = sg.Polygon(fvcom_ll[this_poly_pts, :])
            for i, this_pt in enumerate(grid_point_list):
                if is_island and this_poly.contains(this_pt):
                    loc_code[i] = 1
                elif not is_island and this_poly.contains(this_pt) and loc_code[i] != 1:
                    loc_code[i] = 2

        out_of_domain_mask = np.logical_or(loc_code == 0, loc_code == 1)


    else:
        boundary_polygon_list = get_boundary_polygons(fvcom_tri)
        boundary_polys = [sg.Polygon(fvcom_ll[this_poly_pts, :]) for this_poly_pts in boundary_polygon_list]

        out_of_domain_mask = np.ones(lons.size, dtype=bool)

        for i, this_pt in enumerate(grid_point_list):
            if boundary_polys[0].contains(this_pt):
                out_of_domain_mask[i] = False
                for this_poly in boundary_polys[1:]:
                    if this_poly.contains(this_pt):
                        out_of_domain_mask[i] = True

    return out_of_domain_mask


def mask_to_fvcom_meshgrid(fvcom_ll, fvcom_tri, grid_mesh_lons, grid_mesh_lats, split_domain_check=False):
    """
    Mask a regularly gridded set of coordinates to an FVCOM grid.

    Parameters
    ----------
    fvcom_ll : array
        Px2 array of the lon lat positions of the FVCOM grid nodes
    fvcom_tri : array
        Qx3 python indexed triangulation array for the FVCOM grid
    grid_mesh_lons : array
        MxN array of the longitudes of the regular grid (e.g. output from meshgrid)
    grid_mesh_lats : array
        MxN array of latitudes of regular grid
    split_domain_check : boolean, optional
        The default version of the algorithm makes two assumptions, 1) That the first polygon returned by pf.grid.get_boundary_polygons
        is the exterior of the FVCOM domain (certainly true if the OBC nodes start at 0) and 2) The FVCOM domain is one contiguous piece
        of water. These are normally the case but sometimes, e.g, when slicing by a depth level which makes the domain non-contiguos then
        this should be set to True to do a more thorough job of checking each point

    Returns
    -------
    grid_land_mask : array
        An NxM boolean array true for points outside the FVCOM domain and false for those within

    """

    lons = grid_mesh_lons.ravel()
    lats = grid_mesh_lats.ravel()
    points_mask = mask_to_fvcom(fvcom_ll, fvcom_tri, lons, lats, split_domain_check=False)

    grid_land_mask = np.reshape(points_mask, grid_mesh_lons.shape)

    return grid_land_mask


class MPIRegularInterpolateWorker():
    """
    Worker class for interpolating fvcom to regular grid, some point maybe there should be a more generic parent class

    """

    def __init__(self, fvcom_file, time_indices, comm=None, root=0, verbose=False):
        """
        Create interpolation worker

        Parameters
        ----------
        fvcom_file : str
            Path to the fvcom model output to be interpolated
        time_indices : list-like
            The indexes of the times within the FVCOM file to be interpolated
        comm : mpi4py.MPI.Intracomm, optional
            The MPI intracommunicator object. Omit if not running in parallel.
        root : int, optional
            Specify a given rank to act as the root process. This is only for outputting verbose messages (if enabled
            with `verbose').
        verbose : bool, optional
            Set to True to enabled some verbose output messages. Defaults to False (no messages).

        """
        self.dims = None
        self.fvcom_file = fvcom_file
        self.time_indices = time_indices

        self.have_mpi = True
        try:
            from mpi4py import MPI
            self.MPI = MPI
        except ImportError:
            warn('No mpi4py found in this python installation. Some functions will be disabled.')
            self.have_mpi = False

        self.comm = comm
        if self.have_mpi:
            self.rank = self.comm.Get_rank()
        else:
            self.rank = 0

        self.root = root
        self._noisy = verbose
        if verbose and comm is None:
            print('For verbose output you need to pass a comm object so it knows the process rank. Gonna crash if not.')

        self.fvcom_grid = PassiveStore()
        self.regular_grid = PassiveStore()

    def InitialiseGrid(self, lower_left_ll, upper_right_ll, grid_res, depth_layers, time_varying_depth=True):
        """
        Sets up the grids to interpolate to and from. Part of the work is reducing the domain, then comes the trick step of interpolating onto vertical layers
        which vary in time and may create discontiguous areas of the model domain.

        Parameters
        ----------
        lower_left_ll : 1d array
            A two entry array of the lon/lat of the lower left corner of the regular grid to be interpolated onto
        upper_right_ll : 1d array
            The corresponding lon/lat of the upper right corner
        grid_res : float
            The resolution in degrees of the regular grid
        depth layers : array
            Array of the depths to interpolate onto. These are positive down starting from the top of the free surface or the zero mean
            depending on the setting of time_varying_depth


        """
        if self._noisy:
            print('Rank {}: Loading grid'.format(self.rank))
        # Check ther regular grid limits fit with the grid resolution specified
        if not round((upper_right_ll[0] - lower_left_ll[0]) % grid_res, 10) % grid_res == 0:
            print('Longitudes not divisible by grid resolution, extending grid')
            upper_right_ll[0] = round(np.ceil((upper_right_ll[0] - lower_left_ll[0])/grid_res)*grid_res, 10) + lower_left_ll[0]

        if not round((upper_right_ll[1] - lower_left_ll[1]) % grid_res, 10) % grid_res == 0:
            print('Latitudes not divisible by grid resolution, extending grid')
            upper_right_ll[1] = round(np.ceil((upper_right_ll[1] - lower_left_ll[1])/grid_res)*grid_res, 10) + lower_left_ll[1]

        self.regular_grid.lons = np.linspace(lower_left_ll[0], upper_right_ll[0], (upper_right_ll[0]-lower_left_ll[0])/grid_res + 1)
        self.regular_grid.lats = np.linspace(lower_left_ll[1], upper_right_ll[1], (upper_right_ll[1]-lower_left_ll[1])/grid_res + 1)

        self.regular_grid.mesh_lons, self.regular_grid.mesh_lats = np.meshgrid(self.regular_grid.lons, self.regular_grid.lats)
        self.regular_grid.dep_lays = np.asarray(depth_layers)

        # And load the fvcom grid, reducing to the interpolation area only
        fvcom_grid_fr = FileReader(self.fvcom_file, ['zeta'], dims={'time': self.time_indices})

        fvcom_nodes_reduce = np.logical_and(np.logical_and(fvcom_grid_fr.grid.lon >= lower_left_ll[0], fvcom_grid_fr.grid.lon <= upper_right_ll[0]),
                                            np.logical_and(fvcom_grid_fr.grid.lat >= lower_left_ll[1], fvcom_grid_fr.grid.lat <= upper_right_ll[1]))
        fvcom_nodes_reduce = np.squeeze(np.argwhere(fvcom_nodes_reduce))

        ## extend the node list to include attached nodes outside the area, this should make the interpolation better
        fvcom_nodes = np.isin(fvcom_grid_fr.grid.triangles, fvcom_nodes_reduce)
        self.fvcom_grid.nodes = np.unique(fvcom_grid_fr.grid.triangles[np.asarray(np.sum(fvcom_nodes, axis=1), dtype=bool), :])

        self.fvcom_grid.triangles, self.fvcom_grid.elements = reduce_triangulation(fvcom_grid_fr.grid.triangles, self.fvcom_grid.nodes, return_elements=True)
        self.fvcom_grid.ll = np.asarray([fvcom_grid_fr.grid.lon[self.fvcom_grid.nodes], fvcom_grid_fr.grid.lat[self.fvcom_grid.nodes]]).T
        self.fvcom_grid.elements_ll = np.asarray([fvcom_grid_fr.grid.lonc[self.fvcom_grid.elements], fvcom_grid_fr.grid.latc[self.fvcom_grid.elements]]).T

        self.regular_grid.initial_mask = mask_to_fvcom_meshgrid(self.fvcom_grid.ll, self.fvcom_grid.triangles, self.regular_grid.mesh_lons, self.regular_grid.mesh_lats)

        self.fvcom_grid.h = fvcom_grid_fr.grid.h[self.fvcom_grid.nodes]
        self.fvcom_grid.zeta = fvcom_grid_fr.data.zeta[:, self.fvcom_grid.nodes]
        self.fvcom_grid.zeta_center = node_to_centre(fvcom_grid_fr.data.zeta, fvcom_grid_fr)[:, self.fvcom_grid.elements]
        self.fvcom_grid.sigma = fvcom_grid_fr.grid.siglay[:, self.fvcom_grid.nodes]
        self.fvcom_grid.h_center = fvcom_grid_fr.grid.h_center[self.fvcom_grid.elements]
        self.fvcom_grid.sigma_center = fvcom_grid_fr.grid.siglay_center[:, self.fvcom_grid.elements]

        if time_varying_depth:
            if self._noisy:
                print('Rank {}: Calculating time varying mask'.format(self.rank))
            fvcom_dep_lays = -unstructured_grid_depths(self.fvcom_grid.h, self.fvcom_grid.zeta, self.fvcom_grid.sigma)
            ## change to depth from free surface since I *think* this is what cmems does?
            self.fvcom_grid.dep_lays = fvcom_dep_lays - np.tile(np.min(fvcom_dep_lays, axis=1)[:, np.newaxis, :], [1, fvcom_dep_lays.shape[1], 1])
            self.fvcom_grid.total_depth = np.max(self.fvcom_grid.dep_lays, axis=1)

            self.regular_grid.mask = np.ones([len(self.time_indices), len(self.regular_grid.lons), len(self.regular_grid.lats), len(self.regular_grid.dep_lays)], dtype=bool)
            for this_t in np.arange(0, len(self.time_indices)):
                # retriangulate for each depth layer, can be multiple if there are split regions and interpolate
                for this_depth_lay_ind, this_depth_lay in enumerate(self.regular_grid.dep_lays):
                    if self._noisy:
                        print('Rank {}: Time step {} Depth {}'.format(self.rank, this_t, this_depth_lay_ind))
                    this_depth_layer_nodes = np.where(self.fvcom_grid.total_depth[this_t,:] >=this_depth_lay)[0]
                    if this_depth_layer_nodes.size:
                        this_depth_tri = reduce_triangulation(self.fvcom_grid.triangles, this_depth_layer_nodes)

                        this_td_mask = np.ones(self.regular_grid.mesh_lons.shape, dtype=bool)
                        this_td_mask[~self.regular_grid.initial_mask] = mask_to_fvcom(self.fvcom_grid.ll[this_depth_layer_nodes],
                                                                                      this_depth_tri,
                                                                                      self.regular_grid.mesh_lons[~self.regular_grid.initial_mask],
                                                                                      self.regular_grid.mesh_lats[~self.regular_grid.initial_mask],
                                                                                      split_domain_check=True).T
                        self.regular_grid.mask[this_t, :, :, this_depth_lay_ind] = this_td_mask.T

        else:
            print('Not implemented yet')

    def InterpolateRegular(self, variable, mode='nodes'):
        """
        Actually do the interpolation
        """
        if mode in ['nodes', 'surface']:
            self.fvcom_grid.select_points = self.fvcom_grid.nodes
            self.fvcom_grid.select_ll = self.fvcom_grid.ll
        else:
            self.fvcom_grid.select_points = self.fvcom_grid.elements
            self.fvcom_grid.select_ll = self.fvcom_grid.elements_ll

        if mode in ['nodes', 'surface']:
            self.fvcom_grid.select_dep_lays = self.fvcom_grid.dep_lays
        else:
            fvcom_dep_lays = -unstructured_grid_depths(self.fvcom_grid.h_center, self.fvcom_grid.zeta_center, self.fvcom_grid.sigma_center)
            ## change to depth from free surface since I *think* this is what cmems does?
            self.fvcom_grid.select_dep_lays = fvcom_dep_lays - np.tile(np.min(fvcom_dep_lays, axis=1)[:, np.newaxis, :], [1, fvcom_dep_lays.shape[1], 1])

        if mode == 'surface':
            reg_grid_data = np.zeros([len(self.time_indices), len(self.regular_grid.lons), len(self.regular_grid.lats)])
        else:
            reg_grid_data = np.zeros([len(self.time_indices), len(self.regular_grid.lons), len(self.regular_grid.lats), len(self.regular_grid.dep_lays)])

        reg_grid_data[:] = np.nan

        fvcom_data = getattr(FileReader(self.fvcom_file, [variable], dims={'time': self.time_indices}).data, variable)[..., self.fvcom_grid.select_points]

        for this_t in np.arange(0, len(self.time_indices)):
            if self._noisy:
                print('Rank {}: Interpolating time step {} for {}'.format(self.rank, this_t, variable))
            if mode == 'surface':
                depth_lay_data = fvcom_data[this_t, :]
                interp_data = self._Interpolater(depth_lay_data).T
                interp_data[self.regular_grid.mask[this_t, :, :, 0]] = np.nan
                reg_grid_data[this_t, :, :] = interp_data

            else:
                depth_lay_data = np.zeros([len(self.regular_grid.dep_lays), len(self.fvcom_grid.select_points)])
                for i in np.arange(0, len(self.fvcom_grid.select_points)):
                    depth_lay_data[:, i] = np.interp(self.regular_grid.dep_lays, self.fvcom_grid.select_dep_lays[this_t, :, i],
                                                     fvcom_data[this_t, :, i], left=np.nan, right=np.nan)
                # Replace surface data as it can't interpolate properly there
                if self.regular_grid.dep_lays[0] == 0:
                    depth_lay_data[0, :] = fvcom_data[this_t, 0, :]

                for this_dep_lay_ind, this_dep_lay in enumerate(self.regular_grid.dep_lays):
                    interp_data = self._Interpolater(depth_lay_data[this_dep_lay_ind, :]).T

                    interp_data[self.regular_grid.mask[this_t, :, :, this_dep_lay_ind]] = np.nan
                    reg_grid_data[this_t, :, :, this_dep_lay_ind] = interp_data

        return reg_grid_data

    def _Interpolater(self, data):
        non_nan = ~np.isnan(data)
        if np.sum(non_nan) > 0:
            interped_data = si.griddata(self.fvcom_grid.select_ll[non_nan, :], data[non_nan],
                                        (self.regular_grid.mesh_lons, self.regular_grid.mesh_lats), method='cubic')
        else:
            interped_data = np.zeros(self.regular_grid.mesh_lons.shape)
            interped_data[:] = np.nan
        return interped_data

class MPIUnstructuredInterpolateWorker():
    """
    For interpolating unstructured data to the FVCOM grid. Currently only for single layer (e.g. surface) applications.
    """

    def __init__(self, fvcom_file, data_coords_file, data_file, root=0, comm=None, verbose=False, cartesian=False):
        self.fvcom_file = FileReader(fvcom_file)

        self.have_mpi = True
        try:
            from mpi4py import MPI
            self.MPI = MPI
        except ImportError:
            warn('No mpi4py found in this python installation. Some functions will be disabled.')
            self.have_mpi = False

        self.comm = comm
        if self.have_mpi:
            self.rank = self.comm.Get_rank()
        else:
            self.rank = 0

        self.root = root
        self._noisy = verbose
        if verbose and comm is None:
            print('For verbose output you need to pass a comm object so it knows the process rank. Gonna crash if not.')

        self.data_coords = np.load(data_coords_file)       
        self.data = np.load(data_file)

        if cartesian:
            self.model_coords = np.asarray([self.fvcom_file.grid.x, self.fvcom_file.grid.y]).T
        else:
            self.model_coords = np.asarray([self.fvcom_file.grid.lon, self.fvcom_file.grid.lat]).T

    def InterpolateFVCOM(self, data_indices):
        all_interped_data = [] 
        for this_index in data_indices:
            all_interped_data.append(self._Interpolater(self.data[this_index,:]))

        return np.asarray(all_interped_data)

    def _Interpolater(self, data):
        non_nan = ~np.isnan(data)
        if np.sum(non_nan) > 0:
            interpolater = si.Rbf(self.data_coords[non_nan,0], self.data_coords[non_nan,1],
                            data[non_nan], function='cubic', smooth=0)
            interped_data = interpolater(self.model_coords[:,0], self.model_coords[:,1]) 
        else:
            interped_data = np.zeros(self.model_coords.shape)
            interped_data[:] = np.nan
        return interped_data

def interpolate_data_regular(fvcom_obj, fvcom_name, coarse_name, coarse, interval=1,
                        constrain_coordinates=False,
                        mode='nodes', tide_adjust=False, verbose=False):
    """
    Interpolate the given data onto the open boundary nodes for the period 
    from 'fvcom_obj.time.start' to 'fvcom_obj.time.end'.

    Parameters
    ----------
    fvcom_obj : filereader like object
        The fvcom reader like object onto which to interpolate the regular data
        usually a restart or nest object
    fvcom_name : str
        The data field name to add to the nest object which will be 
        written to netCDF for FVCOM.
    coarse_name : str
        The data field name to use from the coarse object.
    coarse : RegularReader
        The regularly gridded data to interpolate onto the open boundary 
        nodes. This must include time, lon, lat and depth data as well as 
        the time series to interpolate (4D volume [time, depth, lat, lon]).
    interval : float, optional
        Time sampling interval in days. Defaults to 1 day.
    constrain_coordinates : bool, optional
        Set to True to constrain the open boundary coordinates (lon, lat, 
        depth) to the supplied coarse data. This essentially squashes the 
        open boundary to fit inside the coarse data and is, therefore, a 
        bit of a fudge! Defaults to False.
    mode : bool, optional
        Set to 'nodes' to interpolate onto the open boundary node 
        positions or 'elements' for the elements. 'nodes and 'elements' 
        are for input data on z-levels. For 2D data, set to 'surface' 
        (interpolates to the node positions ignoring depth coordinates). 
        Also supported are 'sigma_nodes' and 'sigma_elements' which means 
        we have spatially (and optionally temporally) varying water depths 
        (i.e. sigma layers rather than z-levels). Defaults to 'nodes'.
    tide_adjust : bool, optional
        Some nested forcing doesn't include tidal components and these 
        have to be added from predictions using harmonics. With this set 
        to true the interpolated forcing has the tidal component (required 
        to already exist in fvcom_obj.tide) added to the final data.
    verbose : bool, optional
        Set to True to enable verbose output. Defaults to False 
        (no verbose output).
    """

    # Check we have what we need.
    raise_error = False
    if mode == 'nodes':
        if not np.any(fvcom_obj.grid.nodes):
            if verbose:
                print(f'No {mode} on which to interpolate on this boundary')
            return
        if not hasattr(fvcom_obj.grid.sigma, 'layers'):
            raise_error = True
    elif mode == 'elements':
        if not hasattr(fvcom_obj.grid.sigma, 'layers_center'):
            raise_error = True
        if not np.any(fvcom_obj.grid.elements):
            if verbose:
                print(f'No {mode} on which to interpolate on this boundary')
            return

    if raise_error:
        raise AttributeError('Add vertical sigma coordinates in order to '
                + 'interpolate forcing along this boundary.')

    fvcom_obj.time.interval = interval

    if 'elements' in mode:
        x = fvcom_obj.grid.lonc
        y = fvcom_obj.grid.latc
        # Keep positive down depths.
        z = -fvcom_obj.sigma.layers_center_z
    else:
        x = fvcom_obj.grid.lon
        y = fvcom_obj.grid.lat
        # Keep positive down depths.
        z = -fvcom_obj.sigma.layers_z

    if constrain_coordinates:
        x[x < coarse.grid.lon.min()] = coarse.grid.lon.min()
        x[x > coarse.grid.lon.max()] = coarse.grid.lon.max()
        y[y < coarse.grid.lat.min()] = coarse.grid.lat.min()
        y[y > coarse.grid.lat.max()] = coarse.grid.lat.max()

        # Internal landmasses also need to be dealt with, so test if a 
        # point lies within the mask of the grid and
        # move it to the nearest in grid point if so.
        if not mode == 'surface':
            land_mask = getattr(coarse.data, coarse_name
                    ).mask[0,0,:,:]
        else:
            land_mask = getattr(coarse.data, coarse_name).mask[0,:,:]

        sea_points = np.ones(land_mask.shape)
        sea_points[land_mask] = np.nan

        ft_sea = si.RegularGridInterpolator((coarse.grid.lat, coarse.grid.lon),
                sea_points, method='linear', fill_value=np.nan)
        internal_points = np.isnan(ft_sea(np.asarray([y, x]).T))

        if np.any(internal_points):
            xv, yv = np.meshgrid(coarse.grid.lon, coarse.grid.lat)
            valid_ll = np.asarray([x[~internal_points],
                    y[~internal_points]]).T
            for this_ind in np.where(internal_points)[0]:
                nearest_valid_ind = np.argmin(
                        (valid_ll[:, 0] - x[this_ind])**2
                        + (valid_ll[:, 1] - y[this_ind])**2)
                x[this_ind] = valid_ll[nearest_valid_ind, 0]
                y[this_ind] = valid_ll[nearest_valid_ind, 1]

        # The depth data work differently as we need to squeeze each 
        # FVCOM water column into the available coarse
        # data. The only way to do this is to adjust each FVCOM water 
        # column in turn by comparing with the
        # closest coarse depth.
        if mode != 'surface':
            coarse_depths = np.tile(coarse.grid.depth, [coarse.dims.lat,
                    coarse.dims.lon, 1]).transpose(2, 0, 1)
            coarse_depths = np.ma.masked_array(coarse_depths,
                    mask=getattr(coarse.data, coarse_name)[0, ...].mask)
            coarse_depths = np.max(coarse_depths, axis=0)

            # Find any places where only the zero depth layer exists and 
            # copy down 
            zero_depth_water = np.where(np.logical_and(coarse_depths == 0,
                    ~coarse_depths.mask))
            if zero_depth_water[0].size:
                data_mod = getattr(coarse.data, coarse_name)
                data_mod[:, 1, zero_depth_water[0], zero_depth_water[1]] = (
                        data_mod[:, 0, zero_depth_water[0],
                        zero_depth_water[1]])
                (data_mod.mask[:, 1, zero_depth_water[0],
                        zero_depth_water[1]]) = False
                setattr(coarse.data, coarse_name, data_mod)
                        # Probably isn't needed cos pointers but for clarity

            coarse_depths = np.ma.filled(coarse_depths, 0)

            # Go through each open boundary position and if its depth is 
            # deeper than the closest coarse data,
            # squash the open boundary water column into the coarse water 
            # column.
            for idx, node in enumerate(zip(x, y, z)):
                nearest_lon_ind = np.argmin((coarse.grid.lon - node[0])**2)
                nearest_lat_ind = np.argmin((coarse.grid.lat - node[1])**2)

                if node[0] < coarse.grid.lon[nearest_lon_ind]:
                    nearest_lon_ind = [nearest_lon_ind -1, nearest_lon_ind,
                            nearest_lon_ind -1, nearest_lon_ind]
                else:
                    nearest_lon_ind = [nearest_lon_ind, nearest_lon_ind + 1,
                            nearest_lon_ind, nearest_lon_ind + 1]

                if node[1] < coarse.grid.lat[nearest_lat_ind]:
                    nearest_lat_ind = [nearest_lat_ind -1, nearest_lat_ind
                            -1, nearest_lat_ind, nearest_lat_ind]
                else:
                    nearest_lat_ind = [nearest_lat_ind, nearest_lat_ind,
                            nearest_lat_ind + 1, nearest_lat_ind + 1]

                grid_depth = np.min(coarse_depths[nearest_lat_ind,
                        nearest_lon_ind])

                if grid_depth < node[2].max():
                    # Squash the FVCOM water column into the coarse water 
                    # column.
                    z[idx, :] = (node[2] / node[2].max()) * grid_depth
            # Fix all depths which are shallower than the shallowest 
            # coarse depth. This is more straightforward as
            # it's a single minimum across all the open boundary positions.
            z[z < coarse.grid.depth.min()] = coarse.grid.depth.min()

    nt = len(fvcom_obj.time.time)
    nx = len(x)
    nz = z.shape[-1]

    if verbose:
        print('Interpolating {} times, {} '.format(nt, nz)
                + 'vertical layers and {} points'.format(nx))

    # Make arrays of lon, lat, depth and time for non-sigma interpolation. 
    # Need to make the coordinates match the
    # coarse data shape and then flatten the lot. We should be able to do 
    # the interpolation in one shot this way,
    # but we have to be careful our coarse data covers our model domain 
    # (space and time).
    if mode == 'surface':
        if verbose:
            print('Interpolating surface data...', end=' ')

        # We should use np.meshgrid here instead of all this tiling 
        # business.
        boundary_grid = np.array((
                np.tile(fvcom_obj.time.time, [nx, 1]).T.ravel(),
                np.tile(y, [nt, 1]).transpose(0, 1).ravel(),
                np.tile(x, [nt, 1]).transpose(0, 1).ravel())).T
        ft = si.RegularGridInterpolator((coarse.time.time,
                coarse.grid.lat, coarse.grid.lon),
                getattr(coarse.data, coarse_name),
                method='linear', fill_value=np.nan)
        # Reshape the results to match the un-ravelled boundary_grid array.
        interpolated_coarse_data = ft(boundary_grid).reshape([nt, -1])
    elif 'sigma' in mode:
        if verbose:
            print('Interpolating sigma data...', end=' ')

        nt = coarse.dims.time  # rename!
        interp_args = [(x, y, fvcom_obj.sigma.layers_z,
                coarse, coarse_name, fvcom_obj._debug, t) for t in np.arange(nt)]
        if hasattr(coarse, 'ds'):
            coarse.ds.close()
            delattr(coarse, 'ds')
        pool = multiprocessing.Pool()
        results = pool.map(fvcom_obj._brute_force_interpolator, interp_args)

        # Now we have those data interpolated in space (horizontal and 
        # vertical), interpolate to match in time.
        interp_args = [(coarse.time.time, j, fvcom_obj.time.time)
                for i in np.asarray(results).T for j in i]
        results = pool.map(fvcom_obj._interpolate_in_time, interp_args)
        pool.close()
        # Reshape and transpose to be the correct size for writing to 
        # netCDF (time, depth, node).
        interpolated_coarse_data = np.asarray(results).reshape(
                  nz, nx, -1).transpose(2, 0, 1)
    else:
        if verbose:
            print('Interpolating z-level data...', end=' ')
        # Assume it's z-level data (e.g. HYCOM, CMEMS). We should use 
        # np.meshgrid here instead of all this tiling
        # business.
        boundary_grid = np.array((
                np.tile(fvcom_obj.time.time, [nx, nz, 1]).T.ravel(),
                np.tile(z.T, [nt, 1, 1]).ravel(),
                np.tile(y, [nz, nt, 1]).transpose(1, 0, 2).ravel(),
                np.tile(x, [nz, nt, 1]).transpose(1, 0, 2).ravel())).T
        ft = si.RegularGridInterpolator((coarse.time.time, coarse.grid.depth,
                coarse.grid.lat, coarse.grid.lon),
                np.ma.filled(getattr(coarse.data, coarse_name), np.nan),
                method='linear', fill_value=np.nan)
        # Reshape the results to match the un-ravelled boundary_grid 
        # array (time, depth, node).
        interpolated_coarse_data = ft(boundary_grid).reshape([nt, nz, -1])

    if tide_adjust and fvcom_name in ['u', 'v', 'ua', 'va', 'zeta']:
        if fvcom_name in ['u', 'v']:
            tide_levels = np.tile(getattr(fvcom_obj.tide, fvcom_name)
                    [:, np.newaxis, :], [1, nz, 1])
            interpolated_coarse_data = (interpolated_coarse_data
                    + tide_levels)
        else:
            interpolated_coarse_data = interpolated_coarse_data + getattr(
                    fvcom_obj.tide, fvcom_name)

    return interpolated_coarse_data

def avg_nest_force_vel(fvcom_obj):
    """
    Create depth-averaged velocities (`ua', `va') in the current 
    fvcom_obj.data data.

    """
    layer_thickness = (fvcom_obj.sigma.levels_center.T[0:-1, :]
            - fvcom_obj.sigma.levels_center.T[1:, :])
    fvcom_obj.data.ua = zbar(fvcom_obj.data.u, layer_thickness)
    fvcom_obj.data.va = zbar(fvcom_obj.data.v, layer_thickness)

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
    x, y, fvcom_layer_depth, coarse, coarse_name, verbose, t = args

    num_fvcom_z = fvcom_layer_depth.shape[-1]
    num_fvcom_points = len(x)

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

        interpolator = si.LinearNDInterpolator((coarse_lon, coarse_lat), coarse_data_layer)
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
            interpolator = si.LinearNDInterpolator((x[good_indices], y[good_indices]), horizontal_slice[good_indices])
            interp_fvcom_boundary[:, p_index] = interpolator((x, y))

    return interp_fvcom_boundary

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
    ft = si.RegularGridInterpolator((depth, lat, lon), data, method='linear', fill_value=None)
    interpolated_data = ft(points)

    return interpolated_data

def interpolate_data_curvilinear(fvcom_obj, fvcom_name, coarse_name, coarse, interval=1,
                        constrain_coordinates=False,
                        mode='nodes', tide_adjust=False, cartesian=False, verbose=False):
    """
    Interpolate the given data onto the open boundary nodes for the period 
    from 'fvcom_obj.time.start' to 'fvcom_obj.time.end'.

    Parameters
    ----------
    fvcom_name : str
        The data field name to add to the nest object which will be 
        written to netCDF for FVCOM.
    coarse_name : str
        The data field name to use from the coarse object.
    coarse : RegularReader
        The regularly gridded data to interpolate onto the open boundary 
        nodes. This must include time, lon, lat and depth data as well as 
        the time series to interpolate (4D volume [time, depth, lat, lon]).
    interval : float, optional
        Time sampling interval in days. Defaults to 1 day.
    constrain_coordinates : bool, optional
        Set to True to constrain the open boundary coordinates (lon, lat, 
        depth) to the supplied coarse data. This essentially squashes the 
        open boundary to fit inside the coarse data and is, therefore, a 
        bit of a fudge! Defaults to False.
    mode : bool, optional
        Set to 'nodes' to interpolate onto the open boundary node 
        positions or 'elements' for the elements. 'nodes and 'elements' 
        are for input data on z-levels. For 2D data, set to 'surface' 
        (interpolates to the node positions ignoring depth coordinates). 
        Also supported are 'sigma_nodes' and 'sigma_elements' which means 
        we have spatially (and optionally temporally) varying water depths 
        (i.e. sigma layers rather than z-levels). Defaults to 'nodes'.
    tide_adjust : bool, optional
        Some nested forcing doesn't include tidal components and these 
        have to be added from predictions using harmonics. With this set 
        to true the interpolated forcing has the tidal component (required 
        to already exist in fvcom_obj.tide) added to the final data.
    cartesian : bool, optional
        Use utm coordinates rather than lon/lat. Defaults to False.
    verbose : bool, optional
        Set to True to enable verbose output. Defaults to False 
        (no verbose output).
    """
    # Check we have what we need.
    raise_error = False
    if mode == 'nodes':
        if not np.any(fvcom_obj.nodes):
            if verbose:
                print(f'No {mode} on which to interpolate on this boundary')
            return
        if not hasattr(fvcom_obj.sigma, 'layers'):
            raise_error = True
    elif mode == 'elements':
        if not hasattr(fvcom_obj.sigma, 'layers_center'):
            raise_error = True
        if not np.any(fvcom_obj.elements):
            if verbose:
                print(f'No {mode} on which to interpolate on this boundary')
            return

    if raise_error:
        raise AttributeError('Add vertical sigma coordinates in order to '
                + 'interpolate forcing along this boundary.')
    
    if 'elements' in mode:
        if cartesian:
            x = fvcom_obj.grid.xc
            y = fvcom_obj.grid.yc
        else:
            x = fvcom_obj.grid.lonc
            y = fvcom_obj.grid.latc
        # Keep positive down depths.
        z = -fvcom_obj.sigma.layers_center_z
    else:
        if cartesian:
            x = fvcom_obj.grid.x
            y = fvcom_obj.grid.y
        else:
            x = fvcom_obj.grid.lon
            y = fvcom_obj.grid.lat
        # Keep positive down depths.
        z = -fvcom_obj.sigma.layers_z

    if cartesian:
        x_coarse = coarse.grid.x
        y_coarse = coarse.grid.y
    else:
        x_coarse = coarse.grid.lon
        y_coarse = coarse.grid.lat

    if constrain_coordinates:
        raise AttributeError('Constrain coordinates not implemented for curvilinear data yet')

    nt = len(fvcom_obj.time.time)
    nx = len(x)
    nz = z.shape[-1]

    t_ind_min = np.min(np.where(coarse.time.datetime >= np.min(fvcom_obj.time.datetime)))
    t_ind_max = np.max(np.where(coarse.time.datetime <= np.max(fvcom_obj.time.datetime)))

    t_inds = np.arange(t_ind_min-1,t_ind_max+2)

    if verbose:
        print('Interpolating {} times, {} '.format(nt, nz)
                + 'vertical layers and {} points'.format(nx))

    # For curvilinear grids use scipy radial basis function interpolation
    # since regular grid interpolations require monotonic coordinates. This works best
    # doing each horizontal layer then seperately interpolating in time and vertical

    if mode == 'surface':
        if verbose:
            print('Interpolating surface data...', end=' ')

        interped_data_2d = []
        for this_t in ts:
            interped_data_2d.append(_rbf_interpolator_2d(getattr(coarse.data, coarse_name)[this_t,:],
                                            x_coarse,y_coarse,x,y))
        interped_data_2d = np.asarray(interped_data_2d)

        interped_coarse_data = []

        for this_pt in np.arange(interped_2d.shape[1]):
            interpolated_coarse_data.append(_linear_interpolator_1d(interped_data_2d[:,this_pt], coarse.time.time, fvcom_obj.time.time))

    else:
        if verbose:
            print('Interpolating z-level data...', end=' ')

        interped_3d_data = []

        z_coarse = []
        for this_layer in np.arange(0, coarse.grid.depth.shape[0]):
            z_coarse.append(fvcom_obj._rbf_interpolator_2d(coarse.grid.depth[this_layer,:,:],x_coarse,y_coarse,x,y))
        z_coarse = np.asarray(z_coarse)

        for this_t in t_inds:
            interped_2d_data = []

            for this_layer in np.arange(0, coarse.grid.depth.shape[0]):
                interped_2d_data.append(fvcom_obj._rbf_interpolator_2d(getattr(coarse.data, coarse_name)[this_t,this_layer,:],
                                            x_coarse,y_coarse,x,y))

            interped_2d_data = np.asarray(interped_2d_data)

            temp_array = []
            for this_pt in np.arange(0, interped_2d_data.shape[1]):
                mod_lays = fvcom_obj.sigma.layers_z[this_pt,:]
                mod_lays[mod_lays > np.max(z_coarse[:,this_pt])] = np.max(z_coarse[:,this_pt])
                mod_lays[mod_lays < np.min(z_coarse[:,this_pt])] = np.min(z_coarse[:,this_pt])

                temp_array.append(fvcom_obj._linear_interpolator_1d(interped_2d_data[:,this_pt], z_coarse[:,this_pt], fvcom_obj.sigma.layers_z[this_pt,:]))
            interped_3d_data.append(np.asarray(temp_array))

        interped_3d_data = np.asarray(interped_3d_data)

        interpolated_coarse_data = []
        for this_layer in np.arange(0, nz):
            this_2d = []
            for this_pt in np.arange(0, nx):
                this_2d.append(fvcom_obj._linear_interpolator_1d(interped_3d_data[:,this_pt, this_layer], coarse.time.time[t_inds], fvcom_obj.time.time))
            interpolated_coarse_data.append(np.asarray(this_2d))

    interpolated_coarse_data = np.asarray(interpolated_coarse_data)
    if tide_adjust and fvcom_name in ['u', 'v', 'ua', 'va', 'zeta']:
        if fvcom_name in ['u', 'v']:
            tide_levels = np.tile(getattr(fvcom_obj.tide, fvcom_name)
                    [:, np.newaxis, :], [1, nz, 1])
            interpolated_coarse_data = (interpolated_coarse_data
                    + tide_levels)
        else:
            interpolated_coarse_data = interpolated_coarse_data + getattr(
                    fvcom_obj.tide, fvcom_name)

    return interpolated_coarse_data

def _rbf_interpolator_2d(data, x, y, interp_x, interp_y, remove_mask=True):
    if remove_mask:
        data_mask = data.mask
        x = x[~data_mask]
        y = y[~data_mask]
        data = data[~data_mask]

    interpolater = si.Rbf(x, y, data, function='cubic', smooth=0)
    interped = interpolater(interp_x, interp_y)
    return interped

def _linear_interpolator_1d(data, x, interp_x):
    interpolater = si.interp1d(x,data)
    interped = interpolater(interp_x)
    return interped

def interpolate_variable_from_fvcom(fvcom_obj, var, coarse_fvcom, constrain_coordinates=False):
    """
    Interpolate the given regularly gridded data onto the grid nodes. Presently implemented for restart files so
    is 1-d in time.

    Parameters
    ----------
    coarse : PyFVCOM.preproc.RegularReader
        The regularly gridded data to interpolate onto the grid nodes. This must include time (coarse.time), lon,
        lat and depth data (in coarse.grid) as well as the time series to interpolate (4D volume [time, depth,
        lat, lon]) in coarse.data.
    constrain_coordinates : bool, optional
        Set to True to constrain the grid coordinates (lon, lat, depth) to the supplied coarse data.
        This essentially squashes the ogrid to fit inside the coarse data and is, therefore, a bit of a
        fudge! Defaults to False.
    """

    # This is more or less a copy-paste of PyFVCOM.grid.add_nested_forcing except we use the full grid
    # coordinates instead of those on the open boundary only. Feels like unnecessary duplication of code.

    # We need the vertical grid data for the interpolation, so load it now.

    coarse_data = np.squeeze(getattr(coarse_fvcom.data, var))
    var_shape = coarse_data.shape

    if var_shape[-1] == len(coarse_fvcom.grid.lon):
        mode='node'
        coarse_x = coarse_fvcom.grid.lon
        coarse_y = coarse_fvcom.grid.lat
        coarse_z = coarse_fvcom.grid.h * -coarse_fvcom.grid.siglay
        coarse_tri = coarse_fvcom.grid.triangles
    else:
        mode='elements'
        coarse_x = coarse_fvcom.grid.lonc
        coarse_y = coarse_fvcom.grid.latc
        coarse_z = coarse_fvcom.grid.h_center * -coarse_fvcom.grid.siglay_center

    if len(var_shape) == 1:
        mode+='_surface'

    fvcom_obj.load_data(['siglay'])
    fvcom_obj.data.siglay_center = nodes2elems(fvcom_obj.data.siglay, fvcom_obj.grid.triangles)
    if 'elements' in mode:
        x = copy.deepcopy(fvcom_obj.grid.lonc)
        y = copy.deepcopy(fvcom_obj.grid.latc)
        # Keep depths positive down.
        z = fvcom_obj.grid.h_center * -fvcom_obj.data.siglay_center
    else:
        x = copy.deepcopy(fvcom_obj.grid.lon[:])
        y = copy.deepcopy(fvcom_obj.grid.lat[:])
        # Keep depths positive down.
        z = fvcom_obj.grid.h * -fvcom_obj.data.siglay



    if constrain_coordinates:
        x[x < coarse_x.min()] = coarse_x.min()
        x[x > coarse_x.max()] = coarse_x.max()
        y[y < coarse_y.min()] = coarse_y.min()
        y[y > coarse_y.max()] = coarse_y.max()

        # Internal landmasses also need to be dealt with, so test if a point lies within the mask of the grid and
        # move it to the nearest in grid point if so.

        # use .in_domain
        is_in = coarse_fvcom.in_domain(x,y)

        if np.sum(is_in) < len(is_in):
            if 'elements' in mode:
                close_ind = coarse_fvcom.closest_element([x[~is_in], y[~is_in]])
            else:
                close_ind = coarse_fvcom.closest_node([x[~is_in], y[~is_in]])

            x[~is_in] = coarse_x[close_ind]
            y[~is_in] = coarse_y[close_ind]

        # The depth data work differently as we need to squeeze each FVCOM water column into the available coarse
        # data. The only way to do this is to adjust each FVCOM water column in turn by comparing with the
        # closest coarse depth.
        if 'surface' not in mode:
            # Go through each open boundary position and if its depth is deeper than the closest coarse data,
            # squash the open boundary water column into the coarse water column.
            node_tris = coarse_fvcom.grid.triangles[coarse_fvcom.closest_element([x,y]),:]

            for idx, node in enumerate(zip(x, y, z.T)):
                grid_depth = np.min(coarse_fvcom.grid.h[node_tris[idx,:]])

                if grid_depth < node[2].max():
                    # Squash the FVCOM water column into the coarse water column.
                    z[:, idx] = (node[2] / node[2].max()) * grid_depth
            # Fix all depths which are shallower than the shallowest coarse depth. This is more straightforward as
            # it's a single minimum across all the open boundary positions.
            z[z < coarse_z.min()] = coarse_z.min()


    # Now do the interpolation. We interpolate across each horizontal layer then linearly on depth.
    if 'surface' in mode:
        interp_data = _interp_to_fvcom_layer(coarse_x, coarse_y, coarse_tri, coarse_data, x, y)

    else:
        first_interp_coarse_data = []
        interped_coarse_z =[]
        for this_layer in np.arange(0,len(coarse_data)):
            print('Interp layer {} var'.format(this_layer))
            first_interp_coarse_data.append(_interp_to_fvcom_layer(coarse_x, coarse_y, coarse_tri, coarse_data[this_layer,:], x, y))
            print('Interp layer {} z'.format(this_layer))
            interped_coarse_z.append(_interp_to_fvcom_layer(coarse_x, coarse_y, coarse_tri, coarse_z[this_layer,:], x, y))

        print('Interp layer depth')
        interpolated_coarse_data = _interp_to_fvcom_depth(np.asarray(interped_coarse_z), np.asarray(first_interp_coarse_data), z)

    return interpolated_coarse_data[np.newaxis,:,:]

