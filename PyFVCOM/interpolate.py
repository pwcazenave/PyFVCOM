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
                        print('Rank {}: Time step {}, Depth {}'.format(self.rank, this_t, this_depth_lay_ind))
                    this_depth_layer_nodes = np.where(self.fvcom_grid.total_depth[this_t, :] >= this_depth_lay)[0]
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
