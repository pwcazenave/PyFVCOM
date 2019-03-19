#from PyFVCOM.read import FileReader
from PyFVCOM.grid import unstructured_grid_depths, node_to_centre, get_boundary_polygons
import shapely.geometry as sg
#import scipy.interpolate as si
import numpy as np

#from mpi4py import MPI


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
            this_poly = sg.Polygon(fvcom_ll[this_poly_pts,:])
            for i, this_pt in enumerate(grid_point_list):
                if is_island and this_poly.contains(this_pt):
                    loc_code[i] = 1
                elif not is_island and this_poly.contains(this_pt) and loc_code[i] != 1:
                    loc_code[i] = 2

        out_of_domain_mask = np.logical_or(loc_code == 0, loc_code == 1)
         

    else:
        boundary_polygon_list = get_boundary_polygons(fvcom_tri)
        boundary_polys = [sg.Polygon(fvcom_ll[this_poly_pts,:]) for this_poly_pts in boundary_polygon_list]
    
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

    def __init__(self, comm=None, root=0, verbose=False):
        """
        Create interpolation worker

        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm, optional
            The MPI intracommunicator object. Omit if not running in parallel.
        root : int, optional
            Specify a given rank to act as the root process. This is only for outputting verbose messages (if enabled
            with `verbose').
        verbose : bool, optional
            Set to True to enabled some verbose output messages. Defaults to False (no messages).

        """
        self.dims = None

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

        self.fvcom_grid_vars = PassiveDataStore
        self.regular_grid_vars = PassiveDataStore

    def __loader(self, fvcom_file, variable):
        """
        Function to load fvcom variables

        Parameters
        ----------
        fvcom_file : str, pathlib.Path
            The file to load.
        variable : str
            The variable name to load from `fvcom_file'.

        Provides
        --------
        self.fvcom : PyFVCOM.read.FileReader
            The FVCOM data ready for interpolating.
        """
        pass

    def InitialiseGrid(self, fvcom_file, time_indices, lower_left_ll, upper_right_ll, depth_layers, mode='nodes', time_varying_depth=True):
        

        if mode == 'nodes':
            this_var_h = fvcom_grid_fr.grid.h
            this_var_zeta = fvcom_grid_fr.data.zeta
            this_var_sigma = fvcom_grid_fr.grid.siglay
            this_var_points = fvcom_nodes
            this_var_fvcom_ll = np.asarray([fvcom_grid_fr.grid.lon, fvcom_grid_fr.grid.lat])

        elif mode == 'elements':
            this_var_h = fvcom_grid_fr.grid.h_center
            this_var_zeta = node_to_centre(fvcom_grid_fr.data.zeta, fvcom_grid_fr)
            this_var_sigma = fvcom_grid_gr.grid.siglay_center
            this_var_points = fvcom_elements
            this_var_fvcom_ll = np.asarray([fvcom_grid_fr.grid.lonc, fvcom_grid_fr.grid.latc])

        fvcom_dep_lays = -pf.grid.unstructured_grid_depths(this_var_h, this_var_zeta, this_var_sigma)
        ## change to depth from free surface since I *think* this is what cmems does?
        fvcom_dep_lays = fvcom_dep_lays - np.tile(np.min(fvcom_dep_lays, axis=1)[:,np.newaxis,:], [1,fvcom_dep_lays.shape[1], 1])

        for this_t in time:
            # do this_depth_layer_points for each timestep
            print(this_t)

    def InterpolateRegular(self, fvcom_file, time_indices, variable):
        """
        Actually do the interpolation
        """
        
        for this_t in time_indices:
        
            for this_dep_lay in self.dep_lays: 
                reg_grid_data[:,:,this_depth_lay_ind] = grid_z2[:].T

    def _Interpolater(self):
        grid_z2 = si.griddata(this_var_fvcom_ll[:,this_depth_layer_points].T, depth_lay_data[this_depth_lay_ind,~np.isnan(depth_lay_data[this_depth_lay_ind,:])], (grid_mesh_lons, grid_mesh_lats), method='cubic') 


"""
