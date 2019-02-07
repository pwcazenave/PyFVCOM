import tempfile
import numpy as np
from datetime import datetime
from netCDF4 import Dataset, date2num

from PyFVCOM.coordinate import utm_from_lonlat
from PyFVCOM.grid import nodes2elems, element_side_lengths, unstructured_grid_depths
from PyFVCOM.utilities.time import date_range


def cfl(fvcom, timestep, depth_averaged=False, verbose=False, **kwargs):
    """
    Calculate the time-varying CFL for a given grid from the velocity and surface elevation time series.

    This is a python reimplementation of show_max_CFL written by Simon Waldman from the MATLAB fvcom-toolbox:
        https://gitlab.ecosystem-modelling.pml.ac.uk/fvcom/fvcom-toolbox/blob/dev/fvcom_postproc/show_max_CFL.m

    This differs from that function in that it return the time-varying CFL array rather than just the maximum in time.

    Parameters
    ----------
    fvcom : PyFVCOM.grid.FileReader
        A file reader object loaded from a netCDF file. This must include 'u', 'v' and 'zeta' data.
    timestep : float
        The external time step used in the model.
    depth_averaged : bool, optional
        Set to True to use depth-averaged data. Defaults to False (depth-resolved).
    verbose : bool, optional
        Print the location (sigma layer, element) of the maximum CFL value for the given time step. Defaults to not
        printing anything.

    Additional kwargs are passed to `PyFVCOM.read.FileReader.load_data()'.

    Returns
    -------
    cfl : np.ndarray
        An array of the time-varying CFL number.

    """

    g = 9.81  # acceleration due to gravity

    # Load the relevant data
    uname, vname = 'u', 'v'
    if depth_averaged:
        uname, vname = 'ua', 'va'
    fvcom.load_data(uname, **kwargs)
    fvcom.load_data(vname, **kwargs)
    fvcom.load_data('zeta', **kwargs)

    u = getattr(fvcom.data, uname)
    v = getattr(fvcom.data, vname)
    z = getattr(fvcom.data, 'zeta')

    element_sizes = element_side_lengths(fvcom.grid.triangles, fvcom.grid.x, fvcom.grid.y)
    minimum_element_size = np.min(element_sizes, axis=1)

    if depth_averaged:
        element_water_depth = fvcom.grid.h_center + nodes2elems(z, fvcom.grid.triangles)
    else:
        node_water_depths = unstructured_grid_depths(fvcom.grid.h, z, fvcom.grid.siglay)
        # Make water depths positive down so we don't get NaNs in the square root.
        element_water_depth = nodes2elems(-node_water_depths, fvcom.grid.triangles)

    # This is based on equation 6.1 on pg 33 of the MIKE hydrodynamic module manual (modified for using a single
    # characteristic length rather than deltaX/deltaY)
    cfl = (2 * np.sqrt(g * element_water_depth) + u + v) * (timestep / minimum_element_size)

    if verbose:
        val = np.nanmax(cfl)
        ind = np.unravel_index(np.nanargmax(cfl), cfl.shape)

        if depth_averaged:
            time_ind, element_ind = ind
            message = 'Maximum CFL first reached with an external timestep of {:f} seconds is approximately {:.3f} ' \
                      'in element {:d} (lon/lat: {}, {}) at {}.'
            print(message.format(timestep, val, element_ind,
                                 fvcom.grid.lonc[element_ind], fvcom.grid.latc[element_ind],
                                 fvcom.time.datetime[time_ind].strftime('%Y-%m-%d %H:%M:%S')))
        else:
            time_ind, layer_ind, element_ind = ind
            message = 'Maximum CFL first reached with an external timestep of {:f} seconds is approximately {:.3f} ' \
                      'in element {:d} (lon/lat: {}, {}) layer {:d} at {}.'
            print(message.format(timestep, val, element_ind,
                                 fvcom.grid.lonc[element_ind], fvcom.grid.latc[element_ind],
                                 layer_ind, fvcom.time.datetime[time_ind].strftime('%Y-%m-%d %H:%M:%S')))

    return cfl


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