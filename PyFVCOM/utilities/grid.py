import tempfile
import numpy as np
from datetime import datetime
from netCDF4 import Dataset, date2num

from PyFVCOM.coordinate import utm_from_lonlat
from PyFVCOM.grid import nodes2elems, element_side_lengths, unstructured_grid_depths
from PyFVCOM.utilities.time import date_range


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