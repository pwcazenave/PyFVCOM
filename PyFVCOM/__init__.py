"""
The FVCOM Python toolbox (PyFVCOM)

"""

__version__ = '1.4.1'
__author__ = 'Pierre Cazenave'
__credits__ = ['Pierre Cazenave']
__license__ = 'MIT'
__maintainer__ = 'Pierre Cazenave'
__email__ = 'pica@pml.ac.uk'

import inspect
from warnings import warn

# Import everything!
from PyFVCOM import buoy_tools
from PyFVCOM import cst_tools
from PyFVCOM import ctd_tools
from PyFVCOM import grid_tools
from PyFVCOM import ll2utm
from PyFVCOM import ocean_tools
from PyFVCOM import stats_tools
from PyFVCOM import tide_tools
from PyFVCOM import tidal_ellipse
from PyFVCOM import process_results
from PyFVCOM import read_results

