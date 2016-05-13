"""
The FVCOM Python toolbox (PyFvcom)

"""

__version__ = '1.2'
__author__ = 'Pierre Cazenave'
__credits__ = ['Pierre Cazenave']
__license__ = 'MIT'
__maintainer__ = 'Pierre Cazenave'
__email__ = 'pica@pml.ac.uk'

import inspect
from warnings import warn
# Import numpy so we have it across the board.
import numpy as np

# Import everything!
from PyFVCOM import buoy_tools
from PyFVCOM import cst_tools
from PyFVCOM import ctd_tools
from PyFVCOM import grid_tools
from PyFVCOM import img2xyz
from PyFVCOM import ll2utm
from PyFVCOM import ocean_tools
from PyFVCOM import stats_tools
from PyFVCOM import tide_tools
from PyFVCOM import tidal_ellipse
from PyFVCOM import process_results
from PyFVCOM import read_results

# External TAPPY now instead of my bundled version. Requires my forked version
# of TAPPY from https://github.com/pwcazenave/tappy or
# http://gitlab.em.pml.ac.uk/pica/tappy.
from tappy import tappy

# For backwards-compatibility.
process_FVCOM_results = process_results
read_FVCOM_results = read_results