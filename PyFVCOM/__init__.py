"""
The FVCOM Python toolbox (PyFvcom)

"""

__version__ = '1.2'
__author__ = 'Pierre Cazenave'
__credits__ = ['Pierre Cazenave']
__license__ = 'MIT'
__maintainer__ = 'Pierre Cazenave'
__email__ = 'pica@pml.ac.uk'

# Import numpy so we have it across the board.
import numpy as np

# Import everything!
import buoy_tools
import cst_tools
import ctd_tools
import grid_tools
import img2xyz
import ll2utm
import ocean_tools
import process_FVCOM_results
import read_FVCOM_results
import stats_tools
import tidal_ellipse
import tide_tools
# External TAPPY now instead of my bundled version. Requires my forked version
# of TAPPY from https://github.com/pwcazenave/tappy or
# http://gitlab.em.pml.ac.uk/pica/tappy.
from tappy import tappy
