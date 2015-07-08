"""
The FVCOM Python toolbox (pyfvcom)

"""

__version__ = '1.0'
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
