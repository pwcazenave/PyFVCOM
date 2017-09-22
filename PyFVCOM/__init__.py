"""
The FVCOM Python toolbox (PyFVCOM)

"""

__version__ = '1.6.2'
__author__ = 'Pierre Cazenave'
__credits__ = ['Pierre Cazenave']
__license__ = 'MIT'
__maintainer__ = 'Pierre Cazenave'
__email__ = 'pica@pml.ac.uk'

import inspect
from warnings import warn

# Import everything!
from PyFVCOM import buoy
from PyFVCOM import coast
from PyFVCOM import ctd
from PyFVCOM import current
from PyFVCOM import grid
from PyFVCOM import coordinate
from PyFVCOM import ocean
from PyFVCOM import stats
from PyFVCOM import tidal_ellipse
from PyFVCOM import tide
from PyFVCOM import plot
from PyFVCOM import read_results
from PyFVCOM import utilities
