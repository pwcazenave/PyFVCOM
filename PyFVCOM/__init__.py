"""
The FVCOM Python toolbox (PyFVCOM)

"""

__version__ = '2.2.0'
__author__ = 'Pierre Cazenave'
__credits__ = ['Pierre Cazenave', 'Michael Bedington', 'Ricardo Torres']
__license__ = 'MIT'
__maintainer__ = 'Pierre Cazenave'
__email__ = 'pica@pml.ac.uk'

import sys

# Import everything! Eventually, we're going to hit a circular dependency here...
from PyFVCOM import buoy
from PyFVCOM import ctd
from PyFVCOM import current
from PyFVCOM import grid
from PyFVCOM import coordinate
from PyFVCOM import ocean
from PyFVCOM import stats
from PyFVCOM import tidal_ellipse
from PyFVCOM import tide
from PyFVCOM import plot
from PyFVCOM import preproc
from PyFVCOM import read
from PyFVCOM import utilities
from PyFVCOM import validation
from PyFVCOM import interpolate

if sys.version_info.major < 3 and sys.version_info.minor < 6:
    raise Exception('Must be using Python 3.6 or greater')

