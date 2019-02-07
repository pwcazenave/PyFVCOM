"""
Convert from a range of input coastline file types to a CST file compatible
with SMS.

CST (FVCOM) format is a text file structured as:

  COAST
  nArc
  n z
  x1 y1 z0
  ...
  xn yn zn
  x1 y1 z0

where nArc is the total number of arcs, n is the number of nodes in the arc and
z is a z value (typically zero). The polygon must close with the first value at
the end.


"""


import inspect
import shapefile

import numpy as np

from warnings import warn



def readESRIShapeFile(*args, **kwargs):
    warn('{} is deprecated. Use read_ESRI_shapefile instead.'.format(inspect.stack()[0][3]))
    return read_ESRI_shapefile(*args, **kwargs)


def readArcMIKE(*args, **kwargs):
    warn('{} is deprecated. Use read_arc_MIKE instead.'.format(inspect.stack()[0][3]))
    return read_arc_MIKE(*args, **kwargs)


def readCST(*args, **kwargs):
    warn('{} is deprecated. Use read_CST instead.'.format(inspect.stack()[0][3]))
    return read_CST(*args, **kwargs)


def writeCST(*args, **kwargs):
    warn('{} is deprecated. Use write_CST instead.'.format(inspect.stack()[0][3]))
    return write_CST(*args, **kwargs)


