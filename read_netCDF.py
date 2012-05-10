#!/usr/bin/env python

"""

Read a NetCDF file and list its contents.

"""

import netCDF3 as nc3
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

def dumpNC(file):
    """ Dump the headers from the NetCDF """
    try:
        nc = nc3.Dataset(file, 'r', format='NETCDF')
        print nc.file_format
        print nc.groups
    except:
        print 'booo!'


dumpNC('co2_z0.nc')
