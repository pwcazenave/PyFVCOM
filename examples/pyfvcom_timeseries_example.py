
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import num2date
from matplotlib import rcParams

from PyFVCOM.read_FVCOM_results import ncread
from PyFVCOM.grid_tools import findNearestPoint

# Multiple output files are transparently loaded by ncread.
fvcom = ['sample_april.nc', 'sample_may.nc', 'sample_june.nc']

# Positions we're interested in plotting. The findNearestPoint 
# function will find the closest node in the unstructured grid.
xy = np.array(((-4.5, 55), (-6.9, 52)))  # lon, lat pairs

# Extract only the surface layer for the plot.
dims = {'siglay': '0'}

# Our variables of interest.
varlist = ('lon', 'lat', 'time', 'temp')

FVCOM = ncread(fvcom, vars=varlist, dims=dims)

# Make datetime objects for the time series plots.
FVCOM['datetimes'] = num2date(FVCOM['time'], 'days since 1858-11-17 00:00:00')

# Find the nodes in the grid closest to the positions we're interested in plotting.
nx, ny, dist, idx = findNearestPoint(FVCOM['lon'], FVCOM['lat'], 
                                     xy[:, 0], xy[:, 1])

# Now plot the time series.

rcParams['mathtext.default'] = 'regular'  # sensible font for the LaTeX labelling

fig = plt.figure(figsize=(10, 7))  # size in inches
for c, ind in enumerate(idx):
    ax = fig.add_subplot(len(idx), 1, c + 1)
    ln = ax.plot(FVCOM['datetimes'], FVCOM['temp'][:, ind])
    ax.set_title('Sea surface temperature nearest to position {}, {}'.format(*xy[c, :]))
    ax.set_ylabel('Temperature ($^{\circ}C$)')



