Welcome to fvcom-py!
--------------------

Table of contents
-----------------

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Provides](#provides)
- [Installing](#installing)
- [Examples](#examples)

Introduction
------------

fvcom-py is a collection of various tools and utilities which can be used to extract, analyse and plot input and output files from FVCOM.


Prerequisites
-------------

* Python. This has been written against Python 2.7, so in principle anything newer than that (in the 2.x series) should be OK.

* numpy, again, this has been tested with numpy versions 1.4.1 and 1.6.2.

* scipy, versions 0.7.2 and 0.10.1.

* matplotlib, versions 1.0.1 and 1.2.1.

* netCDF4, version 0.9.9.

* shapefile, a tool to parse ESRI Shapefiles, available from <http://sourceforge.net/projects/pyshape/>.

* OSGeo, another tool to read ESRI Shapefiles, available from <http://wiki.osgeo.org/wiki/OSGeo_Python_Library>.

* TAPPy, a harmonic analysis tool for surface elevation time series, available from <http://sourceforge.net/projects/tappy/>. A slightly modified version of the TAPPy library is included in the toolbox (to allow importing as a module).

Optionally:

* iPython, version 0.10.2. This makes for a good development environment, particularly when invoked with the -pylab argument, which automatically imports matplotlib.pylab and numpy.


Provides
--------

* ctd_tools - interrogate an SQLite data base of CTD casts.
    - getCTDMetadata
    - getCTDData

* cst_tools - create coastline files for SMS from shapefiles or DHI MIKE arcs.
    - readESRIShapeFile
    - readArcMIKE

* grid_tools - tools to parse SMS, DHI MIKE and FVCOM unstructured grids. Also provides functionality to add coasts and clip triangulations to a given domain. Functions to parse SMS river files are also included, as is a function to resample an unstructured grid onto a regular grid (without interpolation, simply finding the nearest point within a threshold distance).
    - parseUnstructuredGridSMS
    - parseUnstructuredGridFVCOM
    - parseUnstructuredGridMIKE
    - writeUnstructuredGridSMS
    - writeUnstructuredGridSMSBathy
    - writeUnstructuredGridMIKE
    - plotUnstructuredGrid
    - plotUnstructuredGridProjected
    - findNearestPoint
    - elementSideLengths
    - fixCoordinates
    - plotCoast
    - clipTri
    - getRiverConfig
    - getRivers
    - mesh2grid

* ll2utm - convert from spherical to cartesian UTM coordinates and back. Available from <http://robotics.ai.uiuc.edu/~hyoon24/LatLongUTMconversion.py>.
    - LLtoUTM
    - UTMtoLL

* oceal_tools - a number of routines to convert between combinations of temperature, salinity, pressure, depth and density.
    - pressure2depth
    - depth2pressure
    - dT_adiab_sw
    - theta_sw
    - cp_sw
    - sw_smow
    - sw_dens0
    - sw_seck
    - sw_dens
    - sw_svan
    - sw_sal78
    - sw_sal80
    - sw_salinity
    - dens_jackett

* process_FVCOM_results - perform some analyses on FVCOM data read in using read_FVCOM_results.
    - calculateTotalCO2
    - CO2LeakBudget
    - dataAverage
    - unstructuredGridVolume
    - animateModelOutput
    - residualFlow

* read_FVCOM_results - parse the NetCDF model output and extract a subset of the variables.
    - readFVCOM
    - elems2nodes
    - nodes2elems
    - getSurfaceElevation

* stats_tools - some basic statistics tools.
    - calculateRegression
    - calculatePolyfit
    - coefficientOfDetermination

* tidal_ellipse - Python version of the Tidal Ellipse MATLAB toolbox <http://woodshole.er.usgs.gov/operations/sea-mat/tidal_ellipse-html/index.html>.
    - ap2ep
    - ep2ap
    - cBEpm
    - get_BE
    - sub2ind
    - plot_ell
    - do_the_plot
    - prep_plot

* tide_tools - tools to use and abuse tidal data from an SQLite database of tidal time series.
    - julianDay
    - gregorianDate
    - addHarmonicResults
    - getObservedData
    - getObservedMetadata
    - cleanObservedData
    - TAPPy
    - runTAPPy
    - parseTAPPyXML
    - getHarmonics
    - readPOLPRED
    - gridPOLPRED
    - getHarmonicsPOLPRED


Installing
----------

In principle, python setup.py should install fvcom-py, though it is untested. Alternatively, download the fvcom-py directory, and add its contents to your PYTHONPATH.

Examples
--------

Below are some brief examples of how to use the toolbox.

```python
""" Plot a surface from an FVCOM model output.

"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from read_FVCOM_results import readFVCOM

if __name__ == '__main__':

    fvcom = '/path/to/fvcom/netcdf/output.nc'

    # Extract only the first 20 time steps.
    dims = {'time':':20'}

    # List of the variables to extract.
    vars = ['lon', 'lat', 'nv', 'zeta', 'Times']

    FVCOM = readFVCOM(fvcom, vars, clipDims=dims, noisy=True)

    # Create the triangulation table array (with Python indexing [zero-based])
    triangles = FVCOM['nv'].transpose() - 1

    # Find the domain extents.
    extents = np.array([FVCOM['lon'].min(),
                       FVCOM['lon'].max(),
                       FVCOM['lat'].min(),
                       FVCOM['lat'].max()]

    # Create a Basemap instance for plotting coastlines and so on.
    m = Basemap(llcrnrlon=extents[0:2].min(),
            llcrnrlat=extents[-2:].min(),
            urcrnrlon=extents[0:2].max(),
            urcrnrlat=extents[-2:].max(),
            rsphere=(6378137.00,6356752.3142),
            resolution='h',
            projection='merc',
            lat_0=extents[-2:].mean(),
            lon_0=extents[0:2].mean(),
            lat_ts=2.0)

    parallels = np.arange(floor(extents[2]), ceil(extents[3]), 1)
    meridians = np.arange(fllor(extents[0]), ceil(extents[1]), 1)

    # Create the new figure and add the data.
    x, y = m(FVCOM['lon'], FVCOM['lat'])
    fig0 = plt.figure(figsize=(10, 10)) # size in inches
    ax = fig0.add_axes([0.1, 0.1, 0.8, 0.8])

    m.drawmapboundary()
    m.drawcoastlines(zorder=100)
    m.fillcontinents(color='0.6', zorder=100)
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=10, linewidth=0)
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10, linewidth=0)

    # Plot the last surface elevation time step.
    CS1 = ax.tripcolor(x, y, triangles, FVCOM['temp'][-1, :])

    # Clip the colour palette.
    CS1.set_clim(4, 11)

    # Add title, colour bar and so on.
    ax.set_title(''.join(FVCOM['Times'][-1, :-4]))
    cb = fig0.colorbar(CS1)
    cb.set_label('Surface elevation (m)')

    fig0.show()
```


```python
""" Plot a time series of temperature at a given position.

"""

import numpy as np
import matplotlib.pyplot as plt

from read_FVCOM_results import readFVCOM
from grid_tools import findNearestPoint

if __name__ == '__main__':

    fvcom = '/path/to/fvcom/netcdf/output.nc'

    # Positions we're interested in plotting. The findNearestPoint function will
    # find the closest node in the unstructured grid.
    xy = np.array([-4.5, 55], [-6.9, 53]) # lon/lat pairs.

    # Extract only the surface layer.
    dims = {'siglay':'0'}

    # List of the variables to extract.
    vars = ['lon', 'lat', 'time', 'temp']

    FVCOM = readFVCOM(fvcom, vars, clipDims=dims, noisy=True)

    # Find the model node indices.
    nearestX, nearestY, dist, idx = findNearestPoint(FVCOM['lon'], FVCOM['lat'],
                                                     xy[:, 0], xy[:, 1],
                                                     noisy=True)

    fig0 = plt.figure()
    for c, ind in enumerate(idx):
        ax = fig0.add_subplot(len(idx), 1, c + 1)
        LN0 = fig0.plot(FVCOM['time'], FVCOM['zeta'][idx, :], 'g')
        ax.set_title('Surface elevation at {}, {}'.format(xy[c, 0], xy[c, 1]))
        ax.set_xlabel('Time (Modified Julian Days)')
        ax.set_ylabel('Surface elevation (m)')

    fig0.show()
```
