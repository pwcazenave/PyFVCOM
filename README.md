Welcome to PyFVCOM!
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

PyFVCOM is a collection of various tools and utilities which can be used to extract, analyse and plot input and output files from FVCOM.


Prerequisites
-------------

* Python. This has been written against Python 2.7, so in principle anything newer than that (in the 2.x series) should be OK.

* numpy, again, this has been tested with numpy versions 1.4.1 and 1.6.2.

* scipy, versions 0.7.2 and 0.10.1.

* matplotlib, versions 1.0.1 and 1.2.1.

* netCDF4, version 0.9.9.

* jdcal, version 1.0.

Optionally:

* iPython, version 0.10.2. This makes for a good development environment, particularly when invoked with the -pylab argument, which automatically imports matplotlib.pylab and numpy.


Provides
--------

* buoy_tools - read data from an SQLite3 database of BODC buoy data.
    - getBuoyMetadata
    - getBuoyData

* cst_tools - create coastline files for SMS from shapefiles or DHI MIKE arcs.
    - readESRIShapeFile
    - readArcMIKE
    - readCST
    - writeCST

* ctd_tools - interrogate an SQLite data base of CTD casts.
    - getCTDMetadata
    - getCTDData
    - getFerryBoxData

* grid_tools - tools to parse SMS, DHI MIKE and FVCOM unstructured grids. Also provides functionality to add coasts and clip triangulations to a given domain. Functions to parse FVCOM river files are also included, as is a function to resample an unstructured grid onto a regular grid (without interpolation, simply finding the nearest point within a threshold distance).
    - parseUnstructuredGridSMS
    - parseUnstructuredGridFVCOM
    - parseUnstructuredGridMIKE
    - parseUnstructuredGridGMSH
    - writeUnstructuredGridSMS
    - writeUnstructuredGridSMSBathy
    - writeUnstructuredGridMIKE
    - findNearestPoint
    - elementSideLengths
    - fixCoordinates
    - clipTri
    - clipDomain
    - getRiverConfig
    - getRivers
    - mesh2grid
    - lineSample
    - OSGB36toWGS84
    - connectivity

* img2xyz - simple script to try and convert images to depths (or elevations).
    - rgb2z

* ll2utm - convert from spherical to cartesian UTM coordinates and back. Available from <http://robotics.ai.uiuc.edu/~hyoon24/LatLongUTMconversion.py>.
    - LLtoUTM
    - UTMtoLL

* ocean_tools - a number of routines to convert between combinations of temperature, salinity, pressure, depth and density.
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
    - cond2salt
    - vorticity (currently empty)
    - zbar
    - epa
    - simpsonhunter
    - mixedlayerdepth
    - stokes

* process_FVCOM_results - perform some analyses on FVCOM data read in using read_FVCOM_results.
    - calculateTotalCO2
    - CO2LeakBudget
    - dataAverage
    - unstructuredGridVolume
    - residualFlow

* read_FVCOM_results - parse the netCDF model output and extract a subset of the variables.
    - readFVCOM
    - ncread (wrapper around readFVCOM)
    - readProbes
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

In principle, python setup.py should install PyFVCOM, though it is untested. Alternatively, download the PyFVCOM directory, and add its contents to your PYTHONPATH.

Examples
--------

Below are some brief examples of how to use the toolbox.

```python
""" Plot a surface from an FVCOM model output.

"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from PyFVCOM.read_FVCOM_results import readFVCOM

if __name__ == '__main__':

    fvcom = '/path/to/fvcom/netcdf/output.nc'

    # Extract only the first 20 time steps.
    dims = {'time':':20'}

    # List of the variables to extract.
    vars = ('lon', 'lat', 'nv', 'zeta', 'Times')

    FVCOM = readFVCOM(fvcom, vars, clipDims=dims, noisy=True)

    # Create the triangulation table array (with Python indexing
    # [zero-based])
    triangles = FVCOM['nv'].transpose() - 1

    # Find the domain extents.
    extents = np.array((FVCOM['lon'].min(),
                       FVCOM['lon'].max(),
                       FVCOM['lat'].min(),
                       FVCOM['lat'].max()))

    # Create a Basemap instance for plotting coastlines and so on.
    m = Basemap(llcrnrlon=extents[:2].min(),
            llcrnrlat=extents[-2:].min(),
            urcrnrlon=extents[:2].max(),
            urcrnrlat=extents[-2:].max(),
            rsphere=(6378137.00, 6356752.3142),
            resolution='h',
            projection='merc',
            lat_0=extents[-2:].mean(),
            lon_0=extents[:2].mean(),
            lat_ts=extents[:2].mean())

    parallels = np.arange(floor(extents[2]), ceil(extents[3]), 1)
    meridians = np.arange(fllor(extents[0]), ceil(extents[1]), 1)

    # Create the new figure and add the data.
    x, y = m(FVCOM['lon'], FVCOM['lat'])
    fig0 = plt.figure(figsize=(10, 10)) # size in inches
    ax = fig0.add_axes([0.1, 0.1, 0.8, 0.8])

    m.drawmapboundary()
    m.drawcoastlines(zorder=100)
    m.fillcontinents(color='0.6', zorder=100)
    m.drawparallels(parallels, labels=[1, 0, 0, 0],
                    fontsize=10, linewidth=0)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1],
                    fontsize=10, linewidth=0)

    # Plot the last surface elevation time step.
    CS1 = ax.tripcolor(x, y, triangles, FVCOM['temp'][-1, :])

    # Clip the colour palette.
    CS1.set_clim(4, 11)

    # Add title, colour bar and so on.
    ax.set_title(''.join(FVCOM['Times'][-1, :-4]))
    cb = fig0.colorbar(CS1)
    cb.set_label('Temperature $(^{\circ}C)$')

    fig0.show()
```


```python
""" Plot a time series of temperature at a given position.

"""

import numpy as np
import matplotlib.pyplot as plt

from PyFVCOM.read_FVCOM_results import readFVCOM
from PyFVCOM.grid_tools import findNearestPoint

if __name__ == '__main__':

    fvcom = '/path/to/fvcom/netcdf/output.nc'

    # Positions we're interested in plotting. The findNearestPoint function will
    # find the closest node in the unstructured grid.
    xy = np.array((-4.5, 55), (-6.9, 53)) # lon/lat pairs.

    # Extract only the surface layer.
    dims = {'siglay':'0'}

    # List of the variables to extract.
    vars = ('lon', 'lat', 'time', 'temp')

    FVCOM = readFVCOM(fvcom, vars, clipDims=dims, noisy=True)

    # Find the model node indices.
    nearestX, nearestY, dist, idx = findNearestPoint(
            FVCOM['lon'], FVCOM['lat'], xy[:, 0], xy[:, 1], noisy=True
            )

    fig0 = plt.figure()
    for c, ind in enumerate(idx):
        ax = fig0.add_subplot(len(idx), 1, c + 1)
        LN0 = fig0.plot(FVCOM['time'], FVCOM['zeta'][idx, :], 'g')
        ax.set_title('Surface elevation at {}, {}'.format(
            xy[c, 0], xy[c, 1])
            )
        ax.set_xlabel('Time (Modified Julian Days)')
        ax.set_ylabel('Surface elevation (m)')

    fig0.show()
```


```python
""" Plot tidal ellipses from a model output.

"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from PyFVCOM.read_FVCOM_results import readFVCOM
from PyFVCOM.tidal_ellipse import ap2ep, prep_plot
from PyFVCOM.tide_tools import TAPPy

if __name__ == '__main__':

    fvcom = '/path/to/fvcom/netcdf/output.nc'

    vars = ['lonc', 'latc', 'u', 'v', 'Times']
    dims = {'time':':360'} # first 15 days at hourly sampling

    # Define a plot subset.
    subset = np.array(((-5, -6), (49, 50))) # ((xmin, xmax), (ymin, ymax))

    # Scaling factor for the ellipses. You may need to experiment with this
    # value.
    scaling = 5000

    # Find the model nodes which fall within the subset defined above.
    FVCOM = readFVCOM(fvcom, ['lonc', 'latc'])
    elems = np.where((FVCOM['lonc'] > subset[0].min()) *
                     (FVCOM['lonc'] < subset[0].max()) *
                     (FVCOM['latc'] > subset[1].min()) *
                     (FVCOM['latc'] < subset[1].max()))[0]
    dims.update({'nele':str(elems.tolist())})

    FVCOM = readFVCOM(fvcom, vars, clipDims=dims, noisy=True)

    # Depth-average the velocity components.
    uBar, vBar = FVCOM['u'].mean(axis=1), FVCOM['v'].mean(axis=1)

    # Create a time array for the TAPPy call.
    years = [''.join(i) for i in FVCOM['Times'][:, 0:4]]
    months = [''.join(i) for i in FVCOM['Times'][:, 5:7]]
    days = [''.join(i) for i in FVCOM['Times'][:, 8:10]]
    hours = [''.join(i) for i in FVCOM['Times'][:, 11:13]]
    minutes = [''.join(i) for i in FVCOM['Times'][:, 14:16]]
    seconds = [''.join(i) for i in FVCOM['Times'][:, 17:19]]
    Times = np.column_stack((
        years,
        months,
        days,
        hours,
        minutes,
        seconds
        )).astype(int)

    # Combine the u and v components as a complex array (so speed is simply
    # np.abs(uv)).
    uv = uBar + 1j * vBar

    # Create dicts for the results.
    uharmonics, vharmonics = {}, {}

    # Transpose so we can analyse a time series for each location. This takes
    # a little while to run for lots of points.
    for i, comp in enumerate(uv.transpose()):

        print('{} of {}...'.format(i, uv.shape[-1])),

        # Combine the Times and velocity data.
        u = np.column_stack((Times, comp.real))
        v = np.column_stack((Times, comp.imag))

        # Create a dict for the TAPPy results.
        uharm, vharm = {}, {}
        uharm['name'], uharm['speed'], uharm['phase'], uharm['amp'], uharm['infer'] = TAPPy(u)
        vharm['name'], vharm['speed'], vharm['phase'], vharm['amp'], vharm['infer'] = TAPPy(v)

        # Put the results into a meta dict with the key being the current
        # position.
        key = '{}-{}'.format(FVCOM['lonc'][i], FVCOM['latc'][i])
        uharmonics[key] = uharm
        vharmonics[key] = vharm

        print('done.')

    # Now plot the M2 results.
    m = Basemap(llcrnrlon=subset[0].min(),
            llcrnrlat=subset[1].min(),
            urcrnrlon=subset[0].max(),
            urcrnrlat=subset[1].max(),
            rsphere=(6378137.00,6356752.3142),
            resolution='h',
            projection='merc',
            lat_0=subset[1].mean(),
            lon_0=subset[0].mean(),
            lat_ts=subset[0].mean())

    parallels = np.arange(np.floor(subset[1].min()),
            np.ceil(subset[1].max()), 0.1)
    meridians = np.arange(np.floor(subset[0].min()),
            np.ceil(subset[0].max()), 0.2)

    x, y = m(FVCOM['lonc'], FVCOM['latc'])

    fig1 = plt.figure(figsize=(12, 12))
    ax1 = fig1.add_subplot(1, 1, 1)

    m.drawmapboundary()
    m.drawcoastlines(zorder=100)
    m.fillcontinents(color='0.6', zorder=100)
    m.drawparallels(parallels, labels=[1,0,0,0],
            fontsize=10, linewidth=0)
    m.drawmeridians(meridians, labels=[0,0,0,1],
            fontsize=10, linewidth=0)

    ax1.set_title('M2 tidal ellipses')

    for i, xy in enumerate(np.column_stack((x, y))):
        # Make a key from the original positions. This is the key which we
        # use to extract the harmonic results.
        key = '{}-{}'.format(FVCOM['lonc'][i], FVCOM['latc'][i])

        # Find the M2 data position for the current position.
        idx = uharmonics[key]['name'].index('M2')

        # Extract the amplitude and phase and calculate the ellipse.
        uZ = float(uharmonics[key]['amp'][idx])
        vZ = float(vharmonics[key]['amp'][idx])
        uG = float(uharmonics[key]['phase'][idx])
        vG = float(vharmonics[key]['phase'][idx])
        SEMA, ECC, INC, PHA, w = ap2ep(uZ, uG, vZ, vG)
        w, wmin, wmax = prep_plot(SEMA, ECC, INC, PHA)

        ax1.plot((scaling * np.real(w)) + xy[0],
                (scaling * np.imag(w)) + xy[1],
                'k', linewidth=2)

        fig1.show()

```
