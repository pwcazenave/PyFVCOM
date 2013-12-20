Welcome to fvcom-py!
--------------------

Table of contents
-----------------

[Introduction](#introduction)
[Prerequisites](#prerequisites)
[Provides](#provides)
[Installing](#installing)
[Examples](#examples)

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


