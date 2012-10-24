Welcome to fvcom-py!
--------------------


Introduction
------------

fvcom-py is a collection of various tools and utilities which can be used to extract, analyse and plot input and output files from FVCOM.


Prerequisites
-------------

* Python. This has been written against Python 2.7, so in principle anything newer than that (in the 2.x series) should be OK.

* numpy, again, this has been written with numpy version 1.4.1, so consider that a minimum.

* scipy, version 0.7.2.

* matplotlib, version 1.0.1.

* netCDF4, version 0.9.9.

* shapefile, a tool to parse ESRI Shapefiles, available from <http://sourceforge.net/projects/pyshape/>.

* OSGeo, another tool to read ESRI Shapefiles, available from <http://wiki.osgeo.org/wiki/OSGeo_Python_Library>.

* TAPPy, a harmonic analysis tool for surface elevation time series, available from <http://sourceforge.net/projects/tappy/>.

Optionally:

* ipython, version 0.10.2. This makes for a good development environment, particularly when invocated with the -pylab argument, which automatically imports matplotlib.pylab and numpy.


Provides
--------

* cst_tools - create coastline files for SMS from shapefiles or DHI MIKE arcs.
    - readESRIShapeFile
    - readArcMIKE

* grid_tools - tools to parse SMS, DHI MIKE and FVCOM unstructured grids.
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

* ll2utm - available from <http://robotics.ai.uiuc.edu/~hyoon24/LatLongUTMconversion.py>.

* process_FVCOM_results - perform some analyses on FVCOM data read in using read_FVCOM_results.
    - calculateTotalCO2
    - CO2LeakBudget
    - dataAverage
    - unstructuredGridVolume
    - animateModelOutput
    - residualFlow

* read_FVCOM_results - parse the NetCDF model output and extract a subset of the variables.
    - readFVCOM
    - getSurfaceElevation

* stats_tools - some basic statistics tools.
    - calculateRegression
    - calculatePolyfit
    - coefficientOfDetermination

* tide_tools - tools to use and abuse tidal data from an SQLite database of tidal time series.
    - julianDay
    - addHarmonicResults
    - getObservedData
    - getObservedMetadata
    - cleanObservedData
    - runTAPPy
    - parseTAPPyXML
    - getHarmonics
    - getHarmonicsPOLPRED


Installing
----------

In principle, python setup.py should install fvcom-py, though it is untested. Alternatively, download the fvcom-py directory, and add its contents to your PYTHONPATH.

Running
-------

See some examples at <https://github.com/pwcazenave/pml-irish-sea/tree/master/python>.
