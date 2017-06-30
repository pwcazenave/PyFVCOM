Welcome to PyFVCOM!
--------------------

Table of contents
-----------------

- [Introduction](#introduction)
- [Installing](#installing)
- [Prerequisites](#prerequisites)
- [Provides](#provides)
- [Examples](#examples)

Introduction
------------

PyFVCOM is a collection of various tools and utilities which can be used to extract, analyse and plot input and output files from FVCOM.

Installing
----------

Easiest way is to install with pip/pip3:

```python
pip install PyFVCOM
```

Prerequisites
-------------

* Python. This has been written against Python 2.7.11 and tentatively against Python 3.4.3, so in principle anything newer than those should be OK.

* numpy, again, this has been tested with numpy version 1.4.1 to 1.10.4.

* scipy, versions 0.7.2 to 0.14.1.

* matplotlib, versions 1.0.1 to 1.5.1.

* netCDF4, version 0.9.9 to 1.1.5.

* jdcal, version 1.0 to 1.2.

* pyshp, version 1.2.3.

Optionally:

* iPython, version 0.10.2 to 3.2.1. This makes for a good development environment, particularly when invoked with the -pylab argument, which automatically imports matplotlib.pylab and numpy, amongst others.


Provides
--------

* buoy_tools - read data from an SQLite3 database of BODC buoy data.
    - get_buoy_metadata
    - get_buoy_data

* cst_tools - create coastline files for SMS from shapefiles or DHI MIKE arcs.
    - read_ESRI_shapefile
    - read_arc_MIKE
    - read_CST
    - write_CST

* ctd_tools - interrogate an SQLite data base of CTD casts.
    - get_CTD_metadata
    - get_CTD_data
    - get_ferrybox_data

* current_tools - convert from vector components to scalars and back.
    - scalar2vector
    - vector2scalar

* grid_tools - tools to parse SMS, DHI MIKE and FVCOM unstructured grids. Also provides functionality to add coasts and clip triangulations to a given domain. Functions to parse FVCOM river files are also included, as is a function to resample an unstructured grid onto a regular grid (without interpolation, simply finding the nearest point within a threshold distance).
    - read_sms_mesh
    - read_fvcom_mesh
    - read_mike_mesh
    - read_gmsh_mesh
    - write_sms_mesh
    - write_sms_bathy
    - write_mike_mesh
    - find_nearest_point
    - element_side_lengths
    - fix_coordinates
    - clip_triangulation
    - get_river_config
    - get_rivers
    - mesh2grid
    - line_sample
    - OSGB36_to_WGS84
    - connectivity
    - clip_domain
    - surrounders
    - get_area

* ll2utm - convert from spherical to cartesian UTM coordinates and back. Available from <http://robotics.ai.uiuc.edu/~hyoon24/LatLongUTMconversion.py>. 
    - LL_to_UTM
    - UTM_to_LL

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
    - pea
    - simpsonhunter
    - mixedlayerdepth
    - stokes
    - dissipation
    - calculate_rhum

* process_results - perform some analyses on FVCOM data read in using read_FVCOM_results.
    - calculate_total_CO2
    - calculate_CO2_leak_budget
    - data_average
    - unstructured_grid_volume
    - residual_flow

* read_results - parse the netCDF model output and extract a subset of the variables.
    - ncwrite
    - ncread
    - read_probes
    - write_probes
    - elems2nodes
    - nodes2elems

* stats_tools - some basic statistics tools.
    - calculate_regression
    - calculate_polyfit
    - coefficient_of_determination
    - fix_range
    - rmse

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
    - julian_day
    - gregorian_date
    - add_harmonic_results
    - get_observed_data
    - get_observed_metadata
    - clean_observed_data
    - parse_TAPPY_XML
    - get_harmonics
    - read_POLPRED
    - grid_POLPRED
    - get_harmonics_POLPRED


Examples
--------

The examples directory includes some Jupyter notebooks of some brief examples of how to use PyFVCOM. There are also sample scripts of those notebooks.

