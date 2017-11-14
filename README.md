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

* Python. This is mostly written to run in Python 3.5+, but may (should) work in 2.7.

* numpy

* scipy

* matplotlib

* netCDF4

* pyshp

* jdcal

* pyproj

* lxml

We recommend Jupyter (formerly iPython) for interactive use of PyFVCOM (and python generally).

Provides
--------

* buoy - read data from an SQLite3 database of BODC buoy data.
    - get_buoy_metadata
    - get_buoy_data

* coast - work with coastlines
    - read_ESRI_shapefile
    - read_arc_MIKE
    - read_CST
    - write_CST

* ctd - interrogate an SQLite data base of CTD casts.
    - get_CTD_metadata
    - get_CTD_data
    - get_ferrybox_data

* current - tools related to processing currents
    - Residuals
    - scalar2vector
    - vector2scalar
    - residual_flow
    - vorticity

* grid - tools to parse SMS, DHI MIKE, GMSH and FVCOM unstructured grids. Also provides functionality to add coasts and clip triangulations to a given domain. Functions to parse FVCOM river files are also included, as is a function to resample an unstructured grid onto a regular grid (without interpolation, simply finding the nearest point within a threshold distance). This module contains a number of generally useful tools related to unstructured grids (node and element lookups, grid connectivity, grid metrics, area tools).
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
    - connectivity
    - clip_domain
    - find_connected_nodes
    - find_connected_elements
    - get_area
    - find_bad_node
    - trigradient
    - rotate_points
    - get_boundary_polygons
    - get_attached_unique_nodes
    - grid_metrics
    - control_volumes
    - node_control_area
    - element_control_area
    - unstructured_grid_volume
    - elems2nodes
    - nodes2elems

* coordinate - convert from spherical and cartesian (UTM) coordinates. Also work with British National Grid coordinates and spherical.
    - utm_from_lonlat
    - lonlat_from_utm
    - british_national_grid_to_lonlat

* ocean - a number of routines to convert between combinations of temperature, salinity, pressure, depth and density.
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
    - dens_jackett
    - cond2salt
    - zbar
    - pea
    - simpsonhunter
    - mixedlayerdepth
    - stokes
    - dissipation
    - calculate_rhum

* plot - plotting class for FVCOM outputs.
    - Time.plot_line
    - Time.plot_scatter
    - Time.plot_quiver
    - Time.plot_surface
    - Plotter.plot_field
    - Plotter.plot_quiver
    - Plotter.plot_lines
    - Plotter.remove_line_plots
    - Plotter.plot_scatter

* read - parse the netCDF model output and extract a subset of the variables.
    - FileReader
    - MFileReader
    - ncwrite
    - ncread
    - read_probes
    - write_probes

* stats - some basic statistics tools.
    - calculate_regression
    - calculate_polyfit
    - rmse
    - calculate_coefficient

* tidal_ellipse - Python version of the Tidal Ellipse MATLAB toolbox <http://woodshole.er.usgs.gov/operations/sea-mat/tidal_ellipse-html/index.html>.
    - ap2ep
    - ep2ap
    - cBEpm
    - get_BE
    - sub2ind
    - plot_ell
    - do_the_plot
    - prep_plot

* tide - tools to use and abuse tidal data from an SQLite database of tidal time series.
    - add_harmonic_results
    - get_observed_data
    - get_observed_metadata
    - clean_observed_data
    - parse_TAPPY_XML
    - get_harmonics
    - read_POLPRED
    - grid_POLPRED
    - get_harmonics_POLPRED
    - make_water_column

* utilities - general utilities (including time utilities)
    - StubFile
    - fix_range
    - julian_day
    - gregorian_date
    - overlap
    - common_time
    - ind2sub


Examples
--------

The examples directory includes some Jupyter notebooks of some brief examples of how to use PyFVCOM. There are also sample scripts of those notebooks.

### Quick oneliners:

#### Grid tools
- Read SMS grid: `triangle, nodes, x, y, z, types, nodestrings = PyFVCOM.grid.read_sms_mesh('mesh.2dm', nodestrings=True)`
- Read FVCOM grid: `triangle, nodes, x, y, z = PyFVCOM.grid.read_fvcom_mesh('mesh.dat')`
- Find elements connected to node: `elements = PyFVCOM.grid.find_connected_elements(n, triangles)`
- Find nodes connected to node: `nodes = PyFVCOM.grid.find_connected_nodes(n, triangles)`
- Find model boundary from a grid: `coast = PyFVCOM.grid.get_boundary_polygons(triangles)`
- Calculate element areas: `area = PyFVCOM.grid.get_area(np.asarray((fvcom.grid.x[fvcom.grid.triangles[:, 0]], fvcom.grid.y[fvcom.grid.triangles[:, 0]])).T, np.asarray((fvcom.grid.x[fvcom.grid.triangles[:, 1]], fvcom.grid.y[fvcom.grid.triangles[:, 1]])).T, np.asarray((fvcom.grid.x[fvcom.grid.triangles[:, 2]], fvcom.grid.y[fvcom.grid.triangles[:, 2]])).T)`
- Calculate node control areas: `node_control_area = [PyFVCOM.grid.node_control_area(n) for n in len(fvcom.dims.node)]`
- Calculate element control areas: `element_control_area = [PyFVCOM.grid.element_control_area(e, fvcom.grid.triangles, area) for e in len(fvcom.dims.nele)]`
- Move a field from elements to nodes: `on_nodes = elems2nodes(fvcom.data.field, fvcom.grid.triangles)`
- Move a field from nodes to elements: `on_elements = nodes2elems(fvcom.data.field, fvcom.grid.triangles)`

#### Model data
- Read model output: `fvcom = PyFVCOM.read.FileReader('casename_0001.nc')`
- Calculate density from temperature and salinity: `density = PyFVCOM.ocean.dens_jackett(fvcom.data.temp, fvcom.data.salinity)`

#### Miscellaneous tools
- Make an array of datetime objects: `times = PyFVCOM.utilities.date_range(start, end, inc=0.5)`
