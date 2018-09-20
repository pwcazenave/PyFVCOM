Welcome to PyFVCOM!
--------------------

Table of contents
-----------------

- [Introduction](#introduction)
- [Installing](#installing)
- [Provides](#provides)
- [Examples](#examples)
- [Coding conventions](#coding-conventions)
- [Citation](#citation)

Introduction
------------

PyFVCOM is a collection of various tools and utilities which can be used to extract, analyse and plot input and output files from FVCOM.

Installing
----------

Easiest way is to install with pip/pip3:

```bash
pip install PyFVCOM
```

If you want to install the development version, checkout the `dev' branch and then from within the top-level directory:

```bash
pip install --user -e .
```

We are targeting Python 3.6+. PyFVCOM no longer supports Python 2.

We recommend Jupyter (formerly iPython) for interactive use of PyFVCOM (and python generally).

Provides
--------

* `buoy` - read data from an SQLite3 database of BODC buoy data.
    - `Buoy` - class to hold a range of time series data from buoys.
    - `get_buoy_metadata`
    - `get_buoy_data`

* `coast` - work with coastlines
    - `read_ESRI_shapefile`
    - `read_arc_MIKE`
    - `read_CST`
    - `write_CST`

* `coordinate` - convert from spherical and cartesian (UTM) coordinates. Also work with British National Grid coordinates and spherical.
    - `utm_from_lonlat`
    - `lonlat_from_utm`
    - `british_national_grid_to_lonlat`

* `ctd` - interrogate an SQLite data base of CTD casts.
    - `CTD` - class to hold a range of time series data from many different CTD formats we (PML) encounter.
    - `get_CTD_metadata`
    - `get_CTD_data`
    - `get_ferrybox_data`

* `current` - tools related to processing currents
    - `Residuals`
    - `scalar2vector`
    - `vector2scalar`
    - `residual_flow`
    - `vorticity`
    - `ebb_flood`
    - `principal_axis`

* `grid` - tools to parse SMS, DHI MIKE, GMSH and FVCOM unstructured grids. Also provides functionality to add coasts and clip triangulations to a given domain. Functions to parse FVCOM river files are also included, as is a function to resample an unstructured grid onto a regular grid (without interpolation, simply finding the nearest point within a threshold distance). This module contains a number of generally useful tools related to unstructured grids (node and element lookups, grid connectivity, grid metrics, area tools).
    - `Domain` - class to abstract loading different grid types away. The `read_*_mesh` methods below are now slighly redundant.
    - `Domain.closest_node`
    - `Domain.closest_element`
    - `Domain.horizontal_transect_nodes`
    - `Domain.horizontal_transect_elements`
    - `Domain.calculate_areas`
    - `OpenBoundary` - class to handle model open boundaries.
    - `OpenBoundary.add_sponge_layer`
    - `OpenBoundary.add_tpxo_tides`
    - `OpenBoundary.add_nested_forcing`
    - `read_sms_mesh`
    - `read_fvcom_mesh`
    - `read_mike_mesh`
    - `read_gmsh_mesh`
    - `write_sms_mesh`
    - `write_sms_bathy`
    - `write_mike_mesh`
    - `find_nearest_point`
    - `element_side_lengths`
    - `clip_triangulation`
    - `get_river_config`
    - `get_rivers`
    - `mesh2grid`
    - `line_sample`
    - `element_sample`
    - `connectivity`
    - `find_connected_nodes`
    - `find_connected_elements`
    - `get_area`
    - `find_bad_node`
    - `trigradient`
    - `rotate_points`
    - `get_boundary_polygons`
    - `get_attached_unique_nodes`
    - `grid_metrics`
    - `control_volumes`
    - `node_control_area`
    - `element_control_area`
    - `unstructured_grid_volume`
    - `unstructured_grid_depths`
    - `elems2nodes`
    - `nodes2elems`
    - `vincenty_distance`
    - `haversine_distance`
    - `shape_coefficients`
    - `reduce_triangulation`
    - `getcrossectiontriangles`
    - `isintriangle`

* `ocean` - a number of routines to convert between combinations of temperature, salinity, pressure, depth and density.
    - `pressure2depth`
    - `depth2pressure`
    - `dT_adiab_sw`
    - `theta_sw`
    - `cp_sw`
    - `sw_smow`
    - `sw_dens0`
    - `sw_seck`
    - `sw_dens`
    - `sw_svan`
    - `sw_sal78`
    - `dens_jackett`
    - `cond2salt`
    - `zbar`
    - `pea`
    - `simpsonhunter`
    - `mixedlayerdepth`
    - `stokes`
    - `dissipation`
    - `calculate_rhum`

* `plot` - plotting class for FVCOM outputs.
    - `Time.plot_line`
    - `Time.plot_scatter`
    - `Time.plot_quiver`
    - `Time.plot_surface`
    - `Plotter.plot_field`
    - `Plotter.plot_quiver`
    - `Plotter.plot_lines`
    - `Plotter.remove_line_plots`
    - `Plotter.plot_scatter`

* `preproc` - class for creating input files for FVCOM model runs.
    - `Model.write_grid`
    - `Model.write_coriolis`
    - `Model.add_bed_roughness`
    - `Model.write_bed_roughness`
    - `Model.interp_sst_assimilation`
    - `Model.write_sstgrd`
    - `Model.add_sigma_coordinates`
    - `Model.sigma_generalized`
    - `Model.sigma_geometric`
    - `Model.sigma_tanh`
    - `Model.hybrid_sigma_coordinate`
    - `Model.write_sigma`
    - `Model.add_open_boundaries`
    - `Model.write_sponge`
    - `Model.add_grid_metrics`
    - `Model.write_tides`
    - `Model.add_rivers`
    - `Model.check_rivers`
    - `Model.write_river_forcing`
    - `Model.write_river_namelist`
    - `Model.read_nemo_rivers`
    - `Model.add_probes`
    - `Model.write_probes`
    - `Model.read_regular`
    - `WriteForcing.add_variable`
    - `WriteForcing.write_fvcom_time`
    - `RegularReader` - like `PyFVCOM.read.FileReader`, but for regularly gridded data.
    - `read_regular` - load multiple regularly gridded files.
    - `HYCOMReader` - like `PyFVCOM.read.FileReader`, but for HYCOM data.
    - `read_hycom` - load multiple regularly gridded files.

* `read` - parse the netCDF model output and extract a subset of the variables.
    - `FileReader`
    - `MFileReader`
    - `FileReaderFromDict`
    - `ncwrite`
    - `ncread`
    - `read_probes`
    - `write_probes`

* `stats` - some basic statistics tools.
    - `calculate_regression`
    - `calculate_polyfit`
    - `rmse`
    - `calculate_coefficient`

* `tidal_ellipse` - Python version of the Tidal Ellipse MATLAB toolbox <http://woodshole.er.usgs.gov/operations/sea-mat/tidal_ellipse-html/index.html>.
    - `ap2ep`
    - `ep2ap`
    - `cBEpm`
    - `get_BE`
    - `sub2ind`
    - `plot_ell`
    - `do_the_plot`
    - `prep_plot`

* `tide` - tools to use and abuse tidal data from an SQLite database of tidal time series.
    - `HarmonicOutput`
    - `add_harmonic_results`
    - `get_observed_data`
    - `get_observed_metadata`
    - `clean_observed_data`
    - `parse_TAPPY_XML`
    - `get_harmonics`
    - `read_POLPRED`
    - `grid_POLPRED`
    - `get_harmonics_POLPRED`
    - `make_water_column`
    - `Lanczos` - Lanczos time filter.
    - `lanczos` - As above, but not a class.

* `utilities` - general utilities (including time utilities)
    - `general.fix_range`
    - `general.ind2sub`
    - `general.flatten_list`
    - `grid.StubFile`
    - `time.julian_day`
    - `time.gregorian_date`
    - `time.overlap`
    - `time.common_time`

* `validation` - post-processing and validation utilities. Some of these are currently incomplete.
    - `validation_db`
    - `validation_db.execute_sql`
    - `validation_db.create_table`
    - `validation_db.insert_into_table`
    - `validation_db.select_qry`
    - `validation_db.table_exists`
    - `validation_db.close_conn`
    - `dt_to_epochsec`
    - `epochsec_to_dt`
    - `plot_map`
    - `plot_tides`
    - `db_tide`
    - `db_tide.make_bodc_tables`
    - `db_tide.insert_tide_file`
    - `db_tide.get_tidal_series`
    - `db_tide.get_gauge_locations`
    - `db_tide.get_nearest_gauge_id`
    - `bodc_annual_tide_file`
    - `db_wco`
    - `db_wco.make_wco_tables`
    - `db_wco.insert_CTD_file`
    - `db_wco.insert_buoy_file`
    - `db_wco.insert_CTD_dir`
    - `db_wco.insert_csv_file`
    - `db_wco.get_observations`
    - `WCO_obs_file`
    - `csv_formatted`
    - `comp_data`
    - `comp_data.retrieve_file_data`
    - `comp_data.retrieve_obs_data`
    - `comp_data.get_comp_data_interpolated`
    - `comp_data.comp_data_nearest`
    - `comp_data.model_closest_time`
    - `comp_data_filereader`
    - `comp_data_filereader.retrieve_file_data`
    - `comp_data_filereader.model_closest_time`
    - `comp_data_probe`
    - `comp_data_probe.retrieve_file_data`
    - `ICES_comp`
    - `ICES_comp.get_var_comp`

Examples
--------

The examples directory includes some Jupyter notebooks of some brief examples of how to use PyFVCOM. There are also sample scripts of those notebooks.

### Quick oneliners:

#### Grid tools
- Read an SMS unstructured grid: `mesh = PyFVCOM.grid.Domain('mesh.2dm')`
- Read an FVCOM unstructured grid: `mesh = PyFVCOM.grid.Domain('mesh.dat')`
- Find elements connected to node: `elements = PyFVCOM.grid.find_connected_elements(n, mesh.grid.triangles)`
- Find nodes connected to node: `nodes = PyFVCOM.grid.find_connected_nodes(n, mesh.grid.triangles)`
- Find model boundary from a grid: `coast = PyFVCOM.grid.get_boundary_polygons(mesh.grid.triangles)`
- Calculate element areas: `mesh.calculate_areas()`
- Calculate node control areas: `node_control_area = [PyFVCOM.grid.node_control_area(n) for n in range(len(mesh.dims.node))]`
- Calculate element control areas: `element_control_area = [PyFVCOM.grid.element_control_area(e, mesh.grid.triangles, area) for e in range(len(fvcom.dims.nele))]`
- Move a field from elements to nodes: `on_nodes = elems2nodes(fvcom.data.field, mesh.grid.triangles)`
- Move a field from nodes to elements: `on_elements = nodes2elems(fvcom.data.field, mesh.grid.triangles)`

#### Model data
- Read model output: `fvcom = PyFVCOM.read.FileReader('casename_0001.nc', variables=['temp', 'salinity'])`
- Calculate density from temperature and salinity: `density = PyFVCOM.ocean.dens_jackett(fvcom.data.temp, fvcom.data.salinity)`

#### Miscellaneous tools
- Make an array of datetime objects: `times = PyFVCOM.utilities.date_range(start, end, inc=0.5)`

Coding conventions
------------------

- Use 4 spaces per indentation level
- Never mix tabs and spaces
- Imports should usually be on separate lines
- `from module import *` is not OK; rather, use `from module import name`
- Imports are always put at the top of the file
- Avoid extraneous whitespace
- Use parentheses sparingly
- Don't put an if/for/while with a small body on a single line
- Do not terminate your lines with semi-colons and do not use semi-colons to put two commands on the same line
- If a class inherits from no other base classes, explicitly inherit from object. This also applies to nested classes.
- Function names should be lowercase, underscore separated. Class names should be of the form `MyClass'.
- Names of members considered private shall start with two underscores
- Use lambda expressions only for one-liners (else: hard to read and to debug)
- Use properties for accessing or setting data where you would normally have used simple, lightweight getter or setter methods
- Use `with` when opening files or explicitly close files and sockets when done with them
- Use TODO comments for code that is temporary, a short-term solution, or good-enough but not perfect

Citation
--------

Cazenave, P. W. and Bedington, M. (2018). PyFVCOM (version x.x.x) [software]. Plymouth, Devon, United Kingdom: Plymouth Marine Laboratory. https://doi.org/10.5281/zenodo.1422462

