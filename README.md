Official repo
-------------

⚠️⚠️⚠️ The official repo for PyFVCOM is at:

https://github.com/pmlmodelling/pyfvcom

⚠️⚠️⚠️ This version is no longer updated regularly.

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

PyFVCOM is a collection of various tools and utilities which can be used to extract, analyse and plot input and output files from FVCOM as well as generate model inputs.

Installing
----------

Easiest way is to install with pip/pip3:

```bash
pip install PyFVCOM
```

If you want to install the development version, checkout the `dev' branch and then from within the top-level directory:

```bash
git clone git@gitlab.ecosystem-modelling.pml.ac.uk:fvcom/pyfvcom.git ./PyFVCOM
cd PyFVCOM
git checkout dev
pip install --user -e .
```

We are targeting Python 3.6+. PyFVCOM no longer supports Python 2.

We recommend Jupyter (formerly iPython) for interactive use of PyFVCOM (and python generally).

If you want to install PyFVCOM within a conda environment the suggested approach is [other than wait until the package is ported to Conda]:
```bash
conda create -n pyfvcom 
conda activate pyfvcom 
conda install pip
# in my case I had to manually install the requirements in conda before using pip to install PyFVCOM
conda install numpy
conda install scipy
conda install shapely
conda install cartopy
cd ~/myinstallationdirectory/pyfvcom 
pip install -e .

```


Provides
--------

* `buoy` - read data from an SQLite3 database of BODC buoy data.
    - `Buoy` - class to hold a range of time series data from buoys.
    - `get_buoy_metadata`
    - `get_buoy_data`

* `coordinate` - convert from spherical and cartesian (UTM) coordinates. Also work with British National Grid coordinates and spherical.
    - `utm_from_lonlat`
    - `lonlat_from_utm`
    - `british_national_grid_to_lonlat`
    - `lonlat_decimal_from_degminsec`
    - `lonlat_decimal_from_degminsec_wco`

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
    - `progressive_vectors`
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
    - `mp_interp_func`
    - `OpenBoundary` - class to handle model open boundaries.
    - `OpenBoundary.add_sponge_layer`
    - `OpenBoundary.add_tpxo_tides`
    - `OpenBoundary.add_nested_forcing`
    - `read_sms_mesh`
    - `read_fvcom_mesh`
    - `read_smesh_mesh`
    - `read_mike_mesh`
    - `read_gmsh_mesh`
    - `read_fvcom_obc`
    - `parse_obc_sections`
    - `read_sms_cst`
    - `write_sms_mesh`
    - `write_sms_bathy`
    - `write_mike_mesh`
    - `write_sms_cst`
    - `shp2cst`
    - `MIKEarc2cst`
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
    - `clockwise`
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
    - `subset_domain`
    - `model_exterior`
    - `fvcom2ugrid`
    - `point_in_pixel`
    - `node_to_centre`
    - `Graph` - class to hold an unstructured grid as a graph
    - `ReducedFVCOMdist` - class to query a grid graph for distance-based metrics
    - `GraphFVCOMdepth` - class to query a grid graph for depth-based metrics

* `interpolate` - a class to handle interpolation between unstructured and regular grids.
    - `mask_to_fvcom`
    - `mask_to_fvcom_meshgrid`
    - `MPIRegularInterpolateWorker`
    - `MPIRegularInterpolateWorker.InitialiseGrid`
    - `MPIRegularInterpolateWorker.InterpolateRegular`

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
    - `Depth` - for plotting vertical slices
    - `Depth.plot_slice`
    - `Time` - for plotting timer series of data
    - `Time.plot_line`
    - `Time.plot_scatter`
    - `Time.plot_quiver`
    - `Time.plot_surface`
    - `Plotter` - for plotting horizontal maps
    - `Plotter.plot_field`
    - `Plotter.plot_quiver`
    - `Plotter.plot_lines`
    - `Plotter.remove_line_plots`
    - `Plotter.plot_scatter`
    - `Plotter.plot_streamlines`
    - `CrossPlotter` - for plotting cross-sections
    - `CrossPlotter.cross_section_init`
    - `CrossPlotter.plot_pcolor_field`
    - `CrossPlotter.plot_quiver`
    - `MPIWorker` - for plotting in parallel with MPI
    - `MPIWorker.plot_field`
    - `MPIWorker.plot_streamlines`
    - `Player` - for interactive animation of horizontal maps
    - `plot_domain` - to quickly plot a FileReader.
    - `colourbar_extension`
    - `cm2inch`

* `preproc` - class for creating input files for FVCOM model runs.
    - `Model` - hold everything needed to generate new model inputs
    - `Model.write_grid`
    - `Model.write_coriolis`
    - `Model.add_bed_roughness`
    - `Model.write_bed_roughness`
    - `Model.interp_sst_assimilation`
    - `Model.write_sstgrd`
    - `Model.interp_ady`
    - `Model.interp_ady_climatology`
    - `Model.write_adygrd`
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
    - `Model.read_ea_river_temperature_climatology`
    - `Model.add_probes`
    - `Model.write_probes`
    - `Model.add_stations`
    - `Model.write_stations`
    - `Model.add_nests`
    - `Model.add_nests_harmonics`
    - `Model.add_nests_regular`
    - `Model.avg_nest_force_vel`
    - `Model.load_nested_forcing`
    - `Model.write_nested_forcing`
    - `Model.add_obc_types`
    - `Model.write_obc`
    - `Model.add_groundwater`
    - `Model.write_groundwater`
    - `Model.read_regular`
    - `Model.subset_existing_nest`
    - `Model.load_elevtide`
    - `Model.write_tsobc`
    - `NameListEntry` - class for holding entries in a NameList class
    - `NameListEntry.string`
    - `NameListEntry.tolist`
    - `NameList` - class for creating FVCOM model namelists
    - `NameList.index`
    - `NameList.value`
    - `NameList.update`
    - `NameList.update_nudging`
    - `NameList.update_nesting_interval`
    - `NameList.valid_nesting_timescale`
    - `NameList.update_ramp`
    - `NameList.write_model_namelist`
    - `write_model_namelist`
    - `Nest` - class for holding nested OpenBoudnary objects
    - `Nest.add_level`
    - `Nest.add_weights`
    - `Nest.add_tpxo_tides`
    - `Nest.add_nested_forcing`
    - `Nest.add_fvcom_tides`
    - `Nest.avg_nest_force_vel`
    - `WriteForcing` - actually a fairly generic class to write netCDFs with a concise syntax
    - `WriteForcing.add_variable`
    - `WriteForcing.write_fvcom_time`
    - `RegularReader` - like `PyFVCOM.read.FileReader`, but for regularly gridded data
    - `RegularReader.closest_element`
    - `RegularReader.closest_node`
    - `read_regular` - load multiple regularly gridded files
    - `HYCOMReader` - like `PyFVCOM.read.FileReader`, but for HYCOM data
    - `HYCOMReader.load_data`
    - `read_hycom` - load multiple regularly gridded files
    - `NEMOReader` - like `RegularReader`, but specifically for NEMO outputs
    - `NEMOReader.load`
    - `NemoRestartRegularReader`
    - `Regular2DReader`
    - `Restart` - class to interact/modify FVCOM restart files
    - `Restart.replace_variable`
    - `Restart.replace_variable_with_regular`
    - `Restart.write_restart`
    - `Restart.read_regular`

* `read` - parse the netCDF model output and extract a subset of the variables.
    - `FileReader` - read in FVCOM outputs
    - `FileReader.add`
    - `FileReader.subtract`
    - `FileReader.multiply`
    - `FileReader.divide`
    - `FileReader.power`
    - `FileReader.load_data`
    - `FileReader.closest_time`
    - `FileReader.grid_volume`
    - `FileReader.total_volume_var`
    - `FileReader.avg_volume_var`
    - `FileReader.time_to_index`
    - `FileReader.time_average`
    - `FileReader.add_river_flow`
    - `FileReader.to_excel`
    - `FileReader.to_csv`
    - `read_nesting_nodes`
    - `apply_mask`
    - `MFileReader` - read in multiple FVCOM outputs
    - `SubDomainReader` - subset a model domain in space
    - `SubDomainReader.add_evap_precip`
    - `SubDomainReader.add_river_data`
    - `SubDomainReader.aopen_integral`
    - `SubDomainReader.volume_integral`
    - `SubDomainReader.surface_integral`
    - `time_to_index`
    - `FileReaderFromDict` - have a go at converting from `ncread` output to `FileReader` format
    - `ncwrite`
    - `ncread` - read netCDF data to a dictionary
    - `read_probes`
    - `write_probes`
    - `WriteFVCOM` - write a FileReader object to a netCDF file in FVCOM format

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
    - `general.PassiveStore` - our template class for lots of other classes
    - `general.fix_range`
    - `general.ind2sub`
    - `general.flatten_list`
    - `general.split_string`
    - `general.ObjectFromDict`
    - `general.clean_html`
    - `general.cart2pol`
    - `general.pol2cart`
    - `time.julian_day`
    - `time.gregorian_date`
    - `time.overlap`
    - `time.common_time`
    - `time.make_signal`
    - `time.ramped_signal`

* `validation` - post-processing and validation utilities. Some of these are currently incomplete.
    - `ValidationDB`
    - `ValidationDB.execute_sql`
    - `ValidationDB.create_table`
    - `ValidationDB.insert_into_table`
    - `ValidationDB.select_qry`
    - `ValidationDB.table_exists`
    - `ValidationDB.close_conn`
    - `dt_to_epochsec`
    - `epochsec_to_dt`
    - `plot_map`
    - `plot_tides`
    - `TideDB`
    - `TideDB.make_bodc_tables`
    - `TideDB.insert_tide_file`
    - `TideDB.get_tidal_series`
    - `TideDB.get_gauge_locations`
    - `TideDB.get_nearest_gauge_id`
    - `BODCAnnualTideFile`
    - `WCODB`
    - `WCODB.make_wco_tables`
    - `WCODB.insert_CTD_file`
    - `WCODB.insert_buoy_file`
    - `WCODB.insert_CTD_dir`
    - `WCODB.insert_csv_file`
    - `WCODB.get_observations`
    - `WCOParseFile`
    - `CSVFormatter`
    - `CompareData`
    - `CompareData.retrieve_file_data`
    - `CompareData.retrieve_obs_data`
    - `CompareData.get_comp_data_interpolated`
    - `CompareData.comp_data_nearest`
    - `CompareData.model_closest_time`
    - `CompareDataFileReader`
    - `CompareDataFileReader.retrieve_file_data`
    - `CompareDataFileReader.model_closest_time`
    - `CompareDataProbe`
    - `CompareDataProbe.retrieve_file_data`
    - `CompareICES`
    - `CompareICES.get_var_comp`

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
- Calculate node control areas: `node_control_area = [PyFVCOM.grid.node_control_area(n) for n in range(mesh.dims.node)]`
- Calculate element control areas: `element_control_area = [PyFVCOM.grid.element_control_area(e, mesh.grid.triangles, area) for e in range(fvcom.dims.nele)]`
- Move a field from elements to nodes: `on_nodes = elems2nodes(fvcom.data.field, mesh.grid.triangles)`
- Move a field from nodes to elements: `on_elements = nodes2elems(fvcom.data.field, mesh.grid.triangles)`

#### Model data
- Read model output: `fvcom = PyFVCOM.read.FileReader('casename_0001.nc', variables=['temp', 'salinity'])`
- Calculate density from temperature and salinity: `density = PyFVCOM.ocean.dens_jackett(fvcom.data.temp, fvcom.data.salinity)`

#### Miscellaneous tools
- Make an array of datetime objects: `times = PyFVCOM.utilities.time.date_range(start, end, inc=0.5)`

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
- Verbose output should be off by default

Citation
--------

Cazenave, P. W. et al. (2018). PyFVCOM (version x.x.x) [software]. Plymouth, Devon, United Kingdom: Plymouth Marine Laboratory. https://doi.org/10.5281/zenodo.1422462

