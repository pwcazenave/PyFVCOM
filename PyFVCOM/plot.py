""" Plotting class for FVCOM results. """

from __future__ import print_function

import copy
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.widgets
import mpl_toolkits.axes_grid1
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from descartes import PolygonPatch
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter, date2num
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon, Point, LineString

from PyFVCOM.coordinate import lonlat_from_utm, utm_from_lonlat
from PyFVCOM.current import vector2scalar
from PyFVCOM.grid import get_boundary_polygons
from PyFVCOM.grid import getcrossectiontriangles, unstructured_grid_depths, Domain, nodes2elems, mp_interp_func
from PyFVCOM.ocean import depth2pressure, dens_jackett
from PyFVCOM.read import FileReader
from PyFVCOM.utilities.general import PassiveStore, warn

have_basemap = True
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    warn('No mpl_toolkits found in this python installation. Some functions will be disabled.')
    Basemap = None
    have_basemap = False

rcParams['mathtext.default'] = 'regular'  # use non-LaTeX fonts


class Depth(object):
    """ Create depth-resolved plots based on output from FVCOM.

    Provides
    --------
    plot_slice

    Author(s)
    ---------
    Pierre Cazenave (Plymouth Marine Laboratory)

    """

    def __init__(self, dataset, figure=None, figsize=(20, 8), axes=None, cmap='viridis', title=None, legend=False,
                 fs=10, date_format=None, cb_label=None, hold=False):
        """
        Parameters
        ----------
        dataset : Dataset, PyFVCOM.read.FileReader
            netCDF4 Dataset or PyFVCOM.read.FileReader object.
        figure : Figure, optional
            Matplotlib Figure object. A figure object is created if not
            provided.
        figsize : tuple(float), optional
            Figure size in cm. This is only used if a new Figure object is
            created.
        axes : Axes, optional
            Matplotlib axes object. An Axes object is created if not
            provided.
        cmap : None, Colormap
            Provide a colourmap to use when plotting vectors or 2D plots (anything with a magnitude). Defaults to
            'viridis'.
        title : str, optional
            Title to use when creating the plot.
        fs : int, optional
            Font size to use when rendering plot text.
        legend : bool, optional
            Set to True to add a legend. Defaults to False.
        date_format : str
            Date format to use.
        cb_label : str
            Label to apply to the colour bar. Defaults to no label.
        hold : bool, optional
            Set to True to keep existing plots when adding to an existing figure. Defaults to False.

        """
        self.ds = dataset
        self.figure = figure
        self.axes = axes
        self.fs = fs
        self.title = title
        self.figsize = figsize
        self.hold = hold
        self.add_legend = legend
        self.cmap = cmap
        self.date_format = date_format
        self.cb_label = cb_label

        # Plot instances (initialise to None for truthiness test later)
        self.slice_plot = None

        # Are we working with a FileReader object or a bog-standard netCDF4 Dataset?
        self._FileReader = False
        if isinstance(dataset, (FileReader, Domain)):
            self._FileReader = True

        # Initialise the figure
        self.__init_figure()

    def __init_figure(self):
        # Initialise the figure
        if self.figure is None:
            figsize = (cm2inch(self.figsize[0]), cm2inch(self.figsize[1]))
            self.figure = plt.figure(figsize=figsize)

        # Create plot axes
        if not self.axes:
            self.axes = self.figure.add_subplot(1, 1, 1)

        if self.title:
            self.axes.set_title(self.title)

    def plot_slice(self, horizontal, depth, variable, fill_seabed=False, *args, **kwargs):
        """

        Parameters
        ----------
        horizontal : np.ndarray
            The horizontal array (x-axis). This can be distance along the slice or a coordinate.
        depth : np.ndarray
            The vertical depth array (positive-down).
        variable : np.ndarray
            The variable to plot in the vertical. Its shape must be compatible with `horizontal' and `depth'.
        fill_seabed : bool, optional
            Set to True to fill the seabed from the maximum water depth to the edge of the plot with gray.

        Remaining args and kwargs are passed to self.axes.pcolormesh.

        """

        # I'm not much of a fan of all this transposing. It feels like it's going to be a pain to debug when it
        # inevitably does something you don't expect.
        try:
            self.slice_plot = self.axes.pcolormesh(horizontal, -depth, variable,
                                                   cmap=self.cmap, *args, **kwargs)
        except TypeError:
            # Try flipping the data array, that might make it work.
            self.slice_plot = self.axes.pcolormesh(horizontal, -depth, variable.T,
                                                   cmap=self.cmap, *args, **kwargs)

        if fill_seabed:
            self.axes.fill_between(horizontal, self.axes.get_ylim()[0], -np.max(depth, axis=0), color='0.6')

        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        self.colorbar = self.figure.colorbar(self.slice_plot, cax=cax)
        self.colorbar.ax.tick_params(labelsize=self.fs)
        if self.cb_label:
            self.colorbar.set_label(self.cb_label)


class Time(object):
    """ Create time series plots based on output from FVCOM.

    Provides
    --------
    plot_line
    plot_scatter
    plot_quiver
    plot_surface

    Author(s)
    ---------
    Pierre Cazenave (Plymouth Marine Laboratory)

    """

    def __init__(self, dataset, figure=None, figsize=(20, 8), axes=None, cmap='viridis', title=None, legend=False,
                 fs=10, date_format=None, cb_label=None, hold=False, extend='neither'):
        """
        Parameters
        ----------
        dataset : Dataset, PyFVCOM.read.FileReader
            netCDF4 Dataset or PyFVCOM.read.FileReader object.
        figure : Figure, optional
            Matplotlib Figure object. A figure object is created if not
            provided.
        figsize : tuple(float), optional
            Figure size in cm. This is only used if a new Figure object is
            created.
        axes : Axes, optional
            Matplotlib axes object. An Axes object is created if not
            provided.
        cmap : None, Colormap
            Provide a colourmap to use when plotting vectors or 2D plots (anything with a magnitude). Defaults to
            'viridis'.
        title : str, optional
            Title to use when creating the plot.
        fs : int, optional
            Font size to use when rendering plot text.
        legend : bool, optional
            Set to True to add a legend. Defaults to False.
        date_format : str
            Date format to use.
        cb_label : str
            Label to apply to the colour bar. Defaults to no label.
        hold : bool, optional
            Set to True to keep existing plots when adding to an existing figure. Defaults to False.
        extend : str, optional
            Set the colour bar extension ('neither', 'both', 'min', 'max').
            Defaults to 'neither').

        """
        self.ds = dataset
        self.figure = figure
        self.axes = axes
        self.fs = fs
        self.title = title
        self.figsize = figsize
        self.hold = hold
        self.add_legend = legend
        self.cmap = cmap
        self.date_format = date_format
        self.cb_label = cb_label
        self.extend = extend

        # Plot instances (initialise to None for truthiness test later)
        self.line_plot = None
        self.scatter_plot = None
        self.quiver_plot = None  # for vectors with time (e.g. currents at a point)
        self.surface_plot = None  # for depth-resolved time, for example.
        self.legend = None
        self.colorbar = None
        self.quiver_key = None

        # Are we working with a FileReader object or a bog-standard netCDF4 Dataset?
        self._FileReader = False
        if isinstance(dataset, (FileReader, Domain)):
            self._FileReader = True

        # Initialise the figure
        self.__init_figure()

    def __init_figure(self):
        # Read in required grid variables
        if self._FileReader:
            self.time = self.ds.time.datetime
        else:
            # Try a couple of time formats.
            try:
                self.time = np.asarray([datetime.strftime('%Y-%m-%dT%H:%M:%S.%f', i) for i in self.ds.variables['Times']])
            except ValueError:
                self.time = np.asarray([datetime.strftime('%Y/%m/%d %H:%M:%S.%f', i) for i in self.ds.variables['Times']])
        self.n_times = len(self.time)

        # Initialise the figure
        if self.figure is None:
            figsize = (cm2inch(self.figsize[0]), cm2inch(self.figsize[1]))
            self.figure = plt.figure(figsize=figsize)

        # Create plot axes
        if not self.axes:
            self.axes = self.figure.add_subplot(1, 1, 1)

        if self.title:
            self.axes.set_title(self.title)

    def plot_line(self, time_series, *args, **kwargs):
        """
        Plot a time series as a line.

        Parameters
        ----------
        time_series : list-like, np.ndarray
            Time series data to plot.

        Additional kwargs are passed to `matplotlib.pyplot.plot'.

        """

        if self.line_plot and not self.hold:
            # Update the current line.
            self.line_plot.set_ydata = time_series
            self.line_plot.set_xdata = self.time
            return

        self.line_plot, = self.axes.plot(self.time, time_series,
                                         *args, **kwargs)

        if self.add_legend:
            self.legend = self.axes.legend(frameon=False)

    def plot_scatter(self, time_series, **kwargs):
        """
        Plot a time series as a set of scatter points.

        Parameters
        ----------
        time_series : list-like, np.ndarray
            Time series data to plot.

        Additional kwargs are passed to `matplotlib.pyplot.scatter'.

        """

        if self.scatter_plot and not self.hold:
            # Update the current scatter. I can't see how to replace both the x, y and colour data (I think set_array
            # does the latter), so just clear the axis and start again.
            self.axes.cla()

        self.scatter_plot = self.axes.scatter(self.time, time_series,
                                              **kwargs)
        if self.add_legend:
            self.legend = self.axes.legend(frameon=False)

    def plot_quiver(self, u, v, field=None, scale=1, **kwargs):
        """
        Plot a time series of vectors.

        Parameters
        ----------
        u, v : list-like, np.ndarray
            Arrays of time-varying vector components.
        field : list-like, np.ndarray, str, optional
            Field by which to colour the vectors. If set to 'magnitude', use the magnitude of the velocity vectors.
            Defaults to colouring by `color'.
        scale : float, optional
            Scale to pass to the quiver. See `matplotlib.pyplot.quiver' for information.

        Additional kwargs are passed to `matplotlib.pyplot.quiver'.

        Notes
        -----

        The `hold' option to PyFVCOM.plot.Time has no effect here: an existing plot is cleared before adding new data.

        """

        # To plot time along the x-axis with quiver, we need to use numerical representations of time. So,
        # convert from datetimes to numbers and then format the x-axis labels after the fact.
        quiver_time = date2num(self.time)

        if field == 'magnitude':
            field = np.hypot(u, v)

        if self.quiver_plot:
            if np.any(field):
                self.quiver_plot.set_UVC(u, v, field)
            else:
                self.quiver_plot.set_UVC(u, v)
            return

        if np.any(field):
            self.quiver_plot = self.axes.quiver(quiver_time, np.zeros(u.shape), u, v, field,
                                                cmap=self.cmap,
                                                units='inches',
                                                scale_units='inches',
                                                scale=scale,
                                                **kwargs)
            divider = make_axes_locatable(self.axes)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            self.colorbar = self.figure.colorbar(self.quiver_plot, cax=cax)
            self.colorbar.ax.tick_params(labelsize=self.fs)
            if self.cb_label:
                self.colorbar.set_label(self.cb_label)

        else:
            self.quiver_plot = self.axes.quiver(quiver_time, np.zeros(u.shape), u, v,
                                                units='inches',
                                                scale_units='inches',
                                                scale=scale,
                                                **kwargs)

        # Something approaching dynamic labelling of dates.
        if not self.date_format:
            x_range = self.quiver_plot.axes.get_xlim()
            x_delta = x_range[1] - x_range[0]
            if x_delta > int(1.5 * 365):
                date_format = DateFormatter('%Y-%m-%d')
            elif x_delta > 2:
                date_format = DateFormatter('%Y-%m-%d %H:%M')
            elif x_delta < 2:
                date_format = DateFormatter('%H:%M:%S')
            else:
                date_format = DateFormatter('%H:%M')
            self.axes.xaxis.set_major_formatter(date_format)
        else:
            self.axes.xaxis.set_major_formatter(self.date_format)

        if self.add_legend:
            label = f'{scale} $\mathrm{{ms^{-1}}}$'
            self.quiver_key = plt.quiverkey(self.quiver_plot, 0.9, 0.9, scale, label, coordinates='axes')

        # Turn off the y-axis labels as they don't correspond to the vector lengths.
        self.axes.get_yaxis().set_visible(False)

    def plot_surface(self, depth, time_series, fill_seabed=False, **kwargs):
        """
        Parameters
        ----------
        depth : np.ndarray
            Depth-varying array of depth. See `PyFVCOM.tide.make_water_column' for more information.
        time_series : np.ndarray
            Depth-varying array of data to plot.
        fill_seabed : bool, optional
            Set to True to fill the seabed from the maximum water depth to the edge of the plot with gray.

        Remaining kwargs are passed to self.axes.pcolormesh.

        """

        # Squeeze out singleton dimensions first.
        depth = np.squeeze(depth)
        time_series = np.squeeze(time_series)

        if not self.surface_plot:
            self.surface_plot = self.axes.pcolormesh(np.tile(self.time, [depth.shape[-1], 1]).T,
                                                     depth,
                                                     time_series,
                                                     cmap=self.cmap,
                                                     **kwargs)

            if fill_seabed:
                self.axes.fill_between(self.time, np.min(depth, axis=1), self.axes.get_ylim()[0], color='0.6')
            divider = make_axes_locatable(self.axes)

            cax = divider.append_axes("right", size="3%", pad=0.1)
            self.colorbar = self.figure.colorbar(self.surface_plot, cax=cax, extend=self.extend)
            self.colorbar.ax.tick_params(labelsize=self.fs)
            if self.cb_label:
                self.colorbar.set_label(self.cb_label)
        else:
            # Update the existing plot with the new data (currently untested!)
            self.surface_plot.set_array(time_series)


class Plotter(object):
    """ Create plot objects based on output from the FVCOM.

    Class to assist in the creation of plots and animations based on output
    from the FVCOM.

    Methods
    -------
    plot_field
    plot_quiver
    plot_lines
    plot_scatter
    plot_streamlines
    add_scale
    set_title
    replot
    close

    Author(s)
    ---------
    James Clark (Plymouth Marine Laboratory)
    Pierre Cazenave (Plymouth Marine Laboratory)
    Mike Bedington (Plymouth Marine Laboratory)

    """

    def __init__(self, dataset, figure=None, axes=None, stations=None, extents=None, vmin=None, vmax=None, mask=None,
                 res='c', fs=10, title=None, cmap='viridis', figsize=(10., 10.), axis_position=None, tick_inc=None,
                 cb_label=None, extend='neither', norm=None, m=None, cartesian=False, mapper='basemap', **bmargs):
        """
        Parameters
        ----------
        dataset : Dataset, PyFVCOM.read.FileReader
            netCDF4 Dataset or PyFVCOM.read.FileReader object.
        stations : 2D array, optional
            List of station coordinates to be plotted ([[lons], [lats]])
        extents : 1D array, optional
            Four element numpy array giving lon/lat limits as west, east, south, north (e.g. [-4.56, -3.76, 49.96,
            50.44])
        vmin : float, optional
            Lower bound to be used on colour bar (plot_field only).
        vmax : float, optional
            Upper bound to be used colour bar (plot_field only).
        mask : float, optional
            Mask out values < mask (plot_field only).
        res : string, optional
            Resolution to use when drawing Basemap object. If None, no coastline is plotted.
        fs : int, optional
            Font size to use when rendering plot text
        title : str, optional
            Title to use when creating the plot
        cmap : string, optional
            Colormap to use when shading field data (plot_field only).
        figure : Figure, optional
            Matplotlib figure object. A figure object is created if not
            provided.
        figsize : tuple(float), optional
            Figure size in cm. This is only used if a new Figure object is
            created.
        axes : Axes, optional
            Matplotlib Axes object. An Axes object is created if not
            provided.
        axis_position : 1D array, optional
            Array giving axis dimensions
        tick_inc : list, optional
            Add coordinate axes (i.e. lat/long) at the intervals specified in
            the list ([lon_spacing, lat_spacing]).
        cb_label : str, optional
            Set the colour bar label.
        extend : str, optional
            Set the colour bar extension ('neither', 'both', 'min', 'max').
            Defaults to 'neither').
        norm : matplotlib.colors.Normalize, optional
            Normalise the luminance to 0, 1. For example, use from matplotlib.colors.LogNorm to do log plots of fields.
        m : mpl_toolkits.basemap.Basemap, optional
            Pass a Basemap object rather than creating one on each invocation.
        cartesian : bool, optional
            Set to True to skip using Basemap and instead return a simple cartesian axis plot. Defaults to False
            (geographical coordinates).
        mapper : string, optional
            Set to 'basemap' to use Basemap for plotting or 'cartopy' for cartopy.
        bmargs : dict, optional
            Additional arguments to pass to Basemap.

        Author(s)
        ---------
        James Clark (Plymouth Marine Laboratory)
        Pierre Cazenave (Plymouth Marine Laboratory)
        Mike Bedington (Plymouth Marine Laboratory)

        """

        self._debug = False

        self.ds = dataset
        self.figure = figure
        self.axes = axes
        self.stations = stations
        self.extents = extents
        self.vmin = vmin
        self.vmax = vmax
        self.mask = mask
        self.res = res
        self.fs = fs
        self.title = title
        self.cmap = cmap
        self.figsize = figsize
        self.axis_position = axis_position
        self.tick_inc = tick_inc
        self.cb_label = cb_label
        self.extend = extend
        self.norm = norm
        self.m = m
        self.cartesian = cartesian
        self.bmargs = bmargs
        self.mapper = mapper

        # Plot instances to hold the plot objects.
        self.quiver_plot = None
        self.quiver_key = None
        self.scatter_plot = None
        self.tripcolor_plot = None
        self.line_plot = None
        self.streamline_plot = None
        self.tri = None
        self.masked_tris = None
        self.colorbar_axis = None
        self.cbar = None

        self.projection = None
        self._plot_projection = {}
        # For cartopy, we need to have a Plate Carree transform defined for doing the actual plotting of data since
        # we're using Lambert for the "display" projection.
        if self.mapper == 'cartopy':
            self._plot_projection = {'transform': ccrs.PlateCarree()}

        # Are we working with a FileReader object or a bog-standard netCDF4 Dataset?
        self._FileReader = False
        if isinstance(dataset, (FileReader, Domain)):
            self._FileReader = True

        # Initialise the figure
        self._init_figure()

    def _init_figure(self):
        # Read in required grid variables
        if self._FileReader:
            self.n_nodes = getattr(self.ds.dims, 'node')
            self.n_elems = getattr(self.ds.dims, 'nele')
            self.lon = getattr(self.ds.grid, 'lon')
            self.lat = getattr(self.ds.grid, 'lat')
            self.lonc = getattr(self.ds.grid, 'lonc')
            self.latc = getattr(self.ds.grid, 'latc')
            self.x = getattr(self.ds.grid, 'x')
            self.y = getattr(self.ds.grid, 'y')
            self.xc = getattr(self.ds.grid, 'xc')
            self.yc = getattr(self.ds.grid, 'yc')
            self.nv = getattr(self.ds.grid, 'nv')
        else:
            self.n_nodes = len(self.ds.dimensions['node'])
            self.n_elems = len(self.ds.dimensions['nele'])
            self.lon = self.ds.variables['lon'][:]
            self.lat = self.ds.variables['lat'][:]
            self.lonc = self.ds.variables['lonc'][:]
            self.latc = self.ds.variables['latc'][:]
            self.x = self.ds.variables['x'][:]
            self.y = self.ds.variables['y'][:]
            self.xc = self.ds.variables['xc'][:]
            self.yc = self.ds.variables['yc'][:]
            self.nv = self.ds.variables['nv'][:]

        if self.nv.min() != 1:
            if self.nv.min() > 0:
                self.nv -= self.nv.min()
            else:
                self.nv += 1 - self.nv.min()

        # Triangles
        self.triangles = self.nv.transpose() - 1

        # Initialise the figure
        if self.figure is None:
            figsize = (cm2inch(self.figsize[0]), cm2inch(self.figsize[1]))
            self.figure = plt.figure(figsize=figsize)
            self.figure.set_facecolor('white')

        # If plot extents were not given, use min/max lat/lon values
        if self.extents is None:
            if self.cartesian:
                self.extents = np.array([self.x.min(), self.x.max(),
                                         self.y.min(), self.y.max()])
            else:
                self.extents = np.array([self.lon.min(), self.lon.max(),
                                         self.lat.min(), self.lat.max()])

        # Create mapping object if appropriate.
        if not self.cartesian:
            if self.mapper == 'basemap':
                if have_basemap:
                    if self.m is None:
                        self.m = Basemap(llcrnrlon=np.min(self.extents[:2]),
                                         llcrnrlat=np.min(self.extents[-2:]),
                                         urcrnrlon=np.max(self.extents[:2]),
                                         urcrnrlat=np.max(self.extents[-2:]),
                                         rsphere=(6378137.00, 6356752.3142),
                                         resolution=self.res,
                                         projection='merc',
                                         lat_0=np.mean(self.extents[-2:]),
                                         lon_0=np.mean(self.extents[:2]),
                                         lat_ts=np.mean(self.extents[-2:]),
                                         ax=self.axes,
                                         **self.bmargs)
                    # Make a set of coordinates.
                    self.mx, self.my = self.m(self.lon, self.lat)
                    self.mxc, self.myc = self.m(self.lonc, self.latc)
                else:
                    raise RuntimeError('mpl_toolkits is not available in this Python.')
            elif self.mapper == 'cartopy':
                self.projection = ccrs.LambertConformal(central_longitude=np.mean(self.extents[:2]),
                                                        central_latitude=np.mean(self.extents[2:]),
                                                        false_easting=400000, false_northing=400000)

                # Make a coastline depending on whether we've got a GSHHS resolution or a Natural Earth one.
                if self.res in ('c', 'l', 'i', 'h', 'f'):
                    # Use the GSHHS data as in Basemap (a lot slower than the cartopy data).
                    land = cfeature.GSHHSFeature(scale=self.res, edgecolor='k', facecolor=0.6)
                else:
                    # Make a land object which is fairly similar to the Basemap on we use.
                    land = cfeature.NaturalEarthFeature('physical', 'land', self.res, edgecolor='k', facecolor='0.6')

                # Make a set of coordinates.
                self.mx, self.my = self.lon, self.lat
                self.mxc, self.myc = self.lonc, self.latc
            else:
                raise ValueError(f"Unrecognised mapper value '{self.mapper}'. Choose 'basemap' (default) or 'cartopy'")
        else:
            # Easy peasy, just the cartesian coordinates.
            self.mx, self.my = self.x, self.y
            self.mxc, self.myc = self.xc, self.yc

        # Create plot axes
        if not self.axes:
            self.axes = self.figure.add_subplot(1, 1, 1, projection=self.projection)
            if self.axis_position:
                self.axes.set_position(self.axis_position)

        if self.mapper == 'cartopy':
            self.axes.set_extent(self.extents, crs=ccrs.PlateCarree())
            self.axes.add_feature(land, zorder=1000)
            # *Must* call show and draw in order to get the axis boundary used to add ticks:
            self.figure.show()
            self.figure.canvas.draw()
        elif self.mapper == 'basemap' and not self.cartesian:
            self.m.drawmapboundary()
            self.m.drawcoastlines(zorder=1000)
            self.m.fillcontinents(color='0.6', zorder=1000)

        if self.title:
            self.axes.set_title(self.title)

        # Check the values of tick_inc aren't bigger than the extents.
        if self.tick_inc is not None:
            if self.tick_inc[0] > self.extents[1] - self.extents[0]:
                warn('The x-axis tick interval is larger than the plot x-axis extent.')
            if self.tick_inc[1] > self.extents[3] - self.extents[2]:
                warn('The y-axis tick interval is larger than the plot y-axis extent.')

        # Add coordinate labels to the x and y axes.
        if self.tick_inc is not None:
            meridians = np.arange(np.floor(np.min(self.extents[:2])), np.ceil(np.max(self.extents[:2])), self.tick_inc[0])
            parallels = np.arange(np.floor(np.min(self.extents[2:])), np.ceil(np.max(self.extents[2:])), self.tick_inc[1])
            if self.cartesian:
                # Cartesian
                self.axes.set_xticks(np.arange(self.extents[0], self.extents[1] + self.tick_inc[0], self.tick_inc[0]))
                self.axes.set_yticks(np.arange(self.extents[2], self.extents[3] + self.tick_inc[1], self.tick_inc[1]))
            elif self.mapper == 'basemap':
                self.m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=self.fs, linewidth=0, ax=self.axes)
                self.m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=self.fs, linewidth=0, ax=self.axes)
            elif self.mapper == 'cartopy':
                self.axes.gridlines(xlocs=meridians, ylocs=parallels, linewidth=0)
                # Label the end-points of the gridlines using the custom tick makers.
                self.axes.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
                self.axes.yaxis.set_major_formatter(LATITUDE_FORMATTER)
                self._lambert_xticks(meridians)
                self._lambert_yticks(parallels)

    # Whole bunch of hackery to get cartopy to label Lambert plots. Shamelessly copied from:
    #   https://nbviewer.jupyter.org/gist/ajdawson/dd536f786741e987ae4e
    @staticmethod
    def _find_side(ls, side):
        """
        Given a shapely LineString which is assumed to be rectangular, return the
        line corresponding to a given side of the rectangle.

        """
        minx, miny, maxx, maxy = ls.bounds
        points = {'left': [(minx, miny), (minx, maxy)],
                  'right': [(maxx, miny), (maxx, maxy)],
                  'bottom': [(minx, miny), (maxx, miny)],
                  'top': [(minx, maxy), (maxx, maxy)], }
        return LineString(points[side])

    def _lambert_xticks(self, ticks):
        """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
        te = lambda xy: xy[0]
        lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
        xticks, xticklabels = self._lambert_ticks(ticks, 'bottom', lc, te)
        self.axes.xaxis.tick_bottom()
        self.axes.set_xticks(xticks)
        self.axes.set_xticklabels([self.axes.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])

    def _lambert_yticks(self, ticks):
        """Draw ricks on the left y-axis of a Lambert Conformal projection."""
        te = lambda xy: xy[1]
        lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
        yticks, yticklabels = self._lambert_ticks(ticks, 'left', lc, te)
        self.axes.yaxis.tick_left()
        self.axes.set_yticks(yticks)
        self.axes.set_yticklabels([self.axes.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])

    def _lambert_ticks(self, ticks, tick_location, line_constructor, tick_extractor):
        """Get the tick locations and labels for an axis of a Lambert Conformal projection."""
        outline_patch = LineString(self.axes.outline_patch.get_path().vertices.tolist())
        axis = self._find_side(outline_patch, tick_location)
        n_steps = 30
        extent = self.axes.get_extent(ccrs.PlateCarree())
        _ticks = []
        for t in ticks:
            xy = line_constructor(t, n_steps, extent)
            proj_xyz = self.axes.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
            xyt = proj_xyz[..., :2]
            ls = LineString(xyt.tolist())
            locs = axis.intersection(ls)
            if not locs:
                tick = [None]
            else:
                tick = tick_extractor(locs.xy)
            _ticks.append(tick[0])
        # Remove ticks that aren't visible:
        ticklabels = copy.copy(ticks).tolist()
        while True:
            try:
                index = _ticks.index(None)
            except ValueError:
                break
            _ticks.pop(index)
            ticklabels.pop(index)
        return _ticks, ticklabels

    def get_colourbar_extension(self, field, clims):
        """
        Find the colourbar extension for the current variable, clipping in space if necessary.

        Parameters
        ----------
        field : np.ndarray
            The data being plotted.
        clims : list, tuple
            The colour limits of the plot.

        Returns
        -------
        extend : str
            The colourbar extension ('neither', 'min', 'max' or 'both').

        """

        # We need to find the nodes/elements for the current variable to make sure our colour bar extends for
        # what we're plotting (not the entire data set). We'll have to guess based on shape here.
        x = self.lon
        y = self.lat
        if self.n_elems == field.shape[0]:
            x = self.lonc
            y = self.latc
        mask = (x > self.extents[0]) & (x < self.extents[1]) & (y < self.extents[3]) & (y > self.extents[2])

        if all(clims) is None:
            clims = [field[..., mask].min(), field[..., mask].max()]
        if clims[0] is None:
            clims[0] = field[..., mask].min()
        if clims[1] is None:
            clims[1] = field[..., mask].max()

        extend = colorbar_extension(clims[0], clims[1], field[..., mask].min(), field[..., mask].max())

        return extend

    def replot(self):
        """
        Helper method to nuke and existing plot in the current self.axes and reset everything to clean.

        """

        self.axes.cla()
        self._init_figure()
        self.tripcolor_plot = None
        self.line_plot = None
        self.quiver_plot = None
        self.quiver_key = None
        self.scatter_plot = None
        self.streamline_plot = None

    def plot_field(self, field, *args, **kwargs):
        """
        Map the given `field'.

        Parameters
        ----------
        field : np.ndarray
            Field to plot (either on elements or nodes).

        Additional arguments and keyword arguments are passed to `matplotlib.pyplot.tripcolor'.

        """

        # We ignore the mask given when initialising the Plotter object and instead use the one given when calling
        # this function. We'll warn in case anything (understandably) gets confused.
        if self.mask is not None:
            warn("The mask given when initiliasing this object is ignored for plotting surfaces. Supply a `mask' "
                 "keyword to this function instead")

        if self.tripcolor_plot:
            # The field needs to be on the elements since that's the how it's plotted internally in tripcolor (unless
            # shading is 'gouraud'). Check if we've been given element data and if not, convert accordingly. If we've
            # been given a mask, things get compliated. We can't mask with a mask which varies in time (so a static
            # in time mask is fine, but one that varies doesn't work with set_array. So we need to firstly find out
            # if we've got a mask whose valid positions matches what we've already got, if so, easy peasy,
            # just update the array with set_array. If it doesn't match, the only way to mask the data properly is to
            # make a brand new plot.
            if 'mask' in kwargs:
                if len(self.tripcolor_plot.get_array()) == ~kwargs['mask'].sum():
                    if self._debug:
                        print('updating')
                    # Mask is probably the same as the previous one (based on number of positions). Mask sense needs
                    # to be inverted when setting the array as we're supplying valid positions, not hiding invalid
                    # ones. Confusing, isn't it. You can imagine the fun I had figuring this out.
                    if len(field) == len(self.mx):
                        self.tripcolor_plot.set_array(nodes2elems(field, self.triangles)[~kwargs['mask']])
                    else:
                        self.tripcolor_plot.set_array(field[~kwargs['mask']])
                    return
                else:
                    # Nothing to do here except clear the plot and make a brand new plot (which is a lot slower),
                    self.tripcolor_plot.remove()
                    if self._debug:
                        print('replotting')
            else:
                if len(field) == len(self.mx):
                    self.tripcolor_plot.set_array(nodes2elems(field, self.triangles))
                else:
                    self.tripcolor_plot.set_array(field)
                return

        self.tripcolor_plot = self.axes.tripcolor(self.mx, self.my, self.triangles, np.squeeze(field), *args,
                                                  vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, norm=self.norm,
                                                  **self._plot_projection, **kwargs)

        if self.cartesian:
            self.axes.set_aspect('equal')
            self.axes.set_xlim(self.mx.min(), self.mx.max())
            self.axes.set_ylim(self.my.min(), self.my.max())

        extend = copy.copy(self.extend)
        if extend is None:
            extend = self.get_colourbar_extension(field, (self.vmin, self.vmax))

        if self.cbar is None:
            if self.cartesian:
                divider = make_axes_locatable(self.axes)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                self.cbar = self.figure.colorbar(self.tripcolor_plot, cax=cax, extend=extend)
            elif self.mapper == 'cartopy':
                self.cbar = self.figure.colorbar(self.tripcolor_plot, extend=extend)
            else:
                self.cbar = self.m.colorbar(self.tripcolor_plot, extend=extend)
            self.cbar.ax.tick_params(labelsize=self.fs)
        if self.cb_label:
            self.cbar.set_label(self.cb_label)

    def plot_quiver(self, u, v, field=False, dx=None, dy=None, add_key=True, scale=1.0, label=None, mask_land=True, *args, **kwargs):
        """
        Quiver plot using velocity components.

        Parameters
        ----------
        u : np.ndarray
            u-component of the velocity field.
        v : np.ndarray
            v-component of the velocity field
        field : np.ndarray
            velocity magnitude field. Used to colour the vectors. Also adds a colour bar which uses the cb_label and
            cmap, if provided.
        add_key : bool, optional
            Add key for the quiver plot. Defaults to True.
        dx, dy : float, optional
            If given, the vectors will be plotted on a regular grid at intervals of `dx' and `dy'. If `dy' is omitted,
            it is assumed to be the same as `dx'.
        scale : float, optional
            Scaling to be provided to arrows with scale_units of inches. Defaults to 1.0.
        label : str, optional
            Give label to use for the quiver key (defaults to "`scale' ms^{-1}").
        mask_land : bool, optional
            Set to False to disable the (slow) masking of regular locations outside the model domain. Defaults to True.

        Additional `args' and `kwargs' are passed to `matplotlib.pyplot.quiver'.

        """

        xy_mask = np.full((len(self.lonc), len(self.latc)))
        if dx is not None:
            if dy is None:
                dy = dx
            if not hasattr(self, '_regular_lon'):
                self._make_regular_grid(dx, dy, mask_land=mask_land)
                xy_mask = self._mask_for_unstructured

            if self.cartesian:
                mxc, myc = self._regular_x, self._regular_y
            else:
                mxc, myc = self.m(self._regular_x, self._regular_y)

        else:
            mxc, myc = self.mxc, self.myc

        u = u[xy_mask]
        v = v[xy_mask]
        if np.any(field):
            field = field[xy_mask]

        if self.quiver_plot:
            if np.any(field):
                self.quiver_plot.set_UVC(u, v, field)
            else:
                self.quiver_plot.set_UVC(u, v)
            return

        if not label:
            label = '{} '.format(scale) + r'$\mathrm{ms^{-1}}$'

        if np.any(field):
            self.quiver_plot = self.axes.quiver(mxc, myc, u, v, field,
                                                cmap=self.cmap,
                                                units='inches',
                                                scale_units='inches',
                                                scale=scale,
                                                *args,
                                                **self._plot_projection,
                                                **kwargs)
            self.cbar = self.m(self.quiver_plot)
            self.cbar.ax.tick_params(labelsize=self.fs)
            if self.cb_label:
                self.cbar.set_label(self.cb_label)
        else:
            self.quiver_plot = self.axes.quiver(mxc, myc, u, v, units='inches', scale_units='inches', scale=scale,
                                                **self._plot_projection)

        if add_key:
            self.quiver_key = plt.quiverkey(self.quiver_plot, 0.9, 0.9, scale, label, coordinates='axes')

        if self.cartesian:
            self.axes.set_aspect('equal')
            self.axes.set_xlim(mxc.min(), mxc.max())
            self.axes.set_ylim(myc.min(), myc.max())

    def plot_lines(self, x, y, zone_number='30N', *args, **kwargs):
        """
        Plot geographical lines.

        Parameters:
        -----------
        x : np.ndarray, list
            Array of x coordinates to plot (cartesian coordinates).
        y : np.ndarray, list
            Array of y coordinates to plot (cartesian coordinates).
        zone_number : string, optional
            See PyFVCOM.coordinates documentation for a full list of supported codes. Defaults to `30N'.

        Additional `args' and `kwargs' are passed to `matplotlib.pyplot.plot'.

        """

        if 'color' not in kwargs:
            kwargs['color'] = 'r'

        lon, lat = lonlat_from_utm(x, y, zone_number)
        if self.cartesian:
            mx, my = lon, lat
        else:
            mx, my = self.m(lon, lat)

        self.line_plot = self.axes.plot(mx, my, *args, **self._plot_projection, **kwargs)

    def plot_scatter(self, x, y,  zone_number='30N', *args, **kwargs):
        """
        Plot scatter points.

        Parameters
        ----------
        x : np.ndarray, list
            Array of x coordinates to plot (cartesian coordinates).
        y : np.ndarray, list
            Array of y coordinates to plot (cartesian coordinates).
        zone_number : string, optional
            See PyFVCOM.coordinates documentation for a full list of supported codes. Defaults to `30N'.

        Additional `args' and `kwargs' are passed to `matplotlib.pyplot.scatter'.

        """

        lon, lat = lonlat_from_utm(x, y, zone_number)
        if self.cartesian:
            mx, my = lon, lat
        else:
            mx, my = self.m(lon, lat)

        self.scatter_plot = self.axes.scatter(mx, my, *args, **self._plot_projection, **kwargs)

    def plot_streamlines(self, u, v, dx=1000, dy=None, mask_land=True, **kwargs):
        """
        Plot streamlines of the given u and v data.

        The data will be interpolated to a regular grid (the streamline plotting function does not support
        unstructured grids.

        Parameters
        ----------
        u, v : np.ndarray
            Unstructured arrays of a velocity field. Single time and depth only.
        dx : float, optional
            Grid spacing for the interpolation in the x direction in metres. Defaults to 1000 metres.
        dy : float, optional
            Grid spacing for the interpolation in the y direction in metres. Defaults to `dx'.
        mask_land : bool, optional
            Set to False to disable the (slow) masking of regular locations outside the model domain. Defaults to True.

        Additional `kwargs' are passed to `matplotlib.pyplot.streamplot'.

        Notes
        -----
        - This method must interpolate the FVCOM grid onto a regular grid prior to plotting, which obviously has a
        performance penalty.
        - The `density' keyword argument for is set by default to [2.5, 5] which seems to work OK for my data. Change
        by passing a different value if performance is dire.
        - To set the colour limits for the arrows, pass a matplotlib.colors.Normalize object with the min/max values
        to PyFVCOM.plot.Plotter. Don't bother trying to do it via self.streamline_plot.arrows.set_clim(). The
        equivalent method on self.streamline_plot.lines works fine, but the arrows one doesn't.

        """

        if self.mapper != 'cartopy':
            raise ValueError("The streamplot function is subtly broken with Basemap plotting. Use cartopy instead.")

        if dx is not None and dy is None:
            dy = dx

        # In theory, changing the x and y positions as well as the colours is possible via a few self.stream_plot
        # methods (set_offsets, set_array), I've not found the correct way of doing this, however. In addition,
        # removing the lines is easy enough (self.streamline_plot.lines.remove()) but the equivalent method for
        # self.streamline_plot.arrows returns "not yet implemented". So, we'll just nuke the plot and start again.
        if self.streamline_plot is not None:
            self.replot()

        # Set a decent initial density if we haven't been given one in kwargs.
        if 'density' not in kwargs:
            kwargs['density'] = [2.5, 5]

        if 'cmap' in kwargs and self.cmap is not None:
            kwargs.pop('cmap', None)
            warn('Ignoring the given colour map as one has been supplied during initialisation.')

        if not hasattr(self, '_mask_for_unstructured'):
            self._make_regular_grid(dx, dy, mask_land=mask_land)

        # Remove singleton dimensions because they break the masking.
        u = np.squeeze(u)
        v = np.squeeze(v)

        if self.cartesian:
            plot_x, plot_y = self._regular_x[0, :], self._regular_y[:, 0]
            fvcom_x, fvcom_y = self.xc[self._mask_for_unstructured], self.yc[self._mask_for_unstructured]
        else:
            if self.mapper == 'cartopy':
                plot_x, plot_y = self._regular_x, self._regular_y
            else:
                # The Basemap version needs 1D arrays only.
                plot_x, plot_y = self._regular_x[0, :], self._regular_y[:, 0]
            fvcom_x, fvcom_y = self.lonc[self._mask_for_unstructured], self.latc[self._mask_for_unstructured]

        # Interpolate whatever positions we have (spherical/cartesian).
        ua_r = mp_interp_func((fvcom_x, fvcom_y, u[self._mask_for_unstructured], self._regular_x, self._regular_y))
        va_r = mp_interp_func((fvcom_x, fvcom_y, v[self._mask_for_unstructured], self._regular_x, self._regular_y))

        # Check for a colour map in kwargs and if we have one, make a magnitude array for the plot. Check we haven't
        # been given a color array in kwargs too.
        speed_r = None
        if self.cmap is not None:
            if 'color' in kwargs:
                speed_r = mp_interp_func((fvcom_x, fvcom_y,
                                          np.squeeze(kwargs['color'])[self._mask_for_unstructured],
                                          self._regular_x, self._regular_y))
                kwargs.pop('color', None)
            else:
                speed_r = np.hypot(ua_r, va_r)

        # Apparently, really tiny velocities fail to plot, so skip if we are in that situation. Exclude NaNs in this
        # check. I'm not a fan of this hardcoded threshold...
        # Nope, don't do this, let the calling script handle the error.
        # if np.all(np.hypot(u[np.isfinite(u)], v[np.isfinite(v)]) < 0.04):
        #     if self._debug:
        #         print('Skipping due to all tiny values in the input vector components.')
        #     return

        # Mask off arrays as appropriate.
        ua_r = np.ma.array(ua_r, mask=self._mask_for_regular)
        va_r = np.ma.array(va_r, mask=self._mask_for_regular)
        # Force the underlying data to NaN for the masked region. This is a problem which manifests itself when
        # plotting with cartopy.
        ua_r.data[self._mask_for_regular] = np.nan
        va_r.data[self._mask_for_regular] = np.nan
        if self.cmap is not None:
            speed_r = np.ma.array(speed_r, mask=self._mask_for_regular)
            speed_r.data[self._mask_for_regular] = np.nan

        # Now we have some data, do the streamline plot.
        self.streamline_plot = self.axes.streamplot(plot_x, plot_y, ua_r, va_r, color=speed_r, cmap=self.cmap,
                                                    norm=self.norm, **self._plot_projection, **kwargs)

        if self.mapper == 'cartopy' and not hasattr(self, '_mask_patch'):
            # I simply cannot get cartopy to not plot arrows outside the domain. So, the only thing I can think of
            # doing is making a polygon out of the region which is outside the model domain and plotting that on top
            # as white. It'll sit just above the arrow zorder. It's not currently possible to simply remove the
            # arrows either.
            warn("Cartopy doesn't mask the arrows on the streamlines correctly, so we're overlaying a white polygon to "
                 "hide them. Things underneath it will disappear.")
            model_boundaries = get_boundary_polygons(self.triangles)
            model_polygons = [Polygon(np.asarray((self.lon[i], self.lat[i])).T) for i in model_boundaries]
            polygon_areas = [i.area for i in model_polygons]
            main_polygon_index = polygon_areas.index(max(polygon_areas))
            model_domain = model_polygons[main_polygon_index]
            # Make a polygon of the regular grid extents and then subtract the model domain from that to yield a
            # masking polyon. Plot that afterwards.
            regular_domain = Polygon(((self._regular_x.min() - 1, self._regular_y.min() - 1),  # lower left
                                      (self._regular_x.min() - 1, self._regular_y.max() + 1),  # upper left
                                      (self._regular_x.max() + 1, self._regular_y.max() + 1),  # upper right
                                      (self._regular_x.max() + 1, self._regular_y.min() - 1)))  # lower right
            mask_domain = regular_domain.difference(model_domain)
            self._mask_patch = PolygonPatch(mask_domain, facecolor='w', edgecolor='none',
                                            **self._plot_projection)
            patch = self.axes.add_patch(self._mask_patch)
            patch.set_zorder(self.streamline_plot.arrows.get_zorder() + 1)

        if self.cmap is not None:
            extend = copy.copy(self.extend)
            if extend is None:
                extend = self.get_colourbar_extension(speed_r, (self.vmin, self.vmax))

            if self.cbar is None:
                if self.cartesian:
                    divider = make_axes_locatable(self.axes)
                    cax = divider.append_axes("right", size="3%", pad=0.1)
                    self.cbar = self.figure.colorbar(self.streamline_plot.lines, cax=cax, extend=extend)
                elif self.mapper == 'cartopy':
                    self.cbar = self.figure.colorbar(self.streamline_plot.lines, extend=extend)
                else:
                    self.cbar = self.m.colorbar(self.streamline_plot.lines, extend=extend)
                self.cbar.ax.tick_params(labelsize=self.fs)

            if self.cb_label:
                self.cbar.set_label(self.cb_label)

        if self.cartesian:
            self.axes.set_aspect('equal')
            self.axes.set_xlim(plot_x.min(), plot_x.max())
            self.axes.set_ylim(plot_y.min(), plot_y.max())

    def _make_regular_grid(self, dx, dy, mask_land=True):
        """
        Make a regular grid at intervals of `dx', `dy' for the current plot domain. Supports both spherical and
        cartesian grids.

        Locations which are either outside the model domain (defined as the largest polygon by area) or on islands
        are stored in the self._mask_for_regular array.

        Locations in the FVCOM grid which are outside the plotting extent are masked in the
        self._mask_for_unstructured array.

        Parameters
        ----------
        dx : float
            Grid spacing in the x-direction in metres.
        dy :
            Grid spacing in the y-direction in metres.
        mask_land : bool, optional
            Set to False to disable the (slow) masking of regular locations outside the model domain. Defaults to True.

        Provides
        --------
        self._regular_x : np.ma.ndarray
            The regularly gridded x positions as a masked array.
        self._regular_y : np.ma.ndarray
            The regularly gridded y positions as a masked array.
        self._mask_for_regular : np.ndarray
            The mask for the regular grid positions.
        self._mask_for_unstructured : np.ndarray
            The mask for the unstructured positions within the current plot domain.

        """

        # To speed things up, extract only the positions actually within the mapping domain.
        if self.cartesian:
            x = self.xc
            y = self.yc
            if self.extents is not None:
                west, east, south, north = self.extents
            else:
                west, east, south, north = self.x.min(), self.x.max(), self.y.min(), self.y.max()
        else:
            x = self.lonc
            y = self.latc
            # Should we use self.extents here?
            if self.mapper == 'basemap':
                west, east, south, north = self.m.llcrnrlon, self.m.urcrnrlon, self.m.llcrnrlat, self.m.urcrnrlat
            else:
                west, east, south, north = self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()

        self._mask_for_unstructured = (x >= west) & (x <= east) & (y >= south) * (y <= north)

        x = x[self._mask_for_unstructured]
        y = y[self._mask_for_unstructured]

        if self.cartesian:
            # Easy peasy, just return the relevant set of numbers with the given increments.
            reg_x = np.arange(x.min(), x.max() + dx, dx)
            reg_y = np.arange(y.min(), y.max() + dy, dy)
        else:
            # Convert dx and dy into spherical distances so we can do a regular grid on the lonc/latc arrays. This is a
            # pretty hacky way of going about this.
            xref, yref = self.xc[self._mask_for_unstructured].mean(), self.yc[self._mask_for_unstructured].mean()
            # Get the zone we're in for the mean position.
            _, _, zone = utm_from_lonlat(x.mean(), y.mean())
            start_x, start_y = lonlat_from_utm(xref, yref, zone=zone[0])
            _, end_y = lonlat_from_utm(xref, yref + dy, zone=zone[0])
            end_x, _ = lonlat_from_utm(xref + dx, yref, zone=zone[0])
            dx_spherical = end_x - start_x
            dy_spherical = end_y - start_y
            reg_x = np.arange(x.min(), x.max() + dx_spherical, dx_spherical)
            reg_y = np.arange(y.min(), y.max() + dy_spherical, dy_spherical)

        self._regular_x, self._regular_y = np.meshgrid(reg_x, reg_y)

        self._mask_for_regular = np.full(self._regular_x.shape, False)
        if mask_land:
            # Make a mask for the regular grid. This uses the model domain to identify points which are outside the
            # grid. Those are set to False whereas those in the domain are True. We assume the longest polygon is the
            # model boundary and all other polygons are islands within it.
            model_boundaries = get_boundary_polygons(self.triangles)
            model_polygons = [Polygon(np.asarray((self.lon[i], self.lat[i])).T) for i in model_boundaries]
            polygon_areas = [i.area for i in model_polygons]
            main_polygon_index = polygon_areas.index(max(polygon_areas))
            # Find locations outside the main model domain.
            ocean_indices, land_indices = [], []
            for index, sample in enumerate(zip(np.array((self._regular_x.ravel(), self._regular_y.ravel())).T)):
                point = Point(sample[0])
                if self._debug:
                    print(f'Checking outside domain point {index} of {len(self._regular_x.ravel())}', flush=True)
                if point.intersects(model_polygons[main_polygon_index]):
                    ocean_indices.append(index)
                else:
                    land_indices.append(index)

            # Mask off indices outside the main model domain.
            ocean_row, ocean_column = np.unravel_index(ocean_indices, self._regular_x.shape)
            land_row, land_column = np.unravel_index(land_indices, self._regular_x.shape)
            self._mask_for_regular[land_row, land_column] = True

            # To remove the sampling stations on islands, identify points which intersect the remaining polygons,
            # and then remove them from the sampling site list.
            land_indices = []
            # Exclude the main polygon from the list of polygons.
            # TODO: This is ripe for parallelisation, especially as it's pretty slow in serial.
            for pi, polygon in enumerate([i for count, i in enumerate(model_polygons) if count != main_polygon_index]):
                for oi, (row, column, index) in enumerate(zip(ocean_row, ocean_column, ocean_indices)):
                    point = Point((self._regular_x[row, column], self._regular_y[row, column]))
                    if self._debug:
                        print(f'Polygon {pi + 1} of {len(model_polygons) - 1}: '
                              f'ocean point {oi} of {len(ocean_indices)}', flush=True)
                    if point.intersects(polygon):
                        land_indices.append(index)

            # Mask off island indices.
            land_row, land_column = np.unravel_index(land_indices, self._regular_x.shape)
            self._mask_for_regular[land_row, land_column] = True

            self._regular_x = np.ma.masked_array(self._regular_x, mask=self._mask_for_regular)
            self._regular_y = np.ma.masked_array(self._regular_y, mask=self._mask_for_regular)

    def set_title(self, title):
        """ Set the title for the current axis. """
        self.axes.set_title(title, fontsize=self.fs)

    def add_scale(self, x, y, ref_lon, ref_lat, length, **kwargs):
        """
        Add a Basemap scale to the plot.

        Parameters
        ----------
        x, y : float
            The position (in map units).
        ref_lon, ref_lat : float
            The reference longitude and latitude for the scale length.
        length : float
            The length of the scale (in kilometres).

        Additional keyword arguments are passed to self.m.drawmapscale.

        """

        self.m.drawmapscale(x, y, ref_lon, ref_lat, length, ax=self.axes, **kwargs)

    def close(self):
        """ Close the current figure. """
        plt.close(self.figure)


class CrossPlotter(Plotter):
    """ Create cross-section plots based on output from the FVCOM.

    Class to assist in the creation of cross section plots of FVCOM data

    Provides
    --------

    cross_section_init(cross_section_points, dist_res) -
        Initialises the cross section working out the time varying y coordinates and wetting and drying.
        cross_section_points - list of 2x2 arrays defining the cross section (piecewise lines)
        dist_res - resolution to sample the cross section at

    plot_pcolor_field(var, timestep) -
        Plot pcolor of variable at given timestep index
        var - string of variable name
        timestep - integer timestep index


    Example
    -------
    >>> import numpy as np
    >>> import PyFVCOM as pf
    >>> import matplotlib.pyplot as plt
    >>> filestr = '/data/euryale2/scratch/mbe/Models_2/FVCOM/tamar/output/depth_tweak2_phys_only/2006/03/tamar_v2_0001.nc'
    >>> filereader = pf.read.FileReader(filestr)
    >>> cross_points = [np.asarray([[413889.37304891, 5589079.54545454], [415101.00156087, 5589616.47727273]])]
    >>> c_plot = pf.plot.CrossPlotter(filereader, cmap='bwr', vmin=5, vmax=10)
    >>> c_plot.cross_section_init(cross_points, dist_res=5)
    >>> c_plot.plot_pcolor_field('temp', 150)
    >>> plt.show()

    Notes
    -----
    Only works with FileReader data. No plans to change this.

    """

    # TODO
    #  - Currently only works for scalar variables, want to get it working for vectors to do u/v/w plots
    #  - Sort colour bars
    #  - Sort left hand channel justification for multiple channels.
    #  - Error handling for no wet/dry, no land
    #  - Plus a lot of other stuff. And tidy it up.

    def __init__(self):

        super(Plotter, self).__init__()

        self.cross_plot_x = None
        self.cross_plot_y = None
        self.cross_plot_x_pcolor = None
        self.cross_plot_y_pcolor = None
        self.sub_samp = None
        self.sample_points = None
        self.sample_points_ind = None
        self.sample_points_ind_pcolor = None
        self.wet_points_data = None
        self.chan_x = None
        self.chan_y = None
        self.sub_samp = None
        self.sel_points = None
        self.xlim_vals = None
        self.ylim_vals = None

    def _init_figure(self):
        """
        Initialise a cross-sectional plot object.

        """

        if self._FileReader:
            self.nv = self.ds.grid.nv
            self.x = self.ds.grid.x
            self.y = self.ds.grid.y
        else:
            print('Only implemented for file reader input')
            raise NotImplementedError

        if self.nv.min() != 1:
            self.nv -= self.nv.min()

        self.triangles = self.nv.transpose() - 1

        if self.figure is None:
            figsize = (cm2inch(self.figsize[0]), cm2inch(self.figsize[1]))
            self.figure = plt.figure(figsize=figsize)
            self.figure.set_facecolor('white')

        if not self.axes:
            self.axes = self.figure.add_subplot(1, 1, 1)
            if self.axis_position:
                self.axes.set_position(self.axis_position)

        if self.title:
            self.axes.set_title(self.title)

    def cross_section_init(self, cross_section_points, dist_res=50, variable_at_cells=False, wetting_and_drying=True):
        """
        Sample the cross section.

        TODO: Finish this docstring!

        Parameters
        ----------
        cross_section_points :
        dist_res :
        variable_at_cells :
        wetting_and_drying :

        """
        [sub_samp, sample_cells, sample_nodes] = getcrossectiontriangles(cross_section_points[0],
                                                                         self.triangles, self.x, self.y, dist_res)

        if len(cross_section_points) > 1:
            for this_cross_section in cross_section_points[1:]:
                [this_sub_samp, this_sample_cells, this_sample_nodes] = getcrossectiontriangles(this_cross_section,
                                                                                                self.triangles,
                                                                                                self.x, self.y,
                                                                                                dist_res)
                sub_samp = np.vstack([sub_samp, this_sub_samp])
                sample_cells = np.append(sample_cells, this_sample_cells)
                sample_nodes = np.append(sample_nodes, this_sample_nodes)

        if variable_at_cells:
            self.sample_points = sample_cells
        else:
            self.sample_points = sample_nodes
        self.sub_samp = sub_samp

        self.sel_points = np.asarray(np.unique(self.sample_points[self.sample_points != -1]), dtype=int)
        sample_points_ind = np.zeros(len(self.sample_points))
        for this_ind, this_point in enumerate(self.sel_points):
            sample_points_ind[self.sample_points == this_point] = this_ind
        sample_points_ind[self.sample_points == -1] = len(self.sel_points)
        self.sample_points_ind = np.asarray(sample_points_ind, dtype=int)

        if not hasattr(self.ds.data, 'zeta'):
            self.ds.load_data(['zeta'])

        if variable_at_cells:
            siglay = self.ds.grid.siglay_center[:, self.sel_points]
            siglev = self.ds.grid.siglev_center[:, self.sel_points]
            h = self.ds.grid.h_center[self.sel_points]
            zeta = np.mean(self.ds.data.zeta[:, self.ds.grid.nv - 1], axis=1)[:, self.sel_points]
        else:
            siglay = self.ds.grid.siglay[:, self.sel_points]
            siglev = self.ds.grid.siglev[:, self.sel_points]
            h = self.ds.grid.h[self.sel_points]
            zeta = self.ds.data.zeta[:, self.sel_points]

        depth_sel = -unstructured_grid_depths(h, zeta, siglay, nan_invalid=True)
        depth_sel_pcolor = -unstructured_grid_depths(h, zeta, siglev, nan_invalid=True)

        depth_sel = self._nan_extend(depth_sel)
        depth_sel_pcolor = self._nan_extend(depth_sel_pcolor)

        # set up the x and y for the plots
        self.cross_plot_x = np.tile(np.arange(0, len(self.sample_points)),
                                    [depth_sel.shape[1], 1]) * dist_res + dist_res * 1/2
        self.cross_plot_x_pcolor = np.tile(np.arange(0, len(self.sample_points) + 1),
                                           [depth_sel_pcolor.shape[1], 1]) * dist_res

        self.cross_plot_y = -depth_sel[:, :, self.sample_points_ind]
        insert_ind = np.min(np.where(self.sample_points_ind != np.max(self.sample_points_ind))[0])
        self.sample_points_ind_pcolor = np.insert(self.sample_points_ind, insert_ind, self.sample_points_ind[insert_ind])
        self.cross_plot_y_pcolor = -depth_sel_pcolor[:, :, self.sample_points_ind_pcolor]

        # pre process the channel variables
        chan_y_raw = np.nanmin(self.cross_plot_y_pcolor, axis=1)[-1, :]
        chan_x_raw = self.cross_plot_x_pcolor[-1, :]
        max_zeta = np.ceil(np.max(zeta))
        if np.any(np.isnan(chan_y_raw)):
            chan_y_raw[np.min(np.where(~np.isnan(chan_y_raw)))] = max_zeta  # bodge to get left bank adjacent
            chan_y_raw[np.isnan(chan_y_raw)] = max_zeta
        self.chan_x, self.chan_y = self._chan_corners(chan_x_raw, chan_y_raw)

        # sort out wetting and drying nodes if requested
        if wetting_and_drying:
            if variable_at_cells:
                self.ds.load_data(['wet_cells'])
                self.wet_points_data = np.asarray(self.ds.data.wet_cells[:, self.sel_points], dtype=bool)
            else:
                self.ds.load_data(['wet_nodes'])
                self.wet_points_data = np.asarray(self.ds.data.wet_nodes[:, self.sel_points], dtype=bool)
        else:
            self.wet_points_data = np.asarray(np.ones((self.ds.dims.time, len(self.sel_points))), dtype=bool)

        self.ylim_vals = [np.floor(np.nanmin(self.cross_plot_y_pcolor)), np.ceil(np.nanmax(self.cross_plot_y_pcolor)) + 1]
        self.xlim_vals = [np.nanmin(self.cross_plot_x_pcolor), np.nanmax(self.cross_plot_x_pcolor)]

    def plot_pcolor_field(self, var, timestep):
        """
        Finish me.

        TODO: docstring!

        Parameters
        ----------
        var :
        timestep :

        """

        if isinstance(var, str):
            plot_z = self._var_prep(var, timestep).T
        else:
            plot_z = var

        plot_x = self.cross_plot_x_pcolor.T
        plot_y = self.cross_plot_y_pcolor[timestep, :, :].T

        if self.vmin is None:
            self.vmin = np.nanmin(plot_z)
        if self.vmax is None:
            self.vmax = np.nanmax(plot_z)

        for this_node in self.sel_points:
            # choose_horiz = np.asarray(self.sample_points == this_node, dtype=bool)
            choose_horiz = np.asarray(np.where(self.sample_points == this_node)[0], dtype=int)
            choose_horiz_extend = np.asarray(np.append(choose_horiz, np.max(choose_horiz) + 1), dtype=int)

            y_uniform = np.tile(np.median(plot_y[choose_horiz_extend, :], axis=0), [len(choose_horiz_extend), 1])
            pc = self.axes.pcolormesh(plot_x[choose_horiz_extend, :],
                                      y_uniform,
                                      plot_z[choose_horiz, :],
                                      cmap=self.cmap,
                                      vmin=self.vmin,
                                      vmax=self.vmax,
                                      **self._plot_projection)

        self.axes.plot(self.chan_x, self.chan_y, linewidth=2, color='black')
        self.figure.colorbar(pc)
        self.axes.set_ylim(self.ylim_vals)
        self.axes.set_xlim(self.xlim_vals)

    def plot_quiver(self, timestep, u_str='u', v_str='v', w_str='ww', w_factor=1):
        """
        Finish me.

        TODO: docstring!

        Parameters
        ----------
        timestep :
        u_str :
        v_str :
        w_str :
        w_factor :

        """
        raw_cross_u = self._var_prep(u_str, timestep)
        raw_cross_v = self._var_prep(v_str, timestep)
        raw_cross_w = self._var_prep(w_str, timestep)

        cross_u, cross_v, cross_io = self._uvw_rectify(raw_cross_u, raw_cross_v, raw_cross_w)

        plot_x = np.ma.masked_invalid(self.cross_plot_x).T
        plot_y = np.ma.masked_invalid(self.cross_plot_y[timestep, :, :]).T

        self.plot_pcolor_field(cross_io.T, timestep)
        self.axes.quiver(plot_x, plot_y, cross_u.T, cross_v.T*w_factor, **self._plot_projection)

    def _var_prep(self, var, timestep):
        """
        Finish me.

        TODO: docstring!

        Parameters
        ----------
        var :
        timestep :

        """
        self.ds.load_data([var], dims={'time': [timestep]})
        var_sel = np.squeeze(getattr(self.ds.data, var))[..., self.sel_points]

        this_step_wet_points = np.asarray(self.wet_points_data[timestep, :], dtype=bool)
        var_sel[:, ~this_step_wet_points] = np.NaN
        self.var_sel = var_sel
        var_sel_ext = self._nan_extend(var_sel)

        cross_plot_z = var_sel_ext[:, self.sample_points_ind]

        return np.ma.masked_invalid(cross_plot_z)

    def _uvw_rectify(self, u_field, v_field, w_field):
        """
        Finish me.

        TODO: docstring!

        Parameters
        ----------
        u_field :
        v_field :
        w_field :

        """
        cross_lr = np.empty(u_field.shape)
        cross_io = np.empty(v_field.shape)
        cross_ud = w_field

        pll_vec = np.empty([len(self.sub_samp), 2])
        for this_ind, (point_1, point_2) in enumerate(zip(self.sub_samp[0:-2], self.sub_samp[2:])):
            # work out pll vectors
            this_pll_vec = np.asarray([point_2[0] - point_1[0], point_2[1] - point_1[1]])
            pll_vec[this_ind + 1, :] = this_pll_vec / np.sqrt(this_pll_vec[0]**2 + this_pll_vec[1]**2)

        pll_vec[0] = pll_vec[1]
        pll_vec[-1] = pll_vec[-2]

        for this_ind, this_samp in enumerate(zip(u_field, v_field)):
            # dot product for parallel component
            cross_lr[this_ind, :] = np.asarray([np.dot(this_uv, this_pll) for this_uv, this_pll in zip(np.asarray(this_samp).T, pll_vec)])
            # cross product for normal component
            cross_io[this_ind, :] = np.asarray([np.cross(this_uv, this_pll) for this_uv, this_pll in zip(np.asarray(this_samp).T, pll_vec)])

        return np.ma.masked_invalid(cross_lr), cross_ud, np.ma.masked_invalid(cross_io)

    @staticmethod
    def _nan_extend(in_array):
        if np.ndim(in_array) == 3:
            nan_ext = np.empty([in_array.shape[0], in_array.shape[1], 1])
        elif np.ndim(in_array) == 2:
            nan_ext = np.empty([in_array.shape[0], 1])
        else:
            raise ValueError('Unsupported number of dimensions.')

        nan_ext[:] = np.NaN
        return np.append(in_array, nan_ext, axis=len(in_array.shape) - 1)

    @staticmethod
    def _chan_corners(chan_x, chan_y):
        new_chan_x = [chan_x[0]]
        new_chan_y = [chan_y[0]]

        for this_ind, this_y in enumerate(chan_y[1:]):
            if this_y != chan_y[this_ind] and not np.isnan(this_y) and not np.isnan(chan_y[this_ind]):
                new_chan_x.append(chan_x[this_ind])
                new_chan_y.append(this_y)

            new_chan_x.append(chan_x[this_ind + 1])
            new_chan_y.append(this_y)

        return np.asarray(new_chan_x), np.asarray(new_chan_y)


class MPIWorker(object):
    """ Worker class for parallel plotting. """

    def __init__(self, comm=None, root=0, verbose=False):
        """
        Create a plotting worker object. MPIWorker.plot_* load and plot a subset in time of the results.

        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm, optional
            The MPI intracommunicator object. Omit if not running in parallel.
        root : int, optional
            Specify a given rank to act as the root process. This is only for outputting verbose messages (if enabled
            with `verbose').
        verbose : bool, optional
            Set to True to enabled some verbose output messages. Defaults to False (no messages).

        """
        self.dims = None

        self.have_mpi = True
        try:
            from mpi4py import MPI
            self.MPI = MPI
        except ImportError:
            warn('No mpi4py found in this python installation. Some functions will be disabled.')
            self.have_mpi = False

        self.comm = comm
        if self.have_mpi:
            self.rank = self.comm.Get_rank()
        else:
            self.rank = 0
        self.root = root
        self._noisy = verbose

        self.field = None
        self.label = None
        self.clims = None

    def __loader(self, fvcom_file, variable):
        """
        Function to load and make meta-variables, if appropriate, which can then be plotted by `plot_*'.

        Parameters
        ----------
        fvcom_file : str, pathlib.Path
            The file to load.
        variable : str
            The variable name to load from `fvcom_file'. This can be a meta-variable name. Currently configured are:
                - 'speed'
                - 'depth_averaged_speed'
                - 'speed_anomaly'
                - 'depth_averaged_speed_anomaly'
                - 'direction'
                - 'depth_averaged_direction'

        Provides
        --------
        self.fvcom : PyFVCOM.read.FileReader
            The FVCOM data ready for plotting.

        """

        load_verbose = False
        if self._noisy and self.rank == self.root:
            load_verbose = True
            print(f'Loading {variable} data from netCDF...', end=' ', flush=True)

        load_vars = [variable]
        if variable in ('speed', 'direction', 'speed_anomaly'):
            load_vars = ['u', 'v']
        elif variable in ('depth_averaged_speed', 'depth_averaged_direction', 'depth_averaged_speed_anomaly'):
            load_vars = ['ua', 'va']
        elif variable == 'tauc':
            load_vars = [variable, 'temp', 'salinity']

        self.fvcom = FileReader(fvcom_file, variables=load_vars, dims=self.dims, verbose=load_verbose)

        try:
            self.fvcom.load_data(['wet_cells'])
        except NameError:
            print('No wetting and drying in model')

        # Make the meta-variable data.
        if variable in ('speed', 'direction'):
            self.fvcom.data.direction, self.fvcom.data.speed = vector2scalar(self.fvcom.data.u, self.fvcom.data.v)
            # Add the attributes for labelling.
            self.fvcom.atts.speed = PassiveStore()
            self.fvcom.atts.speed.long_name = 'speed'
            self.fvcom.atts.speed.units = 'ms^{-1}'
            self.fvcom.atts.direction = PassiveStore()
            self.fvcom.atts.direction.long_name = 'direction'
            self.fvcom.atts.direction.units = '\degree'
            self.fvcom.variable_dimension_names[variable] = self.fvcom.variable_dimension_names['u']

        elif variable in ('depth_averaged_speed', 'depth_averaged_direction'):
            da_dir, da_speed = vector2scalar(self.fvcom.data.ua, self.fvcom.data.va)
            self.fvcom.data.depth_averaged_direction, self.fvcom.data.depth_averaged_speed = da_dir, da_speed
            # Add the attributes for labelling.
            self.fvcom.atts.depth_averaged_speed = PassiveStore()
            self.fvcom.atts.depth_averaged_speed.long_name = 'depth-averaged speed'
            self.fvcom.atts.depth_averaged_speed.units = 'ms^{-1}'
            self.fvcom.atts.depth_averaged_direction = PassiveStore()
            self.fvcom.atts.depth_averaged_direction.long_name = 'depth-averaged direction'
            self.fvcom.atts.depth_averaged_direction.units = '\degree'
            self.fvcom.variable_dimension_names[variable] = self.fvcom.variable_dimension_names['ua']

        if variable == 'speed_anomaly':
            self.fvcom.data.speed_anomaly = self.fvcom.data.speed.mean(axis=0) - self.fvcom.data.speed
            self.fvcom.atts.speed = PassiveStore()
            self.fvcom.atts.speed.long_name = 'speed anomaly'
            self.fvcom.atts.speed.units = 'ms^{-1}'
            self.fvcom.variable_dimension_names[variable] = self.fvcom.variable_dimension_names['u']

        elif variable == 'depth_averaged_speed_anomaly':
            self.fvcom.data.depth_averaged_speed_anomaly = self.fvcom.data.uava.mean(axis=0) - self.fvcom.data.uava
            self.fvcom.atts.depth_averaged_speed_anomaly = PassiveStore()
            self.fvcom.atts.depth_averaged_speed_anomaly.long_name = 'depth-averaged speed anomaly'
            self.fvcom.atts.depth_averaged_speed_anomaly.units = 'ms^{-1}'
            self.fvcom.variable_dimension_names[variable] = self.fvcom.variable_dimension_names['ua']

        elif variable == 'tauc':
            pressure = nodes2elems(depth2pressure(self.fvcom.data.h, self.fvcom.data.y),
                                   self.fvcom.grid.triangles)
            tempc = nodes2elems(self.fvcom.data.temp, self.fvcom.grid.triangles)
            salinityc = nodes2elems(self.fvcom.data.temp, self.fvcom.grid.triangles)
            rho = dens_jackett(tempc, salinityc, pressure[np.newaxis, :])
            self.fvcom.data.tauc *= rho
            self.fvcom.atts.tauc.units = 'Nm^{-2}'
            self.fvcom.variable_dimension_names[variable] = self.fvcom.variable_dimension_names['tauc']

        if self._noisy and self.rank == self.root:
            print(f'done.', flush=True)

    def _figure_prep(self, fvcom_file, variable, dimensions, time_indices, clims, label, **kwargs):
        """ Initialise a bunch of things which can be shared across different plot types. """

        # Should this loading stuff be outside this function?
        self.dims = dimensions
        if self.dims is None:
            self.dims = {}
        self.dims.update({'time': time_indices})

        self.__loader(fvcom_file, variable)
        self.field = np.squeeze(getattr(self.fvcom.data, variable))

        # Find out what the range of data is so we can set the colour limits automatically, if necessary.
        if self.clims is None:
            if self.have_mpi:
                global_min = self.comm.reduce(np.nanmin(self.field), op=self.MPI.MIN)
                global_max = self.comm.reduce(np.nanmax(self.field), op=self.MPI.MAX)
            else:
                # Fall back to local extremes.
                global_min = np.nanmin(self.field)
                global_max = np.nanmax(self.field)
            self.clims = [global_min, global_max]
            if self.have_mpi:
                self.clims = self.comm.bcast(clims, root=0)

        if self.label is None:
            self.label = f'{getattr(self.fvcom.atts, variable).long_name.title()} ' \
                    f'(${getattr(self.fvcom.atts, variable).units}$)'

        grid_mask = np.ones(self.field[0].shape[0], dtype=bool)
        if 'extents' in kwargs:
            # We need to find the nodes/elements for the current variable to make sure our colour bar extends for
            # what we're plotting (not the entire data set).
            if 'node' in self.fvcom.variable_dimension_names[variable]:
                x = self.fvcom.grid.lon
                y = self.fvcom.grid.lat
            elif 'nele' in self.fvcom.variable_dimension_names[variable]:
                x = self.fvcom.grid.lonc
                y = self.fvcom.grid.latc
            extents = kwargs['extents']
            grid_mask = (x > extents[0]) & (x < extents[1]) & (y < extents[3]) & (y > extents[2])

        self.extend = colorbar_extension(clims[0], clims[1],
                                         self.field[..., grid_mask].min(), self.field[..., grid_mask].max())

    def plot_field(self, fvcom_file, time_indices, variable, figures_directory, label=None, set_title=False,
                   dimensions=None, clims=None, norm=None, mask=False, figure_index=None, figure_stem=None,
                   *args, **kwargs):
        """
        Plot a given horizontal surface for `variable' for the time indices in `time_indices'.

        fvcom_file : str, pathlib.Path
            The file to load.
        time_indices : list-like
            The time indices to load from the `fvcom_file'.
        variable : str
            The variable name to load from `fvcom_file'.
        figures_directory : str, pathlib.Path
            Where to save the figures. Figure files are named f'{variable}_{time_index + 1}.png'.
        label : str, optional
            What label to use for the colour bar. If omitted, uses the variable's `long_name' and `units'.
        set_title : bool, optional
            Add a title comprised of each time (formatted as '%Y-%m-%d %H:%M:%S').
        dimensions : str, optional
            What additional dimensions to load (time is handled by the `time_indices' argument).
        clims : tuple, list, optional
            Limit the colour range to these values.
        norm : matplotlib.colors.Normalize, optional
            Apply the normalisation given to the colours in the plot.
        mask : bool
            Set to True to enable masking with the FVCOM wet/dry data.
        figure_index : int
            Give a starting index for the figure names. This is useful if you're calling this function in a loop over
            multiple files.
        figure_stem : str
            Give a file name prefix for the saved figures. Defaults to f'{variable}_streamline'.

        Additional args and kwargs are passed to PyFVCOM.plot.Plotter.

        """

        self._figure_prep(fvcom_file, variable, dimensions, time_indices, clims, label, **kwargs)

        local_plot = Plotter(self.fvcom, cb_label=self.label, *args, **kwargs)

        if norm is not None:
            # Check for zero and negative values if we're LogNorm'ing the data and replace with the colour limit
            # minimum.
            invalid = self.field <= 0
            if np.any(invalid):
                if self.clims is None or self.clims[0] <= 0:
                    raise ValueError("For log-scaling data with zero or negative values, we need a floor with which "
                                     "to replace those values. This is provided through the `clims' argument, "
                                     "which hasn't been supplied, or which has a zero (or below) minimum.")
                self.field[invalid] = self.clims[0]

        if figure_index is None:
            figure_index = 0
        for local_time, global_time in enumerate(time_indices):
            if mask:
                local_mask = getattr(self.fvcom.data, 'wet_cells')[local_time] == 0
            else:
                local_mask = np.zeros(self.fvcom.dims.nele, dtype=bool)
            local_plot.plot_field(self.field[local_time], mask=local_mask)
            local_plot.tripcolor_plot.set_clim(*clims)
            if set_title:
                title_string = self.fvcom.time.datetime[local_time].strftime('%Y-%m-%d %H:%M:%S')
                local_plot.set_title(title_string)
            if figure_stem is None:
                figure_stem = f'{variable}_streamline'
            local_plot.figure.savefig(str(Path(figures_directory, f'{figure_stem}_{figure_index + global_time + 1:04d}.png')),
                                      bbox_inches='tight',
                                      pad_inches=0.2,
                                      dpi=120)

    def plot_streamlines(self, fvcom_file, time_indices, variable, figures_directory, dx=None, dy=None, label=None,
                         set_title=False, dimensions=None, clims=None, mask=False, figure_index=None, figure_stem=None,
                         stkwargs=None, mask_land=True, *args, **kwargs):
        """
        Plot a given horizontal surface for `variable' for the time indices in `time_indices'.

        fvcom_file : str, pathlib.Path
            The file to load.
        time_indices : list-like
            The time indices to load from the `fvcom_file'.
        variable : str
            The variable name to load from `fvcom_file'.
        figures_directory : str, pathlib.Path
            Where to save the figures. Figure files are named f'{variable}_streamlines_{time_index + 1}.png'.
        dx, dy : float, optional
            If given, the streamlines will be plotted on a regular grid at intervals of `dx' and `dy'. If `dy' is
            omitted, it is assumed to be the same as `dx'.
        label : str, optional
            What label to use for the colour bar. If omitted, uses the variable's `long_name' and `units'.
        set_title : bool, optional
            Add a title comprised of each time (formatted as '%Y-%m-%d %H:%M:%S').
        dimensions : str, optional
            What additional dimensions to load (time is handled by the `time_indices' argument).
        clims : tuple, list, optional
            Limit the colour range to these values.
        mask : bool, optional
            Set to True to enable masking with the FVCOM wet/dry data.
        figure_index : int, optional
            Give a starting index for the figure names. This is useful if you're calling this function in a loop over
            multiple files.
        figure_stem : str, optional
            Give a file name prefix for the saved figures. Defaults to f'{variable}_streamline'.
        mask_land : bool, optional
            Set to False to disable the (slow) masking of regular locations outside the model domain. Defaults to True.
        stkwargs : dict, optional
            Additional streamplot keyword arguments to pass.

        Additional args and kwargs are passed to PyFVCOM.plot.Plotter.

        """

        if stkwargs is None:
            stkwargs = {}

        if dx is not None and dy is None:
            dy = dx

        self._figure_prep(fvcom_file, variable, dimensions, time_indices, clims, label, **kwargs)

        local_plot = Plotter(self.fvcom, cb_label=self.label, *args, **kwargs)

        # Get the vector field of interest based on the variable name.
        if 'depth_averaged' in variable:
            u, v = self.fvcom.data.ua, self.fvcom.data.va
        else:
            u, v = np.squeeze(self.fvcom.data.u), np.squeeze(self.fvcom.data.v)

        if figure_index is None:
            figure_index = 0
        for local_time, global_time in enumerate(time_indices):
            if mask:
                local_mask = getattr(self.fvcom.data, 'wet_cells')[local_time] == 0
            else:
                local_mask = np.full(self.fvcom.dims.nele, False)
            u_local = np.ma.masked_array(u[local_time], mask=local_mask)
            v_local = np.ma.masked_array(v[local_time], mask=local_mask)
            magnitude = np.ma.masked_array(self.field[local_time], mask=local_mask)
            try:
                local_plot.plot_streamlines(u_local, v_local, color=magnitude, dx=dx, dy=dy, mask_land=mask_land,
                                            **stkwargs)
            except ValueError:
                # The plot failed (sometimes due to teeny tiny velocities. Save what we've got anyway.
                pass
            # If we got all zeros for the streamline plot, the associated object will be none, so check that here and
            # only update colours if we definitely plotted something.
            if local_plot.streamline_plot is not None:
                # The lines are a LineCollection and we can update the colour limits in one shot. The arrows need
                # iterating.
                local_plot.streamline_plot.lines.set_clim(*clims)
                if set_title:
                    title_string = self.fvcom.time.datetime[local_time].strftime('%Y-%m-%d %H:%M:%S')
                    local_plot.set_title(title_string)
            if figure_stem is None:
                figure_stem = f'{variable}_streamline'
            local_plot.figure.savefig(str(Path(figures_directory, f'{figure_stem}_{figure_index + global_time + 1:04d}.png')),
                                      bbox_inches='tight',
                                      pad_inches=0.2,
                                      dpi=120)


class Player(FuncAnimation):
    """ Animation class for FVCOM outputs. Shamelessly lifted from https://stackoverflow.com/a/46327978 """

    def __init__(self, fig, func, init_func=None, fargs=None, save_count=None, mini=0, maxi=100, pos=(0.125, 0.92),
                 **kwargs):
        """
        Initialise an animation window.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure into which we should animate.
        func : function
            The function describing the animation.
        init_func : function, optional
            An initial function for the first frame.
        fargs : tuple or None, optional
           Additional arguments to pass to each call to ``func``
        save_count : int, optional
           The number of values from `frames` to cache.
        mini : int, optional
            The start index for the animation (defaults to zero).
        maxi : int, optional
            The maximum index for the animation (defaults to 100).
        pos : tuple
            The (x, y) position of the player controls. Defaults to near the top of the figure.

        Additional kwargs are passed to `matplotlib.animation.FuncAnimation'.

        """

        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)

        super().__init__(self.fig, self.func, frames=self.play(), init_func=init_func, fargs=fargs,
                         save_count=save_count, **kwargs)

    def play(self, *dummy):
        """ What to do when we play the animation. """
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.min < self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self, *dummy):
        """ Start the animation. """
        self.runs = True
        self.event_source.start()

    def stop(self, *dummy):
        """ Stop the animation. """
        self.runs = False
        self.event_source.stop()

    def forward(self, *dummy):
        """ Play forwards. """
        self.forwards = True
        self.start()

    def backward(self, *dummy):
        """ Play backwards. """
        self.forwards = False
        self.start()

    def oneforward(self, *dummy):
        """ Skip one forwards. """
        self.forwards = True
        self.onestep()

    def onebackward(self, *dummy):
        """ Skip one backwards. """
        self.forwards = False
        self.onestep()

    def onestep(self, *dummy):
        """ Skip through one frame at a time. """
        if self.min < self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        """ Set up the animation. """
        playerax = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '', self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, i):
        """ Set the slider position. """
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self, i):
        """ Update the slider to the given position. """
        self.slider.set_val(i)


def plot_domain(domain, mesh=False, depth=False, **kwargs):
    """
    Add a domain plot to the given domain (as domain.domain_plot).

    Parameters
    ----------
    mesh : bool
        Set to True to overlay the model mesh. Defaults to False.
    depth : bool
        Set to True to plot water depth. Defaults to False. If enabled, a colour bar is added to the figure.

    Remaining keyword arguments are passed to PyFVCOM.plot.Plotter.

    Provides
    --------
    domain_plot : PyFVCOM.plot.Plotter
        The plot object.
    mesh_plot : matplotlib.axes, optional
        The mesh axis object, if enabled.

    """

    domain.domain_plot = Plotter(domain, **kwargs)

    if mesh:
        mesh_plot = domain.domain_plot.axes.triplot(domain.domain_plot.mx, domain.domain_plot.my,
                                                    domain.grid.triangles, 'k-',
                                                    linewidth=1, zorder=2000, **domain.domain_plot._plot_projection)
        domain.domain_plot.mesh_plot = mesh_plot

    if depth:
        # Make depths negative down.
        if np.all(domain.grid.h < 0):
            domain.domain_plot.plot_field(domain.grid.h)
        else:
            domain.domain_plot.plot_field(-domain.grid.h)


def colorbar_extension(colour_min, colour_max, data_min, data_max):
    """
    For the range specified by `colour_min' to `colour_max', return whether the data range specified by `data_min'
    and `data_max' is inside, outside or partially overlapping. This allows you to automatically set the `extend'
    keyword on a `matplotlib.pyplot.colorbar' call.

    Parameters
    ----------
    colour_min, colour_max : float
        Minimum and maximum value of the current colour bar limits.
    data_min, data_max : float
        Minimum and maximum value of the data limits.

    Returns
    -------
    extension : str
        Will be 'neither', 'both', 'min, or 'max' for the case when the colour_min and colour_max values are: equal
        to the data; inside the data range; only larger or only smaller, respectively.
    """

    if data_min < colour_min and data_max > colour_max:
        extension = 'both'
    elif data_min < colour_min and data_max <= colour_max:
        extension = 'min'
    elif data_min >= colour_min and data_max > colour_max:
        extension = 'max'
    else:
        extension = 'neither'

    return extension


def cm2inch(value):
    """
    Convert centimetres to inches.

    :param value:
    :return:

    """
    return value / 2.54
