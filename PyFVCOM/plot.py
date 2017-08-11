""" Plotting class for FVCOM results. """

from __future__ import print_function

from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.tri.triangulation import Triangulation
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import DateFormatter, date2num
from datetime import datetime

from PyFVCOM.ll2utm import lonlat_from_utm
from PyFVCOM.read_results import FileReader

import numpy as np

rcParams['mathtext.default'] = 'regular'  # use non-LaTeX fonts

class Time:
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
                 fs=10, date_format=None, cb_label=None, hold=False):
        """
        Parameters
        ----------
        dataset : Dataset, PyFVCOM.read_results.FileReader
            netCDF4 Dataset or PyFVCOM.read_results.FileReader object.
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
        self.line_plot = None
        self.scatter_plot = None
        self.quiver_plot = None  # for vectors with time (e.g. currents at a point)
        self.surface_plot = None  # for depth-resolved time, for example.
        self.legend = None
        self.colorbar = None
        self.quiver_key = None

        # Are we working with a FileReader object or a bog-standard netCDF4 Dataset?
        self._FileReader = False
        if isinstance(dataset, FileReader):
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

    def plot_line(self, time_series, **kwargs):
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
            # Only add the colour bar if we're not being told to hold and this is the first time we're plotting.
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
            label = '{} $\mathrm{ms^{-1}}$'.format(scale)
            self.quiver_key = plt.quiverkey(self.quiver_plot, 0.9, 0.9, scale, label, coordinates='axes')

        # Turn off the y-axis labels as they don't correspond to the vector lengths.
        self.axes.get_yaxis().set_visible(False)

    def plot_surface(self, depth, time_series, fill_seabed=False, **kwargs):
        """
        Parameters
        ----------
        depth : np.ndarray
            Depth-varying array of depth. See `PyFVCOM.grid_tools.make_tide' for more information.
        time_series : np.ndarray
            Depth-varying array of data to plot.
        fill_seabed : bool, optional
            Set to True to fill the seabed from the maximum water depth to the edge of the plot with gray.

        """

        # Squeeze out singleton dimensions first.
        depth = np.squeeze(depth)
        time_series = np.squeeze(time_series)

        if not self.surface_plot:
            self.surface_plot = self.axes.pcolormesh(np.tile(self.time, [depth.shape[-1], 1]).T,
                                                     depth,
                                                     np.fliplr(time_series),
                                                     cmap=self.cmap)

            if fill_seabed:
                self.axes.fill_between(self.time, np.min(depth, axis=1), self.axes.get_ylim()[0], color='0.6')
            divider = make_axes_locatable(self.axes)

            cax = divider.append_axes("right", size="3%", pad=0.1)
            self.colorbar = self.figure.colorbar(self.surface_plot, cax=cax)
            self.colorbar.ax.tick_params(labelsize=self.fs)
            if self.cb_label:
                self.colorbar.set_label(self.cb_label)
        else:
            self.surface_plot

class Plotter:
    """ Create plot objects based on output from the FVCOM.

    Class to assist in the creation of plots and animations based on output
    from the FVCOM.

    Provides
    --------
    plot_field
    plot_quiver
    plot_lines
    plot_scatter
    remove_line_plots (N.B., this is mostly specific to PyLag-tools)

    Author(s)
    ---------
    James Clark (Plymouth Marine Laboratory)
    Pierre Cazenave (Plymouth Marine Laboratory)

    """

    def __init__(self, dataset, figure=None, axes=None, stations=None,
                 extents=None, vmin=None, vmax=None, mask=None, res='c', fs=10,
                 title=None, cmap='viridis', figsize=(10., 10.), axis_position=None,
                 edgecolors='none', s_stations=20, s_particles=20, linewidth=1.0,
                 tick_inc=None, cb_label=None, norm=None, m=None):
        """
        Parameters:
        -----------
        dataset : Dataset, PyFVCOM.read_results.FileReader
            netCDF4 Dataset or PyFVCOM.read_results.FileReader object.

        stations : 2D array, optional
            List of station coordinates to be plotted ([[lons], [lats]])

        extents : 1D array, optional
            Four element numpy array giving lon/lat limits (e.g. [-4.56, -3.76,
            49.96, 50.44])

        vmin : float, optional
            Lower bound to be used on colour bar (plot_field only).

        vmax : float, optional
            Upper bound to be used colour bar (plot_field only).

        mask : float, optional
            Mask out values < mask (plot_field only).

        res : string, optional
            Resolution to use when drawing Basemap object

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

        s_stations : int, optional
            Symbol size used when producing scatter plot of station locations

        s_particles : int, optional
            Symbol size used when producing scatter plot of particle locations

        linewidth : float, optional
            Linewidth to be used when generating line plots

        tick_inc : list, optional
            Add coordinate axes (i.e. lat/long) at the intervals specified in
            the list ([lon_spacing, lat_spacing]).

        cb_label : str, optional
            Set the colour bar label.

        norm : matplotlib.colors.Normalize, optional
            Normalise the luminance to 0,1. For example, use from
            matplotlib.colors.LogNorm to do log plots of fields.

        m : mpl_toolkits.basemap.Basemap, optional
            Pass a Basemap object rather than creating one on each invocation.

        Author(s):
        -------
        James Clark (PML)
        Pierre Cazenave (PML)

        """

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
        self.edgecolors = edgecolors
        self.s_stations = s_stations
        self.s_particles = s_particles
        self.linewidth = linewidth
        self.tick_inc = tick_inc
        self.cb_label = cb_label
        self.norm = norm
        self.m = m

        # Plot instances (initialise to None for truthiness test later)
        self.quiver_plot = None
        self.scat_plot = None
        self.tripcolor_plot = None
        self.tri = None
        self.masked_tris = None
        self.cbar = None
        self.line_plot = None

        # Are we working with a FileReader object or a bog-standard netCDF4 Dataset?
        self._FileReader = False
        if isinstance(dataset, FileReader):
            self._FileReader = True

        # Initialise the figure
        self.__init_figure()

    def __init_figure(self):
        # Read in required grid variables
        if self._FileReader:
            self.n_nodes = getattr(self.ds.dims, 'node')
            self.n_elems = getattr(self.ds.dims, 'nele')
            self.lon = self.ds.grid.lon
            self.lat = self.ds.grid.lat
            self.lonc = self.ds.grid.lonc
            self.latc = self.ds.grid.latc
            self.nv = self.ds.grid.nv
        else:
            self.n_nodes = len(self.ds.dimensions['node'])
            self.n_elems = len(self.ds.dimensions['nele'])
            self.lon = self.ds.variables['lon'][:]
            self.lat = self.ds.variables['lat'][:]
            self.lonc = self.ds.variables['lonc'][:]
            self.latc = self.ds.variables['latc'][:]
            self.nv = self.ds.variables['nv'][:]

        if self.nv.min() != 1:
            self.nv -= self.nv.min()

        # Triangles
        self.triangles = self.nv.transpose() - 1

        # Initialise the figure
        if self.figure is None:
            figsize = (cm2inch(self.figsize[0]), cm2inch(self.figsize[1]))
            self.figure = plt.figure(figsize=figsize)
            self.figure.set_facecolor('white')

        # Create plot axes
        if not self.axes:
            self.axes = self.figure.add_subplot(1, 1, 1)
            if self.axis_position:
                self.axes.set_position(self.axis_position)

        # If plot extents were not given, use min/max lat/lon values
        if self.extents is None:
            self.extents = np.array([self.lon.min(), self.lon.max(),
                                     self.lat.min(), self.lat.max()])

        # Create basemap object
        if not self.m:
            self.m = Basemap(llcrnrlon=self.extents[:2].min(),
                             llcrnrlat=self.extents[-2:].min(),
                             urcrnrlon=self.extents[:2].max(),
                             urcrnrlat=self.extents[-2:].max(),
                             rsphere=(6378137.00, 6356752.3142),
                             resolution=self.res,
                             projection='merc',
                             area_thresh=0.1,
                             lat_0=self.extents[-2:].mean(),
                             lon_0=self.extents[:2].mean(),
                             lat_ts=self.extents[-2:].mean(),
                             ax=self.axes)

        self.m.drawmapboundary()
        self.m.drawcoastlines(zorder=2)
        self.m.fillcontinents(color='0.6', zorder=2)

        if self.title:
            self.axes.set_title(self.title)

        # Add coordinate labels to the x and y axes.
        if self.tick_inc:
            meridians = np.arange(np.floor(np.min(self.extents[:2])), np.ceil(np.max(self.extents[:2])), self.tick_inc[0])
            parallels = np.arange(np.floor(np.min(self.extents[2:])), np.ceil(np.max(self.extents[2:])), self.tick_inc[1])
            self.m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=self.fs, linewidth=0, ax=self.axes)
            self.m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=self.fs, linewidth=0, ax=self.axes)

    def plot_field(self, field):
        """ Map the given field.

        Parameters:
        -----------
        field : 1D array TOCHECK
            Field to plot.

        """

        if self.mask is not None:
            field = np.ma.masked_where(field <= self.mask, field)

        # Update array values if the plot has already been initialised
        if self.tripcolor_plot:
            field = field[self.masked_tris].mean(axis=1)
            self.tripcolor_plot.set_array(field)
            return

        # Create tripcolor plot
        x, y = self.m(self.lon, self.lat)
        self.tri = Triangulation(x, y, self.triangles)
        self.masked_tris = self.tri.get_masked_triangles()
        field = field[self.masked_tris].mean(axis=1)
        self.tripcolor_plot = self.axes.tripcolor(x, y, self.triangles, field,
                                                  vmin=self.vmin, vmax=self.vmax, cmap=self.cmap,
                                                  edgecolors=self.edgecolors, zorder=1, norm=self.norm)

        # Overlay the grid
        # self.axes.triplot(self.tri, zorder=2)

        # Overlay stations in the first instance
        if self.stations is not None:
            mx, my = self.m(self.stations[0, :], self.stations[1, :])
            self.axes.scatter(mx, my, marker='*', c='k', s=self.s_stations, edgecolors='none', zorder=4)

        # Add colorbar scaled to axis width
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.figure.colorbar(self.tripcolor_plot, cax=cax)
        self.cbar.ax.tick_params(labelsize=self.fs)
        if self.cb_label:
            self.cbar.set_label(self.cb_label)

        return

    def plot_quiver(self, u, v, field=False, add_key=True, scale=1.0, label=None):
        """ Produce quiver plot using u and v velocity components.

        Parameters:
        -----------
        u : 1D or 2D array
            u-component of the velocity field.

        v : 1D or 2D array
            v-component of the velocity field

        field : 1D or 2D array
            velocity magnitude field. Used to colour the vectors. Also adds a colour bar which uses the cb_label and
            cmap, if provided.

        add_key : bool, optional
            Add key for the quiver plot. Defaults to True.

        scale : float, optional
            Scaling to be provided to arrows with scale_units of inches. Defaults to 1.0.

        label : str, optional
            Give label to use for the quiver key (defaults to "`scale' ms^{-1}").

        """

        if self.quiver_plot:
            if np.any(field):
                self.quiver_plot.set_UVC(u, v, field)
            else:
                self.quiver_plot.set_UVC(u, v)
            return

        if not label:
            label = '{} '.format(scale) + r'$\mathrm{ms^{-1}}$'

        x, y = self.m(self.lonc, self.latc)

        if np.any(field):
            self.quiver_plot = self.axes.quiver(x, y, u, v, field,
                                                cmap=self.cmap,
                                                units='inches',
                                                scale_units='inches',
                                                scale=scale)
            divider = make_axes_locatable(self.axes)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.cbar = self.figure.colorbar(self.quiver_plot, cax=cax)
            self.cbar.ax.tick_params(labelsize=self.fs)
            if self.cb_label:
                self.cbar.set_label(self.cb_label)
        else:
            self.quiver_plot = self.axes.quiver(x, y, u, v, units='inches', scale_units='inches', scale=scale)
        if add_key:
            self.quiver_key = plt.quiverkey(self.quiver_plot, 0.9, 0.9, scale, label, coordinates='axes')

        return

    def plot_lines(self, x, y, group_name='Default', colour='r',
                   zone_number='30N'):
        """ Plot path lines.

        Parameters:
        -----------
        x : 1D array TOCHECK
            Array of x coordinates to plot.

        y : 1D array TOCHECK
            Array of y coordinates to plot.

        group_name : str, optional
            Group name for this set of particles - a separate plot object is
            created for each group name passed in.

            Default `None'

        color : string, optional
            Colour to use when making the plot.

            Default `r'

        zone_number : string, optional
            See PyFVCOM documentation for a full list of supported codes.

        """

        if not self.line_plot:
            self.line_plot = dict()

        # Remove current line plots for this group, if they exist
        if group_name in self.line_plot:
            if self.line_plot[group_name]:
                self.remove_line_plots(group_name)

        lon, lat = lonlat_from_utm(x, y, zone_number)
        mx, my = self.m(lon, lat)
        self.line_plot[group_name] = self.axes.plot(mx, my, color=colour,
                                                    linewidth=self.linewidth, alpha=0.25, zorder=2)

    def remove_line_plots(self, group_name):
        """ Remove line plots for group `group_name'

        Parameters:
        -----------
        group_name : str
            Name of the group for which line plots should be deleted.

        """
        if self.line_plot:
            while self.line_plot[group_name]:
                self.line_plot[group_name].pop(0).remove()

    def plot_scatter(self, x, y, group_name='Default', colour='r',
                     zone_number='30N'):
        """ Plot scatter.

        Parameters:
        -----------
        x : 1D array TOCHECK
            Array of x coordinates to plot.

        y : 1D array TOCHECK
            Array of y coordinates to plot.

        group_name : str, optional
            Group name for this set of particles - a separate plot object is
            created for each group name passed in.

            Default `None'

        color : string, optional
            Colour to use when making the plot.

            Default `r'

        zone_number : string, optional
            See PyFVCOM documentation for a full list of supported codes.

            Default `30N'

        """
        if not self.scat_plot:
            self.scat_plot = dict()

        lon, lat = lonlat_from_utm(x, y, zone_number)
        mx, my = self.m(lon, lat)

        try:
            data = np.array([mx, my])
            self.scat_plot[group_name].set_offsets(data.transpose())
        except KeyError:
            self.scat_plot[group_name] = self.axes.scatter(mx, my, s=self.s_particles, color=colour, edgecolors='none',
                                                           zorder=3)

    def set_title(self, title):
        """ Set the title for the current axis. """
        self.axes.set_title(title, fontsize=self.fs)

    def close(self):
        """ Close the current figure. """
        plt.close(self.figure)


def cm2inch(value):
    """
    Convert centimetres to inches.

    :param value:
    :return:

    """
    return value / 2.54
