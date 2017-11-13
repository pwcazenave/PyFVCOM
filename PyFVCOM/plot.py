""" Plotting class for FVCOM results. """

from __future__ import print_function

from matplotlib import pyplot as plt
from matplotlib.tri.triangulation import Triangulation
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import DateFormatter, date2num
from datetime import datetime

from PyFVCOM.coordinate import lonlat_from_utm
from PyFVCOM.read import FileReader
from PyFVCOM.grid import getcrossectiontriangles, unstructured_grid_depths

import numpy as np

from warnings import warn

have_basemap = True
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    warn('No mpl_toolkits found in this python installation. Some functions will be disabled.')
    have_basemap = False

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
            label = '{} $\mathrm{ms^{-1}}$'.format(scale)
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


class Plotter():
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
                 tick_inc=None, cb_label=None, extend='neither', norm=None, m=None):
        """
        Parameters:
        -----------
        dataset : Dataset, PyFVCOM.read.FileReader
            netCDF4 Dataset or PyFVCOM.read.FileReader object.

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

        extend : str, optional
            Set the colour bar extension ('neither', 'both', 'min', 'max').
            Defaults to 'neither').

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
        self.extend = extend
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
        self._init_figure()

    def _init_figure(self):
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
            if have_basemap:
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
            else:
                raise RuntimeError('mpl_toolkits is not available in this Python.')

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

    def replot(self):
        self.axes.cla()
        self._init_figure()

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
        self.tripcolor_plot = self.axes.tripcolor(self.tri, field,
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
        self.cbar = self.figure.colorbar(self.tripcolor_plot, cax=cax, extend=self.extend)
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


class CrossPlotter(Plotter):
    """ Create plot objects based on output from the FVCOM.

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
    >>> c_plot.plot_pcolor_field('temp',150)
    >>> plt.show()


    TO DO
    -----
    Currently only works for scalar variables, want to get it working for vectors to do u/v/w plots
    Add quiver plots
    Sort colorbars
    Sort left hand channel justification for multiple channels.
    Error handling for no wet/dry, no land
    Plus a lot of other stuff. And tidy it up.

    Notes
    -----
    Only works with FileReader data. No plans to change this.

    """

    def _init_figure(self):
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

    def cross_section_init(self, cross_section_points, dist_res = 50, variable_at_cells=False):
        # sample the cross section
        [sub_samp, sample_cells, sample_nodes] = getcrossectiontriangles(cross_section_points[0], self.triangles, self.x, self.y, dist_res)

        if len(cross_section_points) > 1:
            for this_cross_section in cross_section_points[1:]:
                [this_sub_samp, this_sample_cells, this_sample_nodes] = getcrossectiontriangles(this_cross_section, self.triangles, self.x, self.y, dist_res)
                sub_samp = np.vstack([sub_samp, this_sub_samp])
                sample_cells = np.append(sample_cells, this_sample_cells)
                sample_nodes = np.append(sample_nodes, this_sample_nodes)

        if variable_at_cells:
            self.sample_points = sample_cells
        else:
            self.sample_points = sample_nodes
        self.sub_samp = sub_samp

        self.sel_points = np.asarray(np.unique(self.sample_points[self.sample_points!=-1]), dtype=int)
        sample_points_ind = np.zeros(len(self.sample_points))
        for this_ind, this_point in enumerate(self.sel_points):
            sample_points_ind[self.sample_points == this_point] = this_ind
        sample_points_ind[self.sample_points == -1] = len(self.sel_points)
        self.sample_points_ind = np.asarray(sample_points_ind, dtype=int)

        if not hasattr(self.ds.data, 'zeta'):
            self.ds.load_data(['zeta'])

        if variable_at_cells:
            siglay = self.ds.grid.siglay_center[:,self.sel_points]
            siglev = self.ds.grid.siglev_center[:,self.sel_points]
            h = self.ds.grid.h_center[self.sel_points]
            zeta = np.mean(self.ds.data.zeta[:,self.ds.grid.nv -1], axis=1)[:,self.sel_points]

        else:
            siglay = self.ds.grid.siglay[:,self.sel_points]
            siglev = self.ds.grid.siglev[:,self.sel_points]
            h = self.ds.grid.h[self.sel_points]
            zeta = self.ds.data.zeta[:,self.sel_points]


        depth_sel = -unstructured_grid_depths(h, zeta, siglay, nan_invalid=True)
        depth_sel_pcolor = -unstructured_grid_depths(h, zeta, siglev, nan_invalid=True)

        depth_sel = self._nan_extend(depth_sel)
        depth_sel_pcolor = self._nan_extend(depth_sel_pcolor)

        # set up the x and y for the plots
        self.cross_plot_x = np.tile(np.arange(0, len(self.sample_points)), [depth_sel.shape[1], 1])*dist_res + dist_res*1/2
        self.cross_plot_x_pcolor = np.tile(np.arange(0, len(self.sample_points)+1), [depth_sel_pcolor.shape[1], 1])*dist_res

        self.cross_plot_y = -depth_sel[:,:,self.sample_points_ind]
        insert_ind = np.min(np.where(self.sample_points_ind != np.max(self.sample_points_ind))[0])
        self.sample_points_ind_pcolor = np.insert(self.sample_points_ind, insert_ind, self.sample_points_ind[insert_ind])
        self.cross_plot_y_pcolor = -depth_sel_pcolor[:,:,self.sample_points_ind_pcolor]

        # pre process the channel variables
        chan_y_raw = np.nanmin(self.cross_plot_y_pcolor, axis=1)[-1,:]
        chan_x_raw = self.cross_plot_x_pcolor[-1,:]
        max_zeta = np.ceil(np.max(zeta))
        if np.any(np.isnan(chan_y_raw)):
            chan_y_raw[np.min(np.where(~np.isnan(chan_y_raw)))] = max_zeta # bodge to get left bank adjacent
            chan_y_raw[np.isnan(chan_y_raw)] = max_zeta
        self.chan_x, self.chan_y = self._chan_corners(chan_x_raw, chan_y_raw)

        # sort out wetting and drying nodes
        if variable_at_cells:
            self.ds.load_data(['wet_cells'])
            self.wet_points_data = np.asarray(self.ds.data.wet_cells[:,self.sel_points], dtype=bool)
        else:
            self.ds.load_data(['wet_nodes'])
            self.wet_points_data = np.asarray(self.ds.data.wet_nodes[:,self.sel_points], dtype=bool)

        self.ylim_vals = [np.floor(np.nanmin(self.cross_plot_y_pcolor)), np.ceil(np.nanmax(self.cross_plot_y_pcolor)) + 1]
        self.xlim_vals = [np.nanmin(self.cross_plot_x_pcolor), np.nanmax(self.cross_plot_x_pcolor)]

    def plot_pcolor_field(self, var, timestep):
        if isinstance(var, str):
            plot_z = self._var_prep(var, timestep).T
        else:
            plot_z = var

        plot_x = self.cross_plot_x_pcolor.T
        plot_y = self.cross_plot_y_pcolor[timestep,:,:].T

        if self.vmin is None:
            self.vmin = np.nanmin(plot_z)
        if self.vmax is None:
            self.vmax = np.nanmax(plot_z)

        for this_node in self.sel_points:
            choose_horiz = np.asarray(self.sample_points == this_node, dtype=bool)
            choose_horiz = np.asarray(np.where(self.sample_points == this_node)[0], dtype=int)
            choose_horiz_extend = np.asarray(np.append(choose_horiz, np.max(choose_horiz) +1), dtype=int)

            y_uniform = np.tile(np.median(plot_y[choose_horiz_extend,:], axis=0), [len(choose_horiz_extend),1])
            pc = self.axes.pcolormesh(plot_x[choose_horiz_extend,:], y_uniform,
                                        plot_z[choose_horiz,:], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)

        self.axes.plot(self.chan_x, self.chan_y, linewidth=2, color='black')
        self.figure.colorbar(pc)
        self.axes.set_ylim(self.ylim_vals)
        self.axes.set_xlim(self.xlim_vals)

    def plot_quiver(self, timestep, u_str='u', v_str='v', w_str='ww', w_factor=1):
        raw_cross_u = self._var_prep(u_str, timestep)
        raw_cross_v = self._var_prep(v_str, timestep)
        raw_cross_w = self._var_prep(w_str, timestep)

        cross_u, cross_v, cross_io = self._uvw_rectify(raw_cross_u, raw_cross_v, raw_cross_w)

        plot_x = np.ma.masked_invalid(self.cross_plot_x).T
        plot_y = np.ma.masked_invalid(self.cross_plot_y[timestep,:,:]).T

        self.plot_pcolor_field(cross_io.T, timestep)
        self.axes.quiver(plot_x, plot_y, cross_u.T, cross_v.T*w_factor)

    def _var_prep(self, var, timestep):
        self.ds.load_data([var], start=timestep, end=timestep+1)
        var_sel = np.squeeze(getattr(self.ds.data, var))[..., self.sel_points]

        this_step_wet_points = np.asarray(self.wet_points_data[timestep,:], dtype=bool)
        var_sel[:, ~this_step_wet_points] = np.NAN
        self.var_sel = var_sel
        var_sel_ext = self._nan_extend(var_sel)

        cross_plot_z = var_sel_ext[:, self.sample_points_ind]

        return np.ma.masked_invalid(cross_plot_z)

    def _uvw_rectify(self, u_field, v_field, w_field):
        cross_lr = np.empty(u_field.shape)
        cross_io = np.empty(v_field.shape)
        cross_ud = w_field

        pll_vec = np.empty([len(self.sub_samp), 2])
        for this_ind, (point_1, point_2) in enumerate(zip(self.sub_samp[0:-2], self.sub_samp[2:])):
            # work out pll vectors
            this_pll_vec = np.asarray([point_2[0] - point_1[0], point_2[1] - point_1[1]])
            pll_vec[this_ind + 1,:] = this_pll_vec/np.sqrt(this_pll_vec[0]**2 + this_pll_vec[1]**2)

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
        in_shape = in_array.shape
        if len(in_shape) == 3:
            nan_ext = np.empty([in_shape[0],in_shape[1],1])
        elif len(in_shape)==2:
            nan_ext = np.empty([in_shape[0],1])

        nan_ext[:] = np.NAN
        return np.append(in_array, nan_ext, axis=len(in_shape)-1)

    @staticmethod
    def _chan_corners(chan_x, chan_y):
        new_chan_x = [chan_x[0]]
        new_chan_y = [chan_y[0]]

        for this_ind, this_y in enumerate(chan_y[1:]):
            if this_y != chan_y[this_ind] and not np.isnan(this_y) and not np.isnan(chan_y[this_ind]):
                new_chan_x.append(chan_x[this_ind])
                new_chan_y.append(this_y)

            new_chan_x.append(chan_x[this_ind +1])
            new_chan_y.append(this_y)

        return np.asarray(new_chan_x), np.asarray(new_chan_y)


def cm2inch(value):
    """
    Convert centimetres to inches.

    :param value:
    :return:

    """
    return value / 2.54
