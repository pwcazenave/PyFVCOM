from __future__ import division

import pytz
import jdcal
import tempfile
import numpy as np

from netCDF4 import Dataset, date2num
from datetime import datetime
from collections import namedtuple
from math import atan2

from PyFVCOM.coordinate import utm_from_lonlat
from PyFVCOM.grid import nodes2elems


class StubFile():
    """ Create an FVCOM-formatted netCDF Dataset object. """

    def __init__(self, start, end, interval, lon, lat, triangles, zone='30N'):
        """
        Create a netCDF Dataset object which replicates FVCOM model output.

        This is handy for testing various utilities within PyFVCOM.

        Parameters
        ----------
        start, end : datetime.datetime
            Datetime objects describing the start and end of the netCDF time series.
        interval : float
            Interval (in days) for the netCDF time series.
        lon, lat : list-like
            Arrays of the spherical node positions (element centres will be automatically calculated). Cartesian
            coordinates for the given `zone' (default: 30N) will be calculated automatically.
        triangles : list-like
            Triangulation table for the nodes in `lon' and `lat'. Must be zero-indexed.

        """

        self.grid = type('grid', (object,), {})()
        self.grid.lon = lon
        self.grid.lat = lat
        self.grid.nv = triangles.T + 1  # back to 1-based indexing.
        self.grid.lonc = nodes2elems(lon, triangles)
        self.grid.latc = nodes2elems(lat, triangles)
        self.grid.x, self.grid.y, _ = utm_from_lonlat(self.grid.lon, self.grid.lat, zone=zone)
        self.grid.xc, self.grid.yc, _ = utm_from_lonlat(self.grid.lonc, self.grid.latc, zone=zone)

        # Make up some bathymetry: distance from corner coordinate scaled to 100m maximum.
        self.grid.h = np.hypot(self.grid.x - self.grid.x.min(), self.grid.y - self.grid.y.min())
        self.grid.h = (self.grid.h / self.grid.h.max()) * 100.0
        self.grid.h_center = nodes2elems(self.grid.h, triangles)

        self.grid.siglev = -np.tile(np.arange(0, 1.1, 0.1), [len(self.grid.lon), 1]).T
        self.grid.siglay = -np.tile(np.arange(0.05, 1, 0.1), [len(self.grid.lon), 1]).T
        self.grid.siglev_center = nodes2elems(self.grid.siglev, triangles)
        self.grid.siglay_center = nodes2elems(self.grid.siglay, triangles)

        # Create the all the times we need.
        self.time = type('time', (object,), {})()
        self.time.datetime = date_range(start, end, interval)
        self.time.time = date2num(self.time.datetime, units='days since 1858-11-17 00:00:00')
        self.time.Times = np.array([datetime.strftime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in self.time.datetime])
        self.time.Itime = np.floor(self.time.time)
        self.time.Itime2 = (self.time.time - np.floor(self.time.time)) * 1000 * 60 * 60  # microseconds since midnight

        # Our dimension sizes.
        self.dims = type('dims', (object,), {})()
        self.dims.node = len(self.grid.lon)
        self.dims.nele = len(self.grid.lonc)
        self.dims.siglev = self.grid.siglev.shape[0]
        self.dims.siglay = self.dims.siglev - 1
        self.dims.three = 3
        self.dims.time = 0
        self.dims.actual_time = len(self.time.datetime)
        self.dims.DateStrLen = 26
        self.dims.maxnode = 11
        self.dims.maxelem = 9
        self.dims.four = 4

        # Make the stub netCDF object (self.ds)
        self._make_netCDF()

    def _make_netCDF(self):
        self.ncfile = tempfile.NamedTemporaryFile(mode='w', delete=False)
        ncopts = {'zlib': True, 'complevel': 7}
        self.ds = Dataset(self.ncfile.name, 'w', format='NETCDF4')

        # Create the relevant dimensions.
        self.ds.createDimension('node', self.dims.node)
        self.ds.createDimension('nele', self.dims.nele)
        self.ds.createDimension('siglay', self.dims.siglay)
        self.ds.createDimension('siglev', self.dims.siglev)
        self.ds.createDimension('three', self.dims.three)
        self.ds.createDimension('time', self.dims.time)
        self.ds.createDimension('DateStrLen', self.dims.DateStrLen)
        self.ds.createDimension('maxnode', self.dims.maxnode)
        self.ds.createDimension('maxelem', self.dims.maxelem)
        self.ds.createDimension('four', self.dims.four)

        # Make some global attributes.
        self.ds.setncattr('title', 'Stub FVCOM netCDF for PyFVCOM')
        self.ds.setncattr('institution', 'School for Marine Science and Technology')
        self.ds.setncattr('source', 'FVCOM_3.0')
        self.ds.setncattr('history', 'model started at: 02/08/2017   02:35')
        self.ds.setncattr('references', 'http://fvcom.smast.umassd.edu, http://codfish.smast.umassd.edu')
        self.ds.setncattr('Conventions', 'CF-1.0')
        self.ds.setncattr('CoordinateSystem', 'Cartesian')
        self.ds.setncattr('CoordinateProjection', 'proj=utm +ellps=WGS84 +zone=30')
        self.ds.setncattr('Tidal_Forcing', 'TIDAL ELEVATION FORCING IS OFF!')
        self.ds.setncattr('River_Forcing', 'THERE ARE NO RIVERS IN THIS MODEL')
        self.ds.setncattr('GroundWater_Forcing', 'GROUND WATER FORCING IS OFF!')
        self.ds.setncattr('Surface_Heat_Forcing',
                          'FVCOM variable surface heat forcing file:\nFILE NAME:casename_wnd.nc\nSOURCE:wrf2fvcom version 0.14 (2015-10-26) (Bulk method: COARE 2.6SN)\nMET DATA START DATE:2015-06-26_18:00:00')
        self.ds.setncattr('Surface_Wind_Forcing',
                          'FVCOM variable surface Wind forcing:\nFILE NAME:casename_wnd.nc\nSOURCE:wrf2fvcom version 0.14 (2015-10-26) (Bulk method: COARE 2.6SN)\nMET DATA START DATE:2015-06-26_18:00:00')
        self.ds.setncattr('Surface_PrecipEvap_Forcing',
                          'FVCOM periodic surface precip forcing:\nFILE NAME:casename_wnd.nc\nSOURCE:wrf2fvcom version 0.14 (2015-10-26) (Bulk method: COARE 2.6SN)\nMET DATA START DATE:2015-06-26_18:00:00')

        # Make the combinations of dimensions we're likely to get.
        siglay_node = ['siglay', 'node']
        siglev_node = ['siglev', 'node']
        siglay_nele = ['siglay', 'nele']
        siglev_nele = ['siglev', 'nele']
        nele_three = ['three', 'nele']
        time_nele = ['time', 'nele']
        time_siglay_nele = ['time', 'siglay', 'nele']
        time_siglay_node = ['time', 'siglay', 'node']
        time_siglev_node = ['time', 'siglev', 'node']
        time_node = ['time', 'node']

        # Create our data variables.
        lon = self.ds.createVariable('lon', 'f4', ['node'], **ncopts)
        lon.setncattr('units', 'degrees_east')
        lon.setncattr('long_name', 'nodal longitude')
        lon.setncattr('standard_name', 'longitude')

        lat = self.ds.createVariable('lat', 'f4', ['node'], **ncopts)
        lat.setncattr('units', 'degrees_north')
        lat.setncattr('long_name', 'nodal longitude')
        lat.setncattr('standard_name', 'longitude')

        lonc = self.ds.createVariable('lonc', 'f4', ['nele'], **ncopts)
        lonc.setncattr('units', 'degrees_east')
        lonc.setncattr('long_name', 'zonal longitude')
        lonc.setncattr('standard_name', 'longitude')

        latc = self.ds.createVariable('latc', 'f4', ['nele'], **ncopts)
        latc.setncattr('units', 'degrees_north')
        latc.setncattr('long_name', 'zonal longitude')
        latc.setncattr('standard_name', 'longitude')

        siglay = self.ds.createVariable('siglay', 'f4', siglay_node, **ncopts)
        siglay.setncattr('long_name', 'Sigma Layers')
        siglay.setncattr('standard_name', 'ocean_sigma/general_coordinate')
        siglay.setncattr('positive', 'up')
        siglay.setncattr('valid_min', -1.0)
        siglay.setncattr('valid_max', 0.0)
        siglay.setncattr('formula_terms', 'sigma: siglay eta: zeta depth: h')

        siglev = self.ds.createVariable('siglev', 'f4', siglev_node, **ncopts)
        siglev.setncattr('long_name', 'Sigma Levels')
        siglev.setncattr('standard_name', 'ocean_sigma/general_coordinate')
        siglev.setncattr('positive', 'up')
        siglev.setncattr('valid_min', -1.0)
        siglev.setncattr('valid_max', 0.0)
        siglev.setncattr('formula_terms', 'sigma: siglay eta: zeta depth: h')

        siglay_center = self.ds.createVariable('siglay_center', 'f4', siglay_nele, **ncopts)
        siglay_center.setncattr('long_name', 'Sigma Layers')
        siglay_center.setncattr('standard_name', 'ocean_sigma/general_coordinate')
        siglay_center.setncattr('positive', 'up')
        siglay_center.setncattr('valid_min', -1.0)
        siglay_center.setncattr('valid_max', 0.0)
        siglay_center.setncattr('formula_terms', 'sigma:siglay_center eta: zeta_center depth: h_center')

        siglev_center = self.ds.createVariable('siglev_center', 'f4', siglev_nele, **ncopts)
        siglev_center.setncattr('long_name', 'Sigma Levels')
        siglev_center.setncattr('standard_name', 'ocean_sigma/general_coordinate')
        siglev_center.setncattr('positive', 'up')
        siglev_center.setncattr('valid_min', -1.0)
        siglev_center.setncattr('valid_max', 0.0)
        siglev_center.setncattr('formula_terms', 'sigma:siglay_center eta: zeta_center depth: h_center')

        h_center = self.ds.createVariable('h_center', 'f4', ['nele'], **ncopts)
        h_center.setncattr('long_name', 'Bathymetry')
        h_center.setncattr('standard_name', 'sea_floor_depth_below_geoid')
        h_center.setncattr('units', 'm')
        h_center.setncattr('positive', 'down')
        h_center.setncattr('grid', 'grid1 grid3')
        h_center.setncattr('coordinates', 'latc lonc')
        h_center.setncattr('grid_location', 'center')

        h = self.ds.createVariable('h', 'f4', ['node'], **ncopts)
        h.setncattr('long_name', 'Bathymetry')
        h.setncattr('standard_name', 'sea_floor_depth_below_geoid')
        h.setncattr('units', 'm')
        h.setncattr('positive', 'down')
        h.setncattr('grid', 'Bathymetry_Mesh')
        h.setncattr('coordinates', 'x y')
        h.setncattr('type', 'data')

        nv = self.ds.createVariable('nv', 'f4', nele_three, **ncopts)
        nv.setncattr('long_name', 'nodes surrounding element')

        time = self.ds.createVariable('time', 'f4', ['time'], **ncopts)
        time.setncattr('long_name', 'time')
        time.setncattr('units', 'days since 1858-11-17 00:00:00')
        time.setncattr('format', 'modified julian day (MJD)')
        time.setncattr('time_zone', 'UTC')

        Itime = self.ds.createVariable('Itime', int, ['time'], **ncopts)
        Itime.setncattr('units', 'days since 1858-11-17 00:00:00')
        Itime.setncattr('format', 'modified julian day (MJD)')
        Itime.setncattr('time_zone', 'UTC')

        Itime2 = self.ds.createVariable('Itime2', int, ['time'], **ncopts)
        Itime2.setncattr('units', 'msec since 00:00:00')
        Itime2.setncattr('time_zone', 'UTC')

        Times = self.ds.createVariable('Times', 'c', ['time', 'DateStrLen'], **ncopts)
        Times.setncattr('time_zone', 'UTC')

        # Add a single variable of each size commonly found in FVCOM (2D and 3D time series). It should be possible
        # to use create_variable() here, but I'm not sure I like the idea of spamming self with load of arrays.
        # Perhaps making a self.data would be a nice compromise.

        # 3D nodes siglev
        omega = self.ds.createVariable('omega', 'f4', time_siglev_node)
        omega.setncattr('long_name', 'Vertical Sigma Coordinate Velocity')
        omega.setncattr('units', 's-1')
        omega.setncattr('grid', 'fvcom_grid')
        omega.setncattr('type', 'data')
        # 3D nodes siglay
        temp = self.ds.createVariable('temp', 'f4', time_siglay_node)
        temp.setncattr('long_name', 'temperature')
        temp.setncattr('standard_name', 'sea_water_temperature')
        temp.setncattr('units', 'degrees_C')
        temp.setncattr('grid', 'fvcom_grid')
        temp.setncattr('coordinates', 'time siglay lat lon')
        temp.setncattr('type', 'data')
        temp.setncattr('mesh', 'fvcom_mesh')
        temp.setncattr('location', 'node')
        # 3D elements siglay
        ww = self.ds.createVariable('ww', 'f4', time_siglay_nele)
        ww.setncattr('long_name', 'Upward Water Velocity')
        ww.setncattr('units', 'meters s-1')
        ww.setncattr('grid', 'fvcom_grid')
        ww.setncattr('type', 'data')
        u = self.ds.createVariable('u', 'f4', time_siglay_nele)
        u.setncattr('long_name', 'Eastward Water Velocity')
        u.setncattr('standard_name', 'eastward_sea_water_velocity')
        u.setncattr('units', 'meters s-1')
        u.setncattr('grid', 'fvcom_grid')
        u.setncattr('type', 'data')
        u.setncattr('coordinates', 'time siglay latc lonc')
        u.setncattr('mesh', 'fvcom_mesh')
        u.setncattr('location', 'face')
        v = self.ds.createVariable('v', 'f4', time_siglay_nele)
        v.setncattr('long_name', 'Northward Water Velocity')
        v.setncattr('standard_name', 'Northward_sea_water_velocity')
        v.setncattr('units', 'meters s-1')
        v.setncattr('grid', 'fvcom_grid')
        v.setncattr('type', 'data')
        v.setncattr('coordinates', 'time siglay latc lonc')
        v.setncattr('mesh', 'fvcom_mesh')
        v.setncattr('location', 'face')
        # 2D elements
        ua = self.ds.createVariable('ua', 'f4', time_nele)
        ua.setncattr('long_name', 'Vertically Averaged x-velocity')
        ua.setncattr('units', 'meters s-1')
        ua.setncattr('grid', 'fvcom_grid')
        ua.setncattr('type', 'data')
        va = self.ds.createVariable('va', 'f4', time_nele)
        va.setncattr('long_name', 'Vertically Averaged y-velocity')
        va.setncattr('units', 'meters s-1')
        va.setncattr('grid', 'fvcom_grid')
        va.setncattr('type', 'data')
        # 2D nodes
        zeta = self.ds.createVariable('zeta', 'f4', time_node)
        zeta.setncattr('long_name', 'Water Surface Elevation')
        zeta.setncattr('units', 'meters')
        zeta.setncattr('positive', 'up')
        zeta.setncattr('standard_name', 'sea_surface_height_above_geoid')
        zeta.setncattr('grid', 'Bathymetry_Mesh')
        zeta.setncattr('coordinates', 'time lat lon')
        zeta.setncattr('type', 'data')
        zeta.setncattr('location', 'node')

        # Add our 'data'.
        lon[:] = self.grid.lon
        lat[:] = self.grid.lat
        lonc[:] = self.grid.lonc
        latc[:] = self.grid.latc
        siglay[:] = self.grid.siglay
        siglay_center[:] = self.grid.siglay_center
        siglev[:] = self.grid.siglev
        siglev_center[:] = self.grid.siglev_center
        h[:] = self.grid.h
        h_center[:] = self.grid.h_center
        nv[:] = self.grid.nv
        time[:] = self.time.time
        Times[:] = [list(t) for t in self.time.Times]  # 2D array of characters
        Itime[:] = self.time.Itime
        Itime2[:] = self.time.Itime2

        # Make up something not totally simple.
        period = (1.0 / (12 + (25 / 60))) * 24  # approximate M2 tidal period in days
        amplitude = 1.5
        phase = 0
        _omega = self._make_tide(amplitude / 100, phase + 90, period)
        _temp = np.linspace(9, 15, self.dims.actual_time)
        _ww = self._make_tide(amplitude / 150, phase + 90, period)
        _ua = self._make_tide(amplitude / 10, phase + 45, period / 2)
        _va = self._make_tide(amplitude / 20, phase + 135, period / 4)
        _zeta = self._make_tide(amplitude, phase, period)
        omega[:] = np.tile(_omega, (self.dims.node, self.dims.siglev, 1)).T * (1 - self.grid.siglev)
        temp[:] = np.tile(_temp, (self.dims.node, self.dims.siglay, 1)).T * (1 - self.grid.siglev[1:, :])
        ww[:] = np.tile(_ww, (self.dims.nele, self.dims.siglay, 1)).T * (1 - self.grid.siglev_center[1:, :])
        u[:] = np.tile(_ua, (self.dims.nele, self.dims.siglay, 1)).T * (1 - self.grid.siglev_center[1:, :])
        v[:] = np.tile(_ua, (self.dims.nele, self.dims.siglay, 1)).T * (1 - self.grid.siglev_center[1:, :])
        ua[:] = np.tile(_ua * 0.9, (self.dims.nele, 1)).T
        va[:] = np.tile(_va * 0.9, (self.dims.nele, 1)).T
        zeta[:] = np.tile(_zeta, (self.dims.node, 1)).T

        self.ds.close()

    def create_variable(self, name, dimensions, type='f4', attributes=None):
        """
        Add a variable to the current netCDF object.

        Parameters
        ----------
        name : str
            Variable name.
        dimensions : list
            List of strings describing the dimensions of the data.
        type : str
            Variable data type (defaults to 'f4').
        attributes: dict, optional
            Dictionary of attributes to add.

        """
        array = self.ds.createVariable(name, type, dimensions)
        if attributes:
            for attribute in attributes:
                setattr(array, attribute, attributes[attribute])

        setattr(self.data, name, array)

    def _make_tide(self, amplitude, phase, period):
        """ Create a sinusoid of given amplitude, phase and period. """

        tide = amplitude * np.sin((2 * np.pi * period * (self.time.time - np.min(self.time.time))) + np.deg2rad(phase))

        return tide


def fix_range(a, nmin, nmax):
    """
    Given an array of values `a', scale the values within in to the range
    specified by `nmin' and `nmax'.

    Parameters
    ----------
    a : ndarray
        Array of values to scale.
    nmin, nmax : float
        New minimum and maximum values for the new range.

    Returns
    -------
    b : ndarray
        Scaled array.

    """

    A = a.min()
    B = a.max()
    C = nmin
    D = nmax

    b = (((D - C) * (a - A)) / (B - A)) + C

    return b


def julian_day(gregorianDateTime, mjd=False):
    """
    For a given gregorian date format (YYYY,MM,DD,hh,mm,ss) get the
    Julian Day.

    Output array precision is the same as input precision, so if you
    want sub-day precision, make sure your input data are floats.

    Parameters
    ----------
    gregorianDateTime : ndarray
        Array of Gregorian dates formatted as [[YYYY, MM, DD, hh, mm,
        ss],...,[YYYY, MM, DD, hh, mm, ss]]. If hh, mm, ss are missing
        they are assumed to be zero (i.e. midnight).
    mjd : boolean, optional
        Set to True to convert output from Julian Day to Modified Julian
        Day.

    Returns
    -------
    jd : ndarray
        Modified Julian Day or Julian Day (depending on the value of
        mjd).

    Notes
    -----
    Julian Day epoch: 12:00 January 1, 4713 BC, Monday
    Modified Julain Day epoch: 00:00 November 17, 1858, Wednesday

    """

    try:
        nr, nc = np.shape(gregorianDateTime)
    except:
        nc = np.shape(gregorianDateTime)[0]
        nr = 1

    if nc < 6:
        # We're missing some aspect of the time. Let's assume it's the least
        # significant value (i.e. seconds first, then minutes, then hours).
        # Set missing values to zero.
        numMissing = 6 - nc
        if numMissing > 0:
            extraCols = np.zeros([nr, numMissing])
            if nr == 1:
                gregorianDateTime = np.hstack([gregorianDateTime, extraCols[0]])
            else:
                gregorianDateTime = np.hstack([gregorianDateTime, extraCols])

    if nr > 1:
        year = gregorianDateTime[:, 0]
        month = gregorianDateTime[:, 1]
        day = gregorianDateTime[:, 2]
        hour = gregorianDateTime[:, 3]
        minute = gregorianDateTime[:, 4]
        second = gregorianDateTime[:, 5]
    else:
        year = gregorianDateTime[0]
        month = gregorianDateTime[1]
        day = gregorianDateTime[2]
        hour = gregorianDateTime[3]
        minute = gregorianDateTime[4]
        second = gregorianDateTime[5]
    if nr == 1:
        julian, modified = jdcal.gcal2jd(year, month, day)
        modified += (hour + (minute / 60.0) + (second / 3600.0)) / 24.0
        julian += modified
    else:
        julian, modified = np.empty((nr, 1)), np.empty((nr, 1))
        for ii, tt in enumerate(gregorianDateTime):
            julian[ii], modified[ii] = jdcal.gcal2jd(tt[0], tt[1], tt[2])
            modified[ii] += (hour[ii] + (minute[ii] / 60.0) + (second[ii] / 3600.0)) / 24.0
            julian[ii] += modified[ii]

    if mjd:
        return modified
    else:
        return julian


def gregorian_date(julianDay, mjd=False):
    """
    For a given Julian Day convert to Gregorian date (YYYY, MM, DD, hh, mm,
    ss). Optionally convert from modified Julian Day with mjd=True).

    This function is adapted to Python from the MATLAB julian2greg.m function
    (http://www.mathworks.co.uk/matlabcentral/fileexchange/11410).

    Parameters
    ----------
    julianDay : ndarray
        Array of Julian Days
    mjd : boolean, optional
        Set to True if the input is Modified Julian Days.

    Returns
    -------
    greg : ndarray
        Array of [YYYY, MM, DD, hh, mm, ss].

    Example
    -------
    >>> greg = gregorianDate(np.array([53583.00390625, 55895.9765625]), mjd=True)
    >>> greg.astype(int)
    array([[2005,    8,    1,    0,    5,   37],
           [2011,   11,   30,   23,   26,   15])

    """

    if not mjd:
        # It's easier to use jdcal in Modified Julian Day
        julianDay = julianDay + 2400000.5

    try:
        nt = len(julianDay)
    except TypeError:
        nt = 1

    greg = np.empty((nt, 6))
    if nt == 1:
        ymdf = jdcal.jd2gcal(2400000.5, julianDay)
        fractionalday = ymdf[-1]
        hours = int(fractionalday * 24)
        minutes = int(((fractionalday * 24) - hours) * 60)
        seconds = ((((fractionalday * 24) - hours) * 60) - minutes) * 60
        greg = np.asarray((ymdf[0], ymdf[1], ymdf[2], hours, minutes, seconds))
    else:
        for ii, jj in enumerate(julianDay):
            ymdf = jdcal.jd2gcal(2400000.5, jj)
            greg[ii, :3] = ymdf[:3]
            fractionalday = ymdf[-1]
            hours = int(fractionalday * 24)
            minutes = int(((fractionalday * 24) - hours) * 60)
            seconds = ((((fractionalday * 24) - hours) * 60) - minutes) * 60
            greg[ii, 3:] = [hours, minutes, seconds]

    return greg


def date_range(start_date, end_date, inc=1):
    """
    Make a list of datetimes from start_date to end_date (inclusive).

    Parameters
    ----------
    start_date, end_date : datetime
        Start and end time as datetime objects. `end_date' is inclusive.
    inc : float, optional
        Specify a time increment for the list of dates in days. If omitted,
        defaults to 1 day.

    Returns
    -------
    dates : list
        List of datetimes.

    """

    start_seconds = int(start_date.replace(tzinfo=pytz.UTC).strftime('%s'))
    end_seconds = int(end_date.replace(tzinfo=pytz.UTC).strftime('%s'))

    inc *= 86400  # seconds
    dates = np.arange(start_seconds, end_seconds, inc)
    dates = [datetime.utcfromtimestamp(d) for d in dates]
    if dates[-1] != end_date:
        dates += [end_date]
    dates = np.array(dates)

    return dates


def overlap(t1start, t1end, t2start, t2end):
    """
    Find if two date ranges overlap.

    Parameters
    ----------
    datastart, dataend : datetime
        Observation start and end datetimes.
    modelstart, modelend :
        Observation start and end datetimes.

    Returns
    -------
    overlap : bool
        True if the two periods overlap at all, False otherwise.

    """

    # Shamelessly copied from http://stackoverflow.com/questions/3721249

    return (t1start <= t2start <= t1end) or (t2start <= t1start <= t2end)


def common_time(times1, times2):
    """
    Return the common date rage in two time series. At least three dates are
    required for a valid overlapping time.

    Neither date range supplied need have the same sampling or number of
    times.

    Parameters
    ----------
    times1 : list-like
        First time range (datetime objects). At least three values required.
    times2 : list-like
        Second time range (formatted as above).

    Returns
    -------
    common_time : tuple
        Start and end times indicating the common period between the two data
        sets.

    References
    ----------

    Shamelessly copied from https://stackoverflow.com/questions/9044084.

    """
    if len(times1) < 3 or len(times2) < 3:
        raise ValueError('Too few times for an overlap (times1 = {}, times2 = {})'.format(len(times1), len(times2)))
    Range = namedtuple('Range', ['start', 'end'])
    r1 = Range(start=times1[0], end=times1[-1])
    r2 = Range(start=times2[0], end=times2[-1])
    latest_start = max(r1.start, r2.start)
    earliest_end = min(r1.end, r2.end)

    return latest_start, earliest_end


def make_signal(time, amplitude=1, phase=0, period=1):
    """
    Make an arbitrary sinusoidal signal with given amplitude, phase and period over a specific time interval.

    Parameters
    ----------
    time : np.ndarray
        Time series in number of days.
    amplitude : float, optional
        A specific amplitude (defaults to 1).
    phase : float, optional
        A given phase offset in degrees (defaults to 0).
    period : float, optional
        A period for the sine wave (defaults to 1).

    Returns
    -------
    signal : np.ndarray
        The time series with the given parameters.

    """

    signal = (amplitude * np.sin((2 * np.pi * 1 / period * (time - np.min(time)) + np.deg2rad(phase))))

    return signal


def ind2sub(array_shape, index):
    """
    NOTE: Just use numpy.unravel_index!

    Replicate the MATLAB ind2sub function to return the subscript values (row,
    column) of the index for a matrix shaped `array_shape'.

    Parameters
    ----------
    array_shape : list, tuple, ndarray
        Shape of the array for which to calculate the indices.
    index : int
        Index in the flattened array.

    Returns
    -------
    row, column : int
        Indices of the row and column, respectively, in the array of shape
        `array_shape'.

    """

    # print('WARNING: Just use numpy.unravel_index!')
    # rows = int(np.array(index, dtype=int) / array_shape[1])
    # # Or numpy.mod(ind.astype('int'), array_shape[1])
    # cols = int(np.array(index, dtype=int) % array_shape[1])
    #
    # return (rows, cols)

    return np.unravel_index(index, array_shape)

