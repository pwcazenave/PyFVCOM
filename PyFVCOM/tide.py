"""
A series of tools with which tidal data can be extracted from FVCOM NetCDF
model results. Also provides a number of tools to interrogate the SQLite
database of tidal data collated from a range of sources across the north-west
European continental shelf.

"""

from __future__ import print_function

import os
import sys
from warnings import warn

import numpy as np
import scipy
from lxml import etree
from netCDF4 import Dataset, date2num

from PyFVCOM.grid import find_nearest_point, unstructured_grid_depths
from PyFVCOM.utilities.general import fix_range
from PyFVCOM.utilities.time import julian_day

try:
    import sqlite3
    use_sqlite = True
except ImportError:
    warn('No sqlite standard library found in this python'
         ' installation. Some functions will be disabled.')
    use_sqlite = False


class HarmonicOutput(object):
    """
    Class to create a harmonic output file which creates variables for surface elevation and currents (both
    depth-averaged and depth-resolved). Will optionally save raw data and predicted time series too.

    """
    def __init__(self, ncfile, fvcom, consts, files=None, predict=False, dump_raw=False):
        """
        Create a new netCDF file for harmonic analysis output.

        Parameters
        ----------
        ncfile : str
            Path to the netCDF to create.
        fvcom : PyFVCOM.read.FileReader
            Model data object.
        consts : list
            List of constituents used in the harmonic analysis.
        files : list, optional
            File names used to create the harmonic analysis for the metadata.
        predict : bool, optional
            Set to True to enable predicted variable creation (defaults to False).
        dump_raw : bool, optional
            Set to True to enable output of the raw data used to perform the harmonic analysis (defaults to False).

        """

        # Things to do.
        self._predict = predict
        self._dump_raw = dump_raw

        # The output file
        self._ncfile = ncfile
        self._files = files
        self._ncopts = {'zlib': True, 'complevel': 7}
        self._consts = consts

        # The data arrays
        self._time = fvcom.time.datetime
        self._lon = fvcom.grid.lon
        self._lat = fvcom.grid.lat
        self._lonc = fvcom.grid.lonc
        self._latc = fvcom.grid.latc
        self._Times = fvcom.time.Times
        self._nv = fvcom.grid.nv
        self._h = fvcom.grid.h
        self._h_center = fvcom.grid.h_center
        self._siglay = fvcom.grid.siglay
        self._siglev = fvcom.grid.siglev

        # The dimensions
        self._nx = len(self._lon)
        self._ne = len(self._lonc)
        self._nz = self._siglay.shape[0]
        self._nzlev = self._siglev.shape[0]
        self._nt = self._Times.shape[0]
        self._nconsts = len(self._consts)

        # Make the netCDF and populate the initial values (grid and time).
        self._init_structure()
        self._populate_grid()

        # Sync what we've got to disk now.
        self.sync()

    def _init_structure(self):
        if self._nz == 0:
            # Space last
            self._node_siglay_dims = ['siglay', 'node']
            self._node_siglev_dims = ['siglev', 'node']
            self._three_nele_dims = ['three', 'nele']
            self._nele_time_dims = ['time', 'nele']
            self._nele_siglay_time_dims = ['time', 'siglay', 'nele']
            self._node_time_dims = ['time', 'node']
            self._nele_nconsts_dims = ['nconsts', 'nele']
            self._nele_siglay_nconsts_dims = self._nele_consts_dims  # single-layer only
            self._node_nconsts_dims = ['nconsts', 'node']
            self._nele_coordinates = 'time latc lonc'
            self._nconsts_coordinates = 'nconsts lonc latc'
        else:
            # Space last
            self._node_siglay_dims = ['siglay', 'node']
            self._node_siglev_dims = ['siglev', 'node']
            self._three_nele_dims = ['three', 'nele']
            self._nele_time_dims = ['time', 'nele']
            self._nele_siglay_time_dims = ['time', 'siglay', 'nele']
            self._node_time_dims = ['time', 'node']
            self._nele_nconsts_dims = ['nconsts', 'nele']
            self._nele_siglay_nconsts_dims = ['nconsts', 'siglay', 'nele']  # multi-layer
            self._node_nconsts_dims = ['nconsts', 'node']
            self._nele_coordinates = 'time latc lonc'
            self._nconsts_coordinates = 'nconsts lonc latc'
        self._nc = Dataset(self._ncfile, 'w', format='NETCDF4', clobber=True)

        self._nc.createDimension('node', self._nx)
        self._nc.createDimension('nele', self._ne)
        if self._nz != 0:
            self._nc.createDimension('siglay', self._nz)
            self._nc.createDimension('siglev', self._nzlev)
        # Only create a Times variable if we're actually outputting any time dependent data.
        if self._dump_raw or self._predict:
            self._nc.createDimension('time', 0)
        self._nc.createDimension('nconsts', self._nconsts)
        self._nc.createDimension('three', 3)
        self._nc.createDimension('NameStrLen', 4)
        self._nc.createDimension('DateStrLen', 26)

        self._nc.setncattr('type', 'Harmonic analysis of elevation, u and v data')
        self._nc.setncattr('title', 'FVCOM model results harmonic analysis')
        self._nc.setncattr('author', 'Pierre Cazenave (Plymouth Marine Laboratory)')
        self._nc.setncattr('history', 'File created using {}'.format(os.path.basename(sys.argv[0])))
        if self._files:
            self._nc.setncattr('sources', 'Created from file(s): {}'.format(self._files))

        self.lon = self._nc.createVariable('lon', 'f4', ['node'], **self._ncopts)
        self.lon.setncattr('units', 'degrees_east')
        self.lon.setncattr('long_name', 'nodal longitude')
        self.lon.setncattr('standard_name', 'longitude')

        self.lat = self._nc.createVariable('lat', 'f4', ['node'], **self._ncopts)
        self.lat.setncattr('units', 'degrees_north')
        self.lat.setncattr('long_name', 'nodal longitude')
        self.lat.setncattr('standard_name', 'longitude')

        self.lonc = self._nc.createVariable('lonc', 'f4', ['nele'], **self._ncopts)
        self.lonc.setncattr('units', 'degrees_east')
        self.lonc.setncattr('long_name', 'zonal longitude')
        self.lonc.setncattr('standard_name', 'longitude')

        self.latc = self._nc.createVariable('latc', 'f4', ['nele'], **self._ncopts)
        self.latc.setncattr('units', 'degrees_north')
        self.latc.setncattr('long_name', 'zonal longitude')
        self.latc.setncattr('standard_name', 'longitude')

        self.h = self._nc.createVariable('h', 'f4', ['node'], **self._ncopts)
        self.h.setncattr('long_name', 'Bathymetry')
        self.h.setncattr('standard_name', 'sea_floor_depth_below_geoid')
        self.h.setncattr('units', 'm')
        self.h.setncattr('positive', 'down')
        self.h.setncattr('grid', 'Bathymetry_Mesh')
        self.h.setncattr('coordinates', 'x y')
        self.h.setncattr('type', 'data')

        self.h_center = self._nc.createVariable('h_center', 'f4', ['nele'], **self._ncopts)
        self.h_center.setncattr('long_name', 'Bathymetry')
        self.h_center.setncattr('standard_name', 'sea_floor_depth_below_geoid')
        self.h_center.setncattr('units', 'm')
        self.h_center.setncattr('positive', 'down')
        self.h_center.setncattr('grid', 'grid1 grid3')
        self.h_center.setncattr('coordinates', 'latc lonc')
        self.h_center.setncattr('grid_location', 'center')

        self.siglay = self._nc.createVariable('siglay', 'f4', self._node_siglay_dims, **self._ncopts)
        self.siglay.setncattr('long_name', 'Sigma Layers')
        self.siglay.setncattr('standard_name', 'ocean_sigma/general_coordinate')
        self.siglay.setncattr('positive', 'up')
        self.siglay.setncattr('valid_min', -1.0)
        self.siglay.setncattr('valid_max', 0.0)
        self.siglay.setncattr('formula_terms', 'sigma: siglay eta: zeta depth: h')

        self.siglev = self._nc.createVariable('siglev', 'f4', self._node_siglev_dims, **self._ncopts)
        self.siglev.setncattr('long_name', 'Sigma Levels')
        self.siglev.setncattr('standard_name', 'ocean_sigma/general_coordinate')
        self.siglev.setncattr('positive', 'up')
        self.siglev.setncattr('valid_min', -1.0)
        self.siglev.setncattr('valid_max', 0.0)
        self.siglev.setncattr('formula_terms', 'sigma: siglay eta: zeta depth: h')

        self.nv = self._nc.createVariable('nv', 'f4', self._three_nele_dims, **self._ncopts)
        self.nv.setncattr('long_name', 'nodes surrounding element')

        if self._dump_raw or self._predict:
            self.Times = self._nc.createVariable('Times', 'c', ['time', 'DateStrLen'], **self._ncopts)
            self.Times.setncattr('time_zone', 'UTC')

        if self._dump_raw:
            self.ua_raw = self._nc.createVariable('ua_raw', 'f4', self._nele_time_dims, **self._ncopts)
            self.ua_raw.setncattr('long_name', 'Modelled Eastward Water Depth-averaged Velocity')
            self.ua_raw.setncattr('standard_name', 'fvcom_eastward_sea_water_velocity')
            self.ua_raw.setncattr('units', 'meters s-1')
            self.ua_raw.setncattr('grid', 'fvcom_grid')
            self.ua_raw.setncattr('type', 'data')
            self.ua_raw.setncattr('coordinates', self._nele_coordinates)
            self.ua_raw.setncattr('location', 'face')

            self.va_raw = self._nc.createVariable('va_raw', 'f4', self._nele_time_dims, **self._ncopts)
            self.va_raw.setncattr('long_name', 'Modelled Northward Water Depth-averaged Velocity')
            self.va_raw.setncattr('standard_name', 'fvcom_northward_sea_water_velocity')
            self.va_raw.setncattr('units', 'meters s-1')
            self.va_raw.setncattr('grid', 'fvcom_grid')
            self.va_raw.setncattr('type', 'data')
            self.va_raw.setncattr('coordinates', self._nele_coordinates)
            self.va_raw.setncattr('location', 'face')

            self.u_raw = self._nc.createVariable('u_raw', 'f4', self._nele_siglay_time_dims, **self._ncopts)
            self.u_raw.setncattr('long_name', 'Modelled Eastward Water Velocity')
            self.u_raw.setncattr('standard_name', 'fvcom_eastward_sea_water_velocity')
            self.u_raw.setncattr('units', 'meters s-1')
            self.u_raw.setncattr('grid', 'fvcom_grid')
            self.u_raw.setncattr('type', 'data')
            self.u_raw.setncattr('coordinates', self._nele_coordinates)
            self.u_raw.setncattr('location', 'face')

            self.v_raw = self._nc.createVariable('v_raw', 'f4', self._nele_siglay_time_dims, **self._ncopts)
            self.v_raw.setncattr('long_name', 'Modelled Northward Water Velocity')
            self.v_raw.setncattr('standard_name', 'fvcom_northward_sea_water_velocity')
            self.v_raw.setncattr('units', 'meters s-1')
            self.v_raw.setncattr('grid', 'fvcom_grid')
            self.v_raw.setncattr('type', 'data')
            self.v_raw.setncattr('coordinates', self._nele_coordinates)
            self.v_raw.setncattr('location', 'face')

            self.z_raw = self._nc.createVariable('z_raw', 'f4', self._node_time_dims, **self._ncopts)
            self.z_raw.setncattr('long_name', 'Modelled Surface Elevation')
            self.z_raw.setncattr('standard_name', 'fvcom_surface_elevation')
            self.z_raw.setncattr('units', 'meters')
            self.z_raw.setncattr('grid', 'fvcom_grid')
            self.z_raw.setncattr('type', 'data')
            self.z_raw.setncattr('coordinates', 'time lat lon')
            self.z_raw.setncattr('location', 'node')

        if self._predict:
            self.ua_pred = self._nc.createVariable('ua_pred', 'f4', self._nele_time_dims, **self._ncopts)
            self.ua_pred.setncattr('long_name', 'Predicted Eastward Water Depth-averaged Velocity')
            self.ua_pred.setncattr('standard_name', 'eastward_sea_water_velocity')
            self.ua_pred.setncattr('units', 'meters s-1')
            self.ua_pred.setncattr('grid', 'fvcom_grid')
            self.ua_pred.setncattr('type', 'data')
            self.ua_pred.setncattr('coordinates', self._nele_coordinates)
            self.ua_pred.setncattr('location', 'face')

            self.va_pred = self._nc.createVariable('va_pred', 'f4', self._nele_time_dims, **self._ncopts)
            self.va_pred.setncattr('long_name', 'Predicted Northward Water Depth-averaged Velocity')
            self.va_pred.setncattr('standard_name', 'northward_sea_water_velocity')
            self.va_pred.setncattr('units', 'meters s-1')
            self.va_pred.setncattr('grid', 'fvcom_grid')
            self.va_pred.setncattr('type', 'data')
            self.va_pred.setncattr('coordinates', self._nele_coordinates)
            self.va_pred.setncattr('location', 'face')

            self.u_pred = self._nc.createVariable('u_pred', 'f4', self._nele_siglay_time_dims, **self._ncopts)
            self.u_pred.setncattr('long_name', 'Predicted Eastward Water Velocity')
            self.u_pred.setncattr('standard_name', 'eastward_sea_water_velocity')
            self.u_pred.setncattr('units', 'meters s-1')
            self.u_pred.setncattr('grid', 'fvcom_grid')
            self.u_pred.setncattr('type', 'data')
            self.u_pred.setncattr('coordinates', self._nele_coordinates)
            self.u_pred.setncattr('location', 'face')

            self.v_pred = self._nc.createVariable('v_pred', 'f4', self._nele_siglay_time_dims, **self._ncopts)
            self.v_pred.setncattr('long_name', 'Predicted Northward Water Velocity')
            self.v_pred.setncattr('standard_name', 'northward_sea_water_velocity')
            self.v_pred.setncattr('units', 'meters s-1')
            self.v_pred.setncattr('grid', 'fvcom_grid')
            self.v_pred.setncattr('type', 'data')
            self.v_pred.setncattr('coordinates', self._nele_coordinates)
            self.v_pred.setncattr('location', 'face')

            self.z_pred = self._nc.createVariable('z_pred', 'f4', self._node_time_dims, **self._ncopts)
            self.z_pred.setncattr('long_name', 'Predicted Surface Elevation')
            self.z_pred.setncattr('standard_name', 'surface_elevation')
            self.z_pred.setncattr('units', 'meters')
            self.z_pred.setncattr('grid', 'fvcom_grid')
            self.z_pred.setncattr('type', 'data')
            self.z_pred.setncattr('coordinates', 'time lat lon')
            self.z_pred.setncattr('location', 'node')

        self.u_const_names = self._nc.createVariable('u_const_names', 'c', ['nconsts', 'NameStrLen'], **self._ncopts)
        self.u_const_names.setncattr('long_name', 'Tidal constituent names for u-velocity')
        self.u_const_names.setncattr('standard_name', 'u_constituent_names')

        self.v_const_names = self._nc.createVariable('v_const_names', 'c', ['nconsts', 'NameStrLen'], **self._ncopts)
        self.v_const_names.setncattr('long_name', 'Tidal constituent names for v-velocity')
        self.v_const_names.setncattr('standard_name', 'v_constituent_names')

        self.z_const_names = self._nc.createVariable('z_const_names', 'c', ['nconsts', 'NameStrLen'], **self._ncopts)
        self.z_const_names.setncattr('long_name', 'Tidal constituent names for surface elevation')
        self.z_const_names.setncattr('standard_name', 'z_constituent_names')

        self.u_amp = self._nc.createVariable('u_amp', 'f4', self._nele_siglay_nconsts_dims, **self._ncopts)
        self.u_amp.setncattr('long_name', 'Tidal harmonic amplitudes of the u velocity')
        self.u_amp.setncattr('standard_name', 'u_amplitude')
        self.u_amp.setncattr('units', 'meters')
        self.u_amp.setncattr('grid', 'fvcom_grid')
        self.u_amp.setncattr('type', 'data')
        self.u_amp.setncattr('coordinates', self._nconsts_coordinates)

        self.v_amp = self._nc.createVariable('v_amp', 'f4', self._nele_siglay_nconsts_dims, **self._ncopts)
        self.v_amp.setncattr('long_name', 'Tidal harmonic amplitudes of the v velocity')
        self.v_amp.setncattr('standard_name', 'v_amplitude')
        self.v_amp.setncattr('units', 'meters')
        self.v_amp.setncattr('grid', 'fvcom_grid')
        self.v_amp.setncattr('type', 'data')
        self.v_amp.setncattr('coordinates', self._nconsts_coordinates)

        self.ua_amp = self._nc.createVariable('ua_amp', 'f4', self._nele_nconsts_dims, **self._ncopts)
        self.ua_amp.setncattr('long_name', 'Tidal harmonic amplitudes of the ua velocity')
        self.ua_amp.setncattr('standard_name', 'ua_amplitude')
        self.ua_amp.setncattr('units', 'meters')
        self.ua_amp.setncattr('grid', 'fvcom_grid')
        self.ua_amp.setncattr('type', 'data')
        self.ua_amp.setncattr('coordinates', self._nconsts_coordinates)

        self.va_amp = self._nc.createVariable('va_amp', 'f4', self._nele_nconsts_dims, **self._ncopts)
        self.va_amp.setncattr('long_name', 'Tidal harmonic amplitudes of the va velocity')
        self.va_amp.setncattr('standard_name', 'va_amplitude')
        self.va_amp.setncattr('units', 'meters')
        self.va_amp.setncattr('grid', 'fvcom_grid')
        self.va_amp.setncattr('type', 'data')
        self.va_amp.setncattr('coordinates', self._nconsts_coordinates)

        self.z_amp = self._nc.createVariable('z_amp', 'f4', self._node_nconsts_dims, **self._ncopts)
        self.z_amp.setncattr('long_name', 'Tidal harmonic amplitudes of the surface elevation')
        self.z_amp.setncattr('standard_name', 'z_amplitude')
        self.z_amp.setncattr('units', 'meters')
        self.z_amp.setncattr('grid', 'fvcom_grid')
        self.z_amp.setncattr('type', 'data')
        self.z_amp.setncattr('coordinates', 'lon lat nconsts')

        self.u_phase = self._nc.createVariable('u_phase', 'f4', self._nele_siglay_nconsts_dims, **self._ncopts)
        self.u_phase.setncattr('long_name', 'Tidal harmonic phases of the u velocity')
        self.u_phase.setncattr('standard_name', 'u_amplitude')
        self.u_phase.setncattr('units', 'meters')
        self.u_phase.setncattr('grid', 'fvcom_grid')
        self.u_phase.setncattr('type', 'data')
        self.u_phase.setncattr('coordinates', self._nconsts_coordinates)

        self.v_phase = self._nc.createVariable('v_phase', 'f4', self._nele_siglay_nconsts_dims, **self._ncopts)
        self.v_phase.setncattr('long_name', 'Tidal harmonic phases of the v velocity')
        self.v_phase.setncattr('standard_name', 'v_amplitude')
        self.v_phase.setncattr('units', 'meters')
        self.v_phase.setncattr('grid', 'fvcom_grid')
        self.v_phase.setncattr('type', 'data')
        self.v_phase.setncattr('coordinates', self._nconsts_coordinates)

        self.ua_phase = self._nc.createVariable('ua_phase', 'f4', self._nele_nconsts_dims, **self._ncopts)
        self.ua_phase.setncattr('long_name', 'Tidal harmonic phases of the ua velocity')
        self.ua_phase.setncattr('standard_name', 'ua_amplitude')
        self.ua_phase.setncattr('units', 'meters')
        self.ua_phase.setncattr('grid', 'fvcom_grid')
        self.ua_phase.setncattr('type', 'data')
        self.ua_phase.setncattr('coordinates', self._nconsts_coordinates)

        self.va_phase = self._nc.createVariable('va_phase', 'f4', self._nele_nconsts_dims, **self._ncopts)
        self.va_phase.setncattr('long_name', 'Tidal harmonic phases of the va velocity')
        self.va_phase.setncattr('standard_name', 'va_amplitude')
        self.va_phase.setncattr('units', 'meters')
        self.va_phase.setncattr('grid', 'fvcom_grid')
        self.va_phase.setncattr('type', 'data')
        self.va_phase.setncattr('coordinates', self._nconsts_coordinates)

        self.z_phase = self._nc.createVariable('z_phase', 'f4', self._node_nconsts_dims, **self._ncopts)
        self.z_phase.setncattr('long_name', 'Tidal harmonic phases of the surface elevation'),
        self.z_phase.setncattr('standard_name', 'z_amplitude')
        self.z_phase.setncattr('units', 'meters')
        self.z_phase.setncattr('grid', 'fvcom_grid')
        self.z_phase.setncattr('type', 'data')
        self.z_phase.setncattr('coordinates', 'lon lat nconsts')

    def _populate_grid(self):
        # Add the data we already have.
        self.lon[:] = self._lon
        self.lat[:] = self._lat
        self.lonc[:] = self._lonc
        self.latc[:] = self._latc
        self.h[:] = self._h
        self.h_center[:] = self._h_center
        self.nv[:] = self._nv
        self.z_const_names[:] = self._consts
        self.u_const_names[:] = self._consts
        self.v_const_names[:] = self._consts
        self.siglay[:] = self._siglay
        self.siglev[:] = self._siglev
        if self._predict or self._dump_raw:
            self._write_fvcom_time(self._time)

    def _write_fvcom_time(self, time, **kwargs):
        """
        Write the four standard FVCOM time variables (time, Times, Itime, Itime2) for the given time series.

        Parameters
        ----------
        time : np.ndarray, list, tuple
            Times as datetime objects.

        """

        mjd = date2num(time, units='days since 1858-11-17 00:00:00')
        Itime = np.floor(mjd)  # integer Modified Julian Days
        Itime2 = (mjd - Itime) * 24 * 60 * 60 * 1000  # milliseconds since midnight
        Times = [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in time]

        # time
        self.time = self._nc.createVariable('time', 'f4', ['time'], **self._ncopts)
        self.time.setncattr('units', 'days since 1858-11-17 00:00:00')
        self.time.setncattr('format', 'modified julian day (MJD)')
        self.time.setncattr('long_name', 'time')
        self.time.setncattr('time_zone', 'UTC')
        # Itime
        self.Itime = self._nc.createVariable('Itime', 'i', ['time'], **self._ncopts)
        self.Itime.setncattr('units', 'days since 1858-11-17 00:00:00')
        self.Itime.setncattr('format', 'modified julian day (MJD)')
        self.Itime.setncattr('time_zone', 'UTC')
        self.Itime[:] = Itime
        # Itime2
        self.Itime2 = self._nc.createVariable('Itime2', 'i', ['time'], **self._ncopts)
        self.Itime2.setncattr('units', 'msec since 00:00:00')
        self.Itime2.setncattr('time_zone', 'UTC')
        self.Itime2[:] = Itime2
        # Times
        self.Times = self._nc.createVariable('Itime2', 'c', ['time', 'DateStrLen'], **self._ncopts)
        self.Times.setncattr('long_name', 'Calendar Date')
        self.Times.setncattr('format', 'String: Calendar Time')
        self.Times.setncattr('time_zone', 'UTC')
        self.Times[:] = Times

    def close(self):
        """ Close the netCDF file handle. """
        self._nc.close()

    def sync(self):
        """ Sync data to disk now. """
        self._nc.sync()


def add_harmonic_results(db, stationName, constituentName, phase, amplitude, speed, inferred, ident=None, noisy=False):
    """
    Add data to an SQLite database.

    Parameters
    ----------
    db : str
        Full path to an SQLite database. If absent, it will be created.
    stationName : str
        Short name for the current station. This is the table name.
    constituentName : str
        Name of the current tidal constituent being added.
    phase : float
        Tidal constituent phase (in degrees).
    amplitude : float
        Tidal constituent amplitude (in metres).
    speed : float
        Tidal constituent speed (in degrees per hour).
    inferred : str
        'true' or 'false' indicating whether the values are inferred
        (i.e. the time series is too short to perform a robust harmonic
        analysis).
    ident : str
        Optional prefix for the table names in the SQLite database. Usage of
        this option means you can store both u and v data in the same database.
    noisy : bool
        Set to True to enable verbose output.

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (add_harmonic_results)'
                           ' is unavailable.')

    if not ident:
        ident = ''
    else:
        ident = '_' + ident

    conn = sqlite3.connect(db)
    c = conn.cursor()

    # Create the necessary tables if they don't exist already
    c.execute('CREATE TABLE IF NOT EXISTS TidalConstituents ( \
        shortName TEXT COLLATE nocase, \
        amplitude FLOAT(10), \
        phase FLOAT(10), \
        speed FLOAT(10), \
        constituentName TEXT COLLATE nocase, \
        amplitudeUnits TEXT COLLATE nocase, \
        phaseUnits TEXT COLLATE nocase, \
        speedUnits TEXT COLLATE nocase, \
        inferredConstituent TEXT COLLATE nocase)')

    if noisy:
        print('amplitude, phase and speed.', end=' ')
    for item in range(len(inferred)):
        c.execute('INSERT INTO TidalConstituents VALUES (?,?,?,?,?,?,?,?,?)',
            (stationName + ident, amplitude[item], phase[item], speed[item], constituentName[item], 'metres', 'degrees', 'degrees per mean solar hour', inferred[item]))

    conn.commit()

    conn.close()


def get_observed_data(db, table, startYear=False, endYear=False, noisy=False):
    """
    Extract the tidal data from the SQLite database for a given station.
    Specify the database (db), the table name (table) which needs to be the
    short name version of the station of interest.

    Optionally supply a start and end year (which if equal give all data from
    that year) to limit the returned data. If no data exists for that station,
    the output is returned as False.

    Parameters
    ----------
    db : str
        Full path to the tide data SQLite database.
    table : str
        Name of the table to be extracted (e.g. 'AVO').
    startYear : bool, optional
        Year from which to start extracting data (inclusive).
    endYear : bool, optional
        Year at which to end data extraction (inclusive).
    noisy : bool, optional
        Set to True to enable verbose output.

    See Also
    --------
    tide.get_observed_metadata : extract metadata for a tide station.

    Notes
    -----
    Search is not fuzzy, so "NorthShields" is not the same as "North Shields".
    Search is case insensitive, however.

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (get_observed_data)'
                           ' is unavailable.')

    if noisy:
        print('Getting data for {} from the database...'.format(table), end=' ')

    try:
        con = sqlite3.connect(db)

        with con:
            c = con.cursor()
            if startYear and endYear:
                # We've been given a range of data
                if startYear == endYear:
                    # We have the same start and end dates, so just do a
                    # simpler version
                    c.execute('SELECT * FROM {t} WHERE {t}.year == {sy} ORDER BY year, month, day, hour, minute, second'.format(t=table, sy=startYear))
                else:
                    # We have a date range
                    c.execute('SELECT * FROM {t} WHERE {t}.year >= {sy} AND {t}.year <= {ey} ORDER BY year, month, day, hour, minute, second'.format(t=table, sy=startYear, ey=endYear))
            else:
                # Return all data
                c.execute('SELECT * FROM {} ORDER BY year, month, day, hour, minute, second'.format(table))
            # Now get the data in a format we might actually want to use
            data = c.fetchall()

        con.close()

        if noisy:
            print('done.')

    except sqlite3.Error as e:
        if con:
            con.close()
            print('Error {}:'.format(e.args[0]))
            data = [False]

    return data


def get_observed_metadata(db, originator=False, obsdepth=None):
    """
    Extracts the meta data from the supplied database. If the supplied
    originator is False (default), then information from all stations is
    returned.

    Parameters
    ----------
    db : str
        Full path to the tide data SQLite database.
    originator : str, optional
        Specify an originator (e.g. 'NTSLF', 'NSTD', 'REFMAR') to
        extract only that data. Defaults to all data.
    obsdepth : bool, optional
        Set to True to return the observation depth (useful for current meter
        data). Defaults to False.

    Returns
    -------
    lat, lon : list
        Latitude and longitude of the requested station(s).
    site : list
        Short names (e.g. 'AVO' for 'Avonmouth') of the tide stations.
    longName : list
        Long names of the tide stations (e.g. 'Avonmouth').
    depth : list
        If obsdepth=True on input, then depths are returned, otherwise omitted.

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (get_observed_metadata)'
                           ' is unavailable.')

    con = None
    try:
        con = sqlite3.connect(db)

        c = con.cursor()

        if not originator:
            out = c.execute('SELECT * from Stations where originatorName '
                            'is ? or originatorLongName is ?',
                            [originator, originator])
        else:
            out = c.execute('SELECT * from Stations')

        # Convert it to a set of better formatted values.
        metadata = out.fetchall()
        lat = [float(m[0]) for m in metadata]
        lon = [float(m[1]) for m in metadata]
        site = [str(m[2]) for m in metadata]
        longName = [str(m[3]) for m in metadata]
        if len(metadata) > 4:
            depth = [str(m[4]) for m in metadata]
        else:
            depth = None

    except sqlite3.Error as e:
        if con:
            con.close()
            lat, lon, site, longName, depth = False, False, False, False, False
            raise Exception('SQLite error: {}'.format(e.args[0]))

    if not obsdepth:
        return lat, lon, site, longName
    else:
        return lat, lon, site, longName, depth


def clean_observed_data(data, removeResidual=False):
    """
    Process the observed raw data to a more sensible format. Also
    convert from Gregorian dates to Modified Julian Day (to match FVCOM
    model output times).

    Parameters
    ----------
    data : ndarray
        Array of [YYYY, MM, DD, hh, mm, ss, zeta, flag] data output by
        getObservedData().
    removeResidual : bool, optional
        If True, remove any residual values. Where such data are absent
        (marked by values of -9999 or -99.0), no removal is performed. Defaults
        to False.

    Returns
    -------
    dateMJD : ndarray
        Modified Julian Days of the input data.
    tideDataMSL : ndarray
        Time series of surface elevations from which the mean surface
        elevation has been subtracted. If removeResidual is True, these
        values will omit the atmospheric effects, leaving a harmonic
        signal only.
    npFlagsData : ndarray
        Flag values from the SQLite database (usually -9999, or P, N
        etc. if BODC data).
    allDateTimes : ndarray
        Original date data in [YYYY, MM, DD, hh, mm, ss] format.

    """

    npObsData = []
    npFlagData = []
    for row in data:
        npObsData.append(row[:-1])  # eliminate the flag from the numeric data
        npFlagData.append(row[-1])   # save the flag separately

    # For the tidal data, convert the numbers to floats to avoid issues
    # with truncation.
    npObsData = np.asarray(npObsData, dtype=float)
    npFlagData = np.asarray(npFlagData)

    # Extract the time and tide data
    allObsTideData = np.asarray(npObsData[:, 6])
    allObsTideResidual = np.asarray(npObsData[:, 7])
    allDateTimes = np.asarray(npObsData[:, :6], dtype=float)

    dateMJD = julian_day(allDateTimes, mjd=True)

    # Apply a correction (of sorts) from LAT to MSL by calculating the
    # mean (excluding nodata values (-99 for NTSLF, -9999 for SHOM))
    # and removing that from the elevation.
    tideDataMSL = allObsTideData - np.mean(allObsTideData[allObsTideData > -99])

    if removeResidual:
        # Replace the residuals to remove with zeros where they're -99
        # or -9999 since the net effect at those times is "we don't have
        # a residual, so just leave the original value alone".
        allObsTideResidual[allObsTideResidual <= -99] = 0
        tideDataMSL = tideDataMSL - allObsTideResidual

    return dateMJD, tideDataMSL, npFlagData, allDateTimes


def parse_TAPPY_XML(file):
    """
    Extract values from an XML file created by TAPPY.

    TODO: Allow a list of constituents to be specified when calling
    parse_TAPPY_XML.

    Parameters
    ----------
    file : str
        Full path to a TAPPY output XML file.

    Returns
    -------
    constituentName : list
        Tidal constituent names.
    constituentSpeed : list
        Tidal constituent speeds (in degrees per hour).
    constituentPhase : list
        Tidal constituent phases (in degrees).
    constituentAmplitude : list
        Tidal constituent amplitudes (in metres).
    constituentInference : list
        Flag of whether the tidal constituent was inferred due to a
        short time series for the given constituent.

    """

    tree = etree.parse(open(file, 'r'))

    constituentName = []
    constituentSpeed = []
    constituentInference = []
    constituentPhase = []
    constituentAmplitude = []

    for harmonic in tree.iter('Harmonic'):

        # Still not pretty
        for item in harmonic.iter('name'):
            constituentName.append(item.text)

        for item in harmonic.iter('speed'):
            constituentSpeed.append(item.text)

        for item in harmonic.iter('inferred'):
            constituentInference.append(item.text)

        for item in harmonic.iter('phaseAngle'):
            constituentPhase.append(item.text)

        for item in harmonic.iter('amplitude'):
            constituentAmplitude.append(item.text)

    return constituentName, constituentSpeed, constituentPhase, constituentAmplitude, constituentInference


def get_harmonics(db, stationName, noisy=False):
    """
    Use the harmonics database to extract the results of the harmonic analysis
    for a given station (stationName).

    Parameters
    ----------
    db : str
        Full path to the tidal harmonics SQLite database.
    stationName : str
        Station short name (i.e. table name).
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    siteHarmonics : dict
        Contains all the harmonics data for the given tide station. Keys and units are:
            - 'stationName' (e.g. 'AVO')
            - 'amplitude' (m)
            - 'phase' (degrees)
            - 'speed' (degrees per mean solar hour)
            - 'constituentName' (e.g. 'M2')
            - 'inferredConstituent' ('true'|'false')

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (get_harmonics) is'
                           ' unavailable.')

    if noisy:
        print('Getting harmonics data for site {}...'.format(stationName), end=' ')

    try:
        con = sqlite3.connect(db)

        with con:
            c = con.cursor()
            c.execute('SELECT * FROM TidalConstituents WHERE shortName = \'' + stationName + '\'')
            data = c.fetchall()

        con.close()
    except sqlite3.Error as e:
        if con:
            con.close()
            print('Error %s:' % e.args[0])
            data = [False]

        if noisy:
            print('extraction failed.')

    # Convert data to a dict of value pairs
    siteHarmonics = {}
    tAmp = np.empty(np.shape(data)[0])
    tPhase = np.empty(np.shape(data)[0])
    tSpeed = np.empty(np.shape(data)[0])
    tConst = np.empty(np.shape(data)[0], dtype="|S7")
    tInfer = np.empty(np.shape(data)[0], dtype=bool)
    for i, constituent in enumerate(data):
        tAmp[i] = constituent[1]
        tPhase[i] = constituent[2]
        tSpeed[i] = constituent[3]
        tConst[i] = str(constituent[4])
        if str(constituent[-1]) == 'false':
            tInfer[i] = False
        else:
            tInfer[i] = True
    siteHarmonics['amplitude'] = tAmp
    siteHarmonics['phase'] = tPhase
    siteHarmonics['speed'] = tSpeed
    siteHarmonics['constituentName'] = tConst
    siteHarmonics['inferredConstituent'] = tInfer

    if noisy:
        print('done.')

    return siteHarmonics


def read_POLPRED(harmonics, noisy=False):
    """
    Load a POLPRED data file into a NumPy array. This can then be used by
    get_harmonics_POLPRED to extract the harmonics at a given loaction, or
    otherwise can be used to simply extract the positions of the POLCOMS grid.

    Parameters
    ----------
    harmonics : str
        Full path to the POLPRED ASCII data file.
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    header : dict
        Contains the header data from the POLPRED ASCII file.
    values : ndarray
        Harmonic constituent data formatted as [x, y, nConst * [zZ, zG,
        uZ, uG, vZ, vG]], where nConst is the number of constituents in
        the POLPRED data (15) and z, u and v refer to surface elevation,
        u-vector and v-vector components, respectively. The suffixes Z
        and G refer to amplitude and phase of the z, u and v data.

    See Also
    --------
    tide.grid_POLPRED : Converts the POLPRED data into a rectangular
        gridded data set with values of -999.9 outside the POLPRED domain.

    """

    # Open the harmonics file
    f = open(harmonics, 'r')
    polpred = f.readlines()
    f.close()

    # Read the header into a dict.
    readingHeader = True
    header = {}
    values = []

    if noisy:
        print('Parsing POLPRED raw data...', end=' ')

    for line in polpred:
        if readingHeader:
            if not line.strip():
                # Blank line, which means the end of the header
                readingHeader = False
            else:
                key, parameters = line.split(':')
                header[key.strip()] = parameters.strip()
        else:
            # Remove duplicate whitespaces and split on the resulting
            # single spaces.
            line = line.strip()
            line = ' '.join(line.split())
            values.append(line.split(' '))

    # Make the values into a numpy array
    values = np.asarray(values, dtype=float)

    if noisy:
        print('done.')

    return header, values


def grid_POLPRED(values, noisy=False):
    """
    The POLPRED data are stored as a 2D array, with a single row for each
    location. As such, the lat and long positions are stored in two 1D arrays.
    For the purposes of subsampling, it is much more convenient to have a
    rectangular grid. However, since the POLCOMS model domain is not
    rectangular, it is not possible to simply reshape the POLPRED data.

    To create a rectangular grid, this function builds a lookup table which
    maps locations in the 1D arrays to the equivalent in the 2D array. This is
    achieved as follows:

    1. Create a vector of the unique x and y positions.
    2. Use those positions to search through the 1D array to find the index of
    that position.
    3. Save the 1D index and the 2D indices in a lookup table.
    4. Create a rectangular array whose dimensions match the extent of the
    POLPRED data.
    5. Populate that array with the data, creating a 3D array (x by y by z,
    where z is the number of harmonics).
    6. Use meshgrid to create a rectangular position array (for use with
    pcolor, for example).

    This approach means the grid can be more readily subsampled without the
    need for interpolation (which would change the harmonic constituents).

    Where no data exist (i.e. outside the POLPRED domain), set all values as
    -999.9 (as per POLPRED's land value).

    Parameters
    ----------
    values : ndarray
        Output from read_POLPRED(). See `tide.read_POLPRED'.
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    PX : ndarray
        X values created using np.meshgrid.
    PY : ndarray
        Y values created using np.meshgrid.
    PZ : ndarray
        3D array of harmonic constituent values for the 15 harmonics in
        the POLPRED data at each location in PX and PY. The first two
        dimensions are x and y values (in latitude and longitdue) and
        the third dimension is the amplitude and phases for each of the
        15 constituents for z, u and v data.

    See Also
    --------
    tide.read_POLPRED : Reads in the POLPRED ASCII data.
    tide.get_harmonics_POLPRED : Extract tidal harmonics within a
        threshold distance of a supplied coordinate.

    """

    # Create rectangular arrays of the coordinates in the POLCOMS domain.
    px = np.unique(values[:, 1])
    py = np.unique(values[:, 0])
    PX, PY = np.meshgrid(px, py)

    # I think appending to a list is faster than a NumPy array.
    arridx = []
    for i, (xx, yy) in enumerate(values[:, [1, 0]]):
        if noisy:
            # Only on the first, last and every 1000th line.
            if i == 0 or np.mod(i + 1, 1000) == 0 or i == values[:, 0].shape[0] - 1:
                print('{} of {}'.format(i + 1, np.shape(values)[0]))
        arridx.append([i, px.tolist().index(xx), py.tolist().index(yy)])

    # Now use the lookup table to get the values out of values and into PZ.
    PZ = np.ones([np.shape(py)[0], np.shape(px)[0], np.shape(values)[-1]]) * -999.9
    for idx, xidx, yidx in arridx:
        # Order is the other way around in arridx.
        PZ[yidx, xidx, :] = values[idx, :]

    return PX, PY, PZ


def get_harmonics_POLPRED(harmonics, constituents, lon, lat, stations, noisy=False, distThresh=0.5):
    """
    Function to extract the given constituents at the positions defined
    by lon and lat from a given POLPRED text file.

    The supplied list of names for the stations will be used to generate a
    dict whose structure matches that I've used in the plot_harmonics.py
    script.

    Parameters
    ----------
    harmonics : str
        Full path to the POLPRED ASCII harmonics data.
    constituents : list
        List of tidal constituent names to extract (e.g. ['M2', 'S2']).
    lon, lat : ndarray
        Longitude and latitude positions to find the closest POLPRED
        data point. Uses grid.find_nearest_point to identify the
        closest point. See distThresh below.
    stations : list
        List of tide station names (or coordinates) which are used as
        the keys in the output dict.
    noisy : bool, optional
        Set to True to enable verbose output.
    distThresh : float, optional
        Give a value (in the units of lon and lat) which limits the
        distance to which POLPRED points are returned. Essentially gives
        an upper threshold distance beyond which a point is considered
        not close enough.

    Returns
    -------
    out : dict
        A dict whose keys are the station names. Within each of those
        dicts is another dict whose keys are 'amplitude', 'phase' and
        'constituentName'.
        In addition to the elevation amplitude and phases, the u and v
        amplitudes and phases are also extracted into the dict, with the
        keys 'uH', 'vH', 'uG' and 'vG'.
        Finally, the positions from the POLPRED data is stored with the
        keys 'latitude' and 'longitude'. The length of the arrays within
        each of the secondary dicts is dependent on the number of
        constituents requested.

    See Also
    --------
    tide.read_POLPRED : Read in the POLPRED data to split the ASCII
        file into a header dict and an ndarray of values.
    grid.find_nearest_point : Find the closest point in one set of
        coordinates to a specified point or set of points.

    """

    header, values = read_POLPRED(harmonics, noisy=noisy)

    # Find the nearest points in the POLCOMS grid to the locations
    # requested.
    nearestX, nearestY, distance, index = find_nearest_point(values[:, 1],
                                                             values[:, 0],
                                                             lon,
                                                             lat,
                                                             maxDistance=distThresh)

    # Get a list of the indices from the header for the constituents we're
    # extracting.
    ci = np.empty([np.shape(constituents)[0], 6], dtype=int)
    for i, con in enumerate(constituents):
        tmp = header['Harmonics'].split(' ').index(con)
        # Times 6 because of the columns per constituent
        ci[i, :] = np.repeat(tmp * 6, 6)
        # Add the offsets for the six harmonic components (amplitude and phase
        # of z, u and v).
        ci[i, :] = ci[i, :] + np.arange(6)

    # Plus 3 because of the lat, long and flag columns.
    ci += 3

    # Make a dict of dicts for each station supplied.
    out = {}

    # Find the relevant data for the current site.
    for c, key in enumerate(stations):
        if noisy:
            print('Extracting site {}...'.format(key), end=' ')
            sys.stdout.flush()

        data = {}
        if np.isnan(index[c]):
            if noisy:
                print('skipping (outside domain).')
        else:
            keys = ['amplitude', 'phase', 'uH', 'ug', 'vH', 'vg']
            for n, val in enumerate(keys):
                data[val] = values[index[c], ci[:, n]]

            data['constituentName'] = constituents
            data['latitude'] = values[index[c], 0]
            data['longitude'] = values[index[c], 1]

            out[key] = data

            if noisy:
                print('done.')
                sys.stdout.flush()

    return out


def make_water_column(zeta, h, siglay, **kwargs):
    """
    Calculate the depth time series for cells in an unstructured grid.

    Parameters
    ----------
    zeta : np.ndarray
        Surface elevation time series
    h : np.ndarray
        Water depth
    sigma : np.ndarray
        Sigma level layer thickness, range 0-1 (`siglev' or `siglay')
    nan_invalid : bool, optional
        Set values shallower than the mean sea level (`h') to NaN. Defaults to not doing that.

    Returns
    -------
    z : np.ndarray
        Time series of model depths.

    """

    # This function has been replaced with a call to the more correct PyFVCOM.grid.unstructured_grid_depths and this
    # alias remains for compatibility.
    z = unstructured_grid_depths(h, zeta, siglay, **kwargs)

    # Transpose so the shape is the same as in the old version.
    return z.transpose(1, 0, 2)


class Lanczos(object):
    """
    Create a Lanczos filter object with specific parameters. Pass a time series to filter() to apply that filter to
    the time series.

    Notes
    -----
    This is a python reimplementation of the MATLAB lanczosfilter.m function from
    https://mathworks.com/matlabcentral/fileexchange/14041.

    NaN values are replaced by the mean of the time series and ignored. If you have a better idea, just let me know.

    Reference
    ---------
    Emery, W. J. and R. E. Thomson. "Data Analysis Methods in Physical Oceanography". Elsevier, 2d ed.,
    2004. On pages 533-539.

    """
    def __init__(self, dt=1, cutoff=None, samples=100, passtype='low'):
        """

        Parameters
        ----------
        dt : float, optional
            Sampling interval in minutes. Defaults to 1. (dT in the MATLAB version).
        cutoff : float, optional
            Cutoff frequency in minutes at which to pass data. Defaults to the half the Nyquist frequency. (Cf in the
            MATLAB version).
        samples : int, optional
            Number of samples in the window. Defaults to 100. (M in the MATLAB version)
        passtype : str
            Set the filter to `low' to low-pass (default) or `high' to high-pass. (pass in the MATLAB version).

        """

        self.dt = dt
        self.cutoff = cutoff
        self.samples = samples
        self.passtype = passtype

        if self.passtype == 'low':
            filterindex = 0
        elif self.passtype == 'high':
            filterindex = 1
        else:
            raise ValueError("Specified `passtype' is invalid. Select `high' or `low'.")

        # Nyquist frequency
        self.nyquist_frequency = 1 / (2 * self.dt)
        if not self.cutoff:
            cutoff = self.nyquist_frequency / 2

        # Normalize the cut off frequency with the Nyquist frequency:
        self.cutoff = self.cutoff / self.nyquist_frequency

        # Lanczos cosine coefficients:
        self._lanczos_filter_coef()
        self.coef = self.coef[:, filterindex]

    def _lanczos_filter_coef(self):
        # Positive coefficients of Lanczos [low high]-pass.
        _samples = np.linspace(1, self.samples, self.samples)
        hkcs = self.cutoff * np.array([1] + (np.sin(np.pi * _samples * self.cutoff) / (np.pi * _samples * self.cutoff)).tolist())
        sigma = np.array([1] + (np.sin(np.pi * _samples / self.samples) / (np.pi * _samples / self.samples)).tolist())
        hkB = hkcs * sigma
        hkA = -hkB
        hkA[0] = hkA[0] + 1

        self.coef = np.array([hkB.ravel(), hkA.ravel()]).T

    def _spectral_window(self):
        # Window of cosine filter in frequency space.
        eps = np.finfo(np.float32).eps
        self.Ff = np.arange(0, 1 + eps, 2 / self.N)  # add an epsilon to enclose the stop in the range.
        self.window = np.zeros(len(self.Ff))
        for i in range(len(self.Ff)):
            self.window[i] = self.coef[0] + 2 * np.sum(self.coef[1:] * np.cos((np.arange(1, len(self.coef))) * np.pi * self.Ff[i]))

    def _spectral_filtering(self, x):
        # Filtering in frequency space is multiplication, (convolution in time space).
        Cx = scipy.fft(x.ravel())
        Cx = Cx[:(self.N // 2) + 1]
        CxH = Cx * self.window.ravel()
        # Mirror CxH and append it to itself, dropping the values depending on the length of the input.
        CxH = np.concatenate((CxH, scipy.conj(CxH[1:self.N - len(CxH) + 1][::-1])))
        y = np.real(scipy.ifft(CxH))

        return y

    def filter(self, x):
        """
        Filter the given time series values and return the filtered data.

        Parameters
        ----------
        x : np.ndarray
            Time series values (1D).

        Returns
        -------
        y : np.ndarray
            Filtered time series values (1D).

        """

        # Filter in frequency space:
        self.N = len(x)
        self._spectral_window()
        self.Ff *= self.nyquist_frequency

        # Replace NaNs with the mean (ideas?):
        inan = np.isnan(x)
        if np.any(inan):
            xmean = np.nanmean(x)
            x[inan] = xmean

        # Filtering:
        y = self._spectral_filtering(x)

        # Make sure we've got arrays which match in size.
        if not (len(x) == len(y)):
            raise ValueError('Hmmmm. Fix the arrays!')

        return y


def lanczos(x, dt=1, cutoff=None, samples=100, passtype='low'):
    """
    Apply a Lanczos low- or high-pass filter to a time series.

    Parameters
    ----------
    x : np.ndarray
        1-D times series values.
    dt : float, optional
        Sampling interval. Defaults to 1. (dT in the MATLAB version).
    cutoff : float, optional
        Cutoff frequency in minutes at which to pass data. Defaults to the half the Nyquist frequency. (Cf in the
        MATLAB version).
    samples : int, optional
        Number of samples in the window. Defaults to 100. (M in the MATLAB version)
    passtype : str
        Set the filter to `low' to low-pass (default) or `high' to high-pass. (pass in the MATLAB version).

    Returns
    -------
    y : np.ndarray
        Smoothed time series.
    coef : np.ndarray
        Coefficients of the time window (cosine)
    window : np.ndarray
        Frequency window (aprox. ones for Ff lower(greater) than Fc if low(high)-pass filter and ceros otherwise)
    Cx : np.ndarray
        Complex Fourier Transform of X for Ff frequencies
    Ff : np.ndarray
        Fourier frequencies, from 0 to the Nyquist frequency.

    Notes
    -----
    This is a python reimplementation of the MATLAB lanczosfilter.m function from
    https://mathworks.com/matlabcentral/fileexchange/14041.

    NaN values are replaced by the mean of the time series and ignored. If you have a better idea, just let me know.

    Reference
    ---------
    Emery, W. J. and R. E. Thomson. "Data Analysis Methods in Physical Oceanography". Elsevier, 2d ed.,
    2004. On pages 533-539.

    """

    if passtype == 'low':
        filterindex = 0
    elif passtype == 'high':
        filterindex = 1
    else:
        raise ValueError("Specified `passtype' is invalid. Select `high' or `low'.")

    # Nyquist frequency
    nyquist_frequency = 1 / (2 * dt)
    if not cutoff:
        cutoff = nyquist_frequency / 2

    # Normalize the cut off frequency with the Nyquist frequency:
    cutoff = cutoff / nyquist_frequency

    # Lanczos cosine coefficients:
    coef = _lanczos_filter_coef(cutoff, samples)
    coef = coef[:, filterindex]

    # Filter in frequency space:
    window, Ff = _spectral_window(coef, len(x))
    Ff = Ff * nyquist_frequency

    # Replace NaNs with the mean (ideas?):
    inan = np.isnan(x)
    if np.any(inan):
        xmean = np.nanmean(x)
        x[inan] = xmean

    # Filtering:
    y, Cx = _spectral_filtering(x, window)

    # Make sure we've got arrays which match in size.
    if not (len(x) == len(y)):
        raise ValueError('Hmmmm. Fix the arrays!')

    return y, coef, window, Cx, Ff


def _lanczos_filter_coef(cutoff, samples):
    # Positive coefficients of Lanczos [low high]-pass.
    hkcs = cutoff * np.array([1] + (np.sin(np.pi * np.linspace(1, samples, samples) * cutoff) / (np.pi * np.linspace(1, samples, samples) * cutoff)).tolist())
    sigma = np.array([1] + (np.sin(np.pi * np.linspace(1, samples, samples) / samples) / (np.pi * np.linspace(1, samples, samples) / samples)).tolist())
    hkB = hkcs * sigma
    hkA = -hkB
    hkA[0] = hkA[0] + 1
    coef = np.array([hkB.ravel(), hkA.ravel()]).T

    return coef


def _spectral_window(coef, N):
    # Window of cosine filter in frequency space.
    eps = np.finfo(np.float32).eps
    Ff = np.arange(0, 1 + eps, 2 / N)  # add an epsilon to enclose the stop in the range.
    window = np.zeros(len(Ff))
    for i in range(len(Ff)):
        window[i] = coef[0] + 2 * np.sum(coef[1:] * np.cos((np.arange(1, len(coef))) * np.pi * Ff[i]))

    return window, Ff


def _spectral_filtering(x, window):
    # Filtering in frequency space is multiplication, (convolution in time space).
    Nx = len(x)
    Cx = scipy.fft(x.ravel())
    Cx = Cx[:(Nx // 2) + 1]
    CxH = Cx * window.ravel()
    # Mirror CxH and append it to itself, dropping the values depending on the length of the input.
    CxH = np.concatenate((CxH, scipy.conj(CxH[1:Nx-len(CxH)+1][::-1])))
    y = np.real(scipy.ifft(CxH))
    return y, Cx

