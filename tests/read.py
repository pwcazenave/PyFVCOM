import os

import numpy.testing as test
import numpy as np
import tempfile

from unittest import TestCase
from datetime import datetime
from dateutil.relativedelta import relativedelta

from PyFVCOM.read import FileReader
from PyFVCOM.grid import nodes2elems
from PyFVCOM.coordinate import utm_from_lonlat
from PyFVCOM.utilities.time import date_range
from netCDF4 import Dataset, date2num


class StubFile(object):
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
        # to use create_variable() here, but I'm not sure I like the idea of spamming self with loads of arrays.
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


class FileReader_test(TestCase):

    def setUp(self):
        self.starttime, self.endtime, self.interval, self.lon, self.lat, self.triangles = _prep()
        self.stub = StubFile(self.starttime, self.endtime, self.interval,
                             lon=self.lon, lat=self.lat, triangles=self.triangles, zone='30N')
        self.reference = FileReader(self.stub.ncfile.name, variables=['ww', 'zeta', 'temp', 'h'])

    def tearDown(self):
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)
        del(self.stub)

    def test_get_single_lon(self):
        result = FileReader(self.stub.ncfile.name, dims={'node': [0]})
        test.assert_almost_equal(result.grid.lon, self.reference.grid.lon[0], decimal=5)

    def test_get_single_lat(self):
        result = FileReader(self.stub.ncfile.name, dims={'node': [29]})
        test.assert_almost_equal(result.grid.lat, self.reference.grid.lat[29], decimal=5)

    def test_get_single_lonc(self):
        result = FileReader(self.stub.ncfile.name, dims={'nele': [0]})
        test.assert_almost_equal(result.grid.lonc, self.reference.grid.lonc[0], decimal=5)

    def test_get_single_latc(self):
        F = FileReader(self.stub.ncfile.name, dims={'nele': [29]})
        test.assert_almost_equal(F.grid.latc, self.reference.grid.latc[29], decimal=5)

    def test_get_multipe_lon(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [0, 5]})
        test.assert_almost_equal(F.grid.lon, self.reference.grid.lon[[0, 5]], decimal=5)

    def test_get_multipe_lat(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [29, 34]})
        test.assert_almost_equal(F.grid.lat, self.reference.grid.lat[[29, 34]], decimal=5)

    def test_get_multipe_lonc(self):
        F = FileReader(self.stub.ncfile.name, dims={'nele': [0, 5]})
        test.assert_almost_equal(F.grid.lonc, self.reference.grid.lonc[[0, 5]], decimal=5)

    def test_get_multipe_latc(self):
        F = FileReader(self.stub.ncfile.name, dims={'nele': [29, 34]})
        test.assert_almost_equal(F.grid.latc, self.reference.grid.latc[[29, 34]], decimal=5)

    # def test_get_bounding_box(self):
    #     wesn = [-5, -3, 50, 55]
    #     extents = [-4.9847326278686523, -3.0939722061157227,
    #                50.19110107421875, 54.946651458740234]
    #     F = FileReader(self.stub.ncfile.name, dims={'wesn': wesn})
    #     test.assert_equal(F.grid.lon.min(), extents[0])
    #     test.assert_equal(F.grid.lon.max(), extents[1])
    #     test.assert_equal(F.grid.lat.min(), extents[2])
    #     test.assert_equal(F.grid.lat.max(), extents[3])

    def test_get_water_column(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [5], 'time': 10}, variables=['temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), self.reference.data.temp[10, :, 5], decimal=5)

    def test_get_time_series(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': np.arange(10, 40)}, variables=['zeta'])
        test.assert_almost_equal(np.squeeze(F.data.zeta), self.reference.data.zeta[10:40, 10], decimal=5)

    def test_get_negative_time_series(self):
        F1 = FileReader(self.stub.ncfile.name, dims={'node': [10]}, variables=['zeta'])
        F2 = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': -np.arange(10, 40)}, variables=['zeta'])
        test.assert_almost_equal(F2.data.zeta, F1.data.zeta[-np.arange(10, 40)], decimal=5)

    def test_get_single_time(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': [10]}, variables=['zeta'])
        test.assert_almost_equal(np.squeeze(F.data.zeta), self.reference.data.zeta[10, 10], decimal=5)

    def test_get_single_time_negative_index(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': [-10]}, variables=['zeta'])
        test.assert_almost_equal(np.squeeze(F.data.zeta), self.reference.data.zeta[-10, 10], decimal=5)

    def test_get_layer(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5]}, variables=['ww'])
        test.assert_almost_equal(np.squeeze(F.data.ww), self.reference.data.ww[:, 5, :], decimal=5)

    def test_get_layer_get_nodes(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'node': np.arange(4)}, variables=['temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), self.reference.data.temp[:, 5, :4], decimal=5)

    def test_get_layer_get_nodes_get_elements(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'node': np.arange(4), 'nele': np.arange(3)}, variables=['ww', 'temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), self.reference.data.temp[:, 5, :4], decimal=5)
        test.assert_almost_equal(np.squeeze(F.data.ww), self.reference.data.ww[:, 5, :3], decimal=5)

    def test_get_layer_get_level_get_nodes_get_elements(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'siglev': [4], 'node': np.arange(4), 'nele': np.arange(3)}, variables=['ww', 'temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), self.reference.data.temp[:, 5, :4], decimal=5)
        test.assert_almost_equal(np.squeeze(F.data.ww), self.reference.data.ww[:, 5, :3], decimal=5)

    def test_get_layer_no_variable(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': np.arange(0, 10, 2)})
        test.assert_almost_equal(F.grid.siglay, self.reference.grid.siglay[0:10:2])

    def test_get_level_no_variable(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglev': np.arange(0, 11, 2)})
        test.assert_almost_equal(F.grid.siglev, self.reference.grid.siglev[0:11:2])

    def test_non_temporal_variable(self):
        h = np.asarray([1.64808428, 12.75706577, 18.34670639, 24.29236031,
                        29.7772541, 25.00211716, 22.69193077, 18.70510674,
                        21.96312141, 27.35856438, 35.32657623, 32.48567581,
                        38.93023682, 43.63704681, 51.21723175, 53.23581314,
                        59.78393555, 55.53053284, 52.84440994, 57.36302185,
                        62.2620163, 66.50558472, 61.24137878, 60.91600418,
                        67.42472839, 73.38938904, 70.63117981, 70.62969208,
                        75.18034363, 79.09741974, 84.4043808 , 81.0752182,
                        88.22835541, 90.34424591, 97.57055664, 98.27231598,
                        100.0000000, 96.82516479, 91.1933136 , 88.29994202,
                        89.59196472, 91.40013885, 85.90748596, 79.28456879,
                        74.37998199, 70.46596527, 70.78884888, 70.06604004,
                        63.42258453, 63.06575394, 59.99647141, 57.27880096,
                        55.11286545, 61.5132103, 62.31158066, 59.2288208,
                        53.60129929, 50.73873138, 56.42451477, 52.42653656,
                        44.78648376, 39.55376434, 32.51250839, 28.38024521,
                        20.91413689, 18.19268227, 11.62014961, 7.51470757,
                        38.44644928, 45.77177048, 34.9041214, 51.38194275,
                        77.87741852, 81.04411316])
        F = FileReader(self.stub.ncfile.name, variables=['h'])
        test.assert_almost_equal(F.data.h, h)

    def test_non_temporal_variable_with_dimension(self):
        h = np.asarray([1.64808428, 12.75706577, 18.34670639, 24.29236031])
        F = FileReader(self.stub.ncfile.name, variables=['h'], dims={'node': np.arange(4)})
        test.assert_almost_equal(F.data.h, h)

    def test_add_files(self):
        # Make another stub file which follows in time from the existing one. Then only load a section of that in
        # time and make sure the results are the same as if we'd loaded them manually and added them together.
        next_stub = StubFile(self.endtime, self.endtime + relativedelta(months=1), self.interval,
                             lon=self.lon, lat=self.lat, triangles=self.triangles, zone='30N')

        # Append the new stub file to the old one.
        F1 = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        F2 = FileReader(next_stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        all_times = np.concatenate((F1.time.datetime[:], F2.time.datetime[:]), axis=0)
        all_data = np.concatenate((F1.data.ww[:], F2.data.ww[:]), axis=0)
        # Repeat the process, but use the __add__ method in FileReader.
        F1 = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        F2 = FileReader(next_stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        F = F2 >> F1

        test.assert_equal(F.time.datetime, all_times)
        test.assert_equal(F.data.ww, all_data)

    def test_get_time_with_string(self):
        time_dims = ['2001-02-12 09:00:00.00000', '2001-02-14 12:00:00.00000']
        returned_indices = np.arange(26, 77)

        F = FileReader(self.stub.ncfile.name, dims={'time': time_dims})
        test.assert_equal(F._dims['time'], returned_indices)

    def test_get_time_with_datetime(self):
        time_dims = [datetime.strptime('2001-02-12 09:00:00.00000', '%Y-%m-%d %H:%M:%S.%f'),
                     datetime.strptime('2001-02-14 12:00:00.00000', '%Y-%m-%d %H:%M:%S.%f')]
        returned_indices = [26, 76]

        F = FileReader(self.stub.ncfile.name, dims={'time': time_dims})
        test.assert_equal(F._dims['time'][0], returned_indices[0])
        test.assert_equal(F._dims['time'][-1], returned_indices[-1])

    def test_get_time_with_tolerance(self):
        time_dims = [datetime.strptime('2001-02-12 09:00:00.00000', '%Y-%m-%d %H:%M:%S.%f'),
                     datetime.strptime('2001-02-12 09:14:02.00000', '%Y-%m-%d %H:%M:%S.%f')]
        returned_indices = [None, 26]

        F = FileReader(self.stub.ncfile.name)
        file_indices = [F.time_to_index(i, tolerance=10) for i in time_dims]
        test.assert_equal(file_indices, returned_indices)


def _prep(starttime=None, duration=None, interval=None):
    """
    Make some input data (a grid and a time range).

    Parameters
    ----------
    starttime : datetime.datetime, optional
        Provide a start time from which to create the time range. Defaults to '2001-02-11 07:14:02'.
    duration : dateutil.relativedelta, optional
        Give a duration for the time range. Defaults to a month.
    interval : float, optional
        Sampling interval in days. Defaults to hourly.

    Returns
    -------
    starttime, endtime : datetime.datetime
        Start and end times.
    interval : float
        Sampling interval for the netCDF stub.
    lon, lat : np.ndarray
        Longitude and latitudes for the grid.
    triangles : np.ndarray
        Triangulation table for the grid.

    Notes
    -----
    The triangulation is lifted from the matplotlib.triplot demo:
       https://matplotlib.org/examples/pylab_examples/triplot_demo.html

    """

    xy = np.asarray([
        [-0.101, 0.872], [-0.080, 0.883], [-0.069, 0.888], [-0.054, 0.890],
        [-0.045, 0.897], [-0.057, 0.895], [-0.073, 0.900], [-0.087, 0.898],
        [-0.090, 0.904], [-0.069, 0.907], [-0.069, 0.921], [-0.080, 0.919],
        [-0.073, 0.928], [-0.052, 0.930], [-0.048, 0.942], [-0.062, 0.949],
        [-0.054, 0.958], [-0.069, 0.954], [-0.087, 0.952], [-0.087, 0.959],
        [-0.080, 0.966], [-0.085, 0.973], [-0.087, 0.965], [-0.097, 0.965],
        [-0.097, 0.975], [-0.092, 0.984], [-0.101, 0.980], [-0.108, 0.980],
        [-0.104, 0.987], [-0.102, 0.993], [-0.115, 1.001], [-0.099, 0.996],
        [-0.101, 1.007], [-0.090, 1.010], [-0.087, 1.021], [-0.069, 1.021],
        [-0.052, 1.022], [-0.052, 1.017], [-0.069, 1.010], [-0.064, 1.005],
        [-0.048, 1.005], [-0.031, 1.005], [-0.031, 0.996], [-0.040, 0.987],
        [-0.045, 0.980], [-0.052, 0.975], [-0.040, 0.973], [-0.026, 0.968],
        [-0.020, 0.954], [-0.006, 0.947], [ 0.003, 0.935], [ 0.006, 0.926],
        [ 0.005, 0.921], [ 0.022, 0.923], [ 0.033, 0.912], [ 0.029, 0.905],
        [ 0.017, 0.900], [ 0.012, 0.895], [ 0.027, 0.893], [ 0.019, 0.886],
        [ 0.001, 0.883], [-0.012, 0.884], [-0.029, 0.883], [-0.038, 0.879],
        [-0.057, 0.881], [-0.062, 0.876], [-0.078, 0.876], [-0.087, 0.872],
        [-0.030, 0.907], [-0.007, 0.905], [-0.057, 0.916], [-0.025, 0.933],
        [-0.077, 0.990], [-0.059, 0.993]])
    lon = np.degrees(xy[:, 0])
    lat = np.degrees(xy[:, 1])
    triangles = np.asarray([
        [67, 66, 1], [65, 2, 66], [1, 66, 2], [64, 2, 65], [63, 3, 64],
        [60, 59, 57], [2, 64, 3], [3, 63, 4], [0, 67, 1], [62, 4, 63],
        [57, 59, 56], [59, 58, 56], [61, 60, 69], [57, 69, 60], [4, 62, 68],
        [6, 5, 9], [61, 68, 62], [69, 68, 61], [9, 5, 70], [6, 8, 7],
        [4, 70, 5], [8, 6, 9], [56, 69, 57], [69, 56, 52], [70, 10, 9],
        [54, 53, 55], [56, 55, 53], [68, 70, 4], [52, 56, 53], [11, 10, 12],
        [69, 71, 68], [68, 13, 70], [10, 70, 13], [51, 50, 52], [13, 68, 71],
        [52, 71, 69], [12, 10, 13], [71, 52, 50], [71, 14, 13], [50, 49, 71],
        [49, 48, 71], [14, 16, 15], [14, 71, 48], [17, 19, 18], [17, 20, 19],
        [48, 16, 14], [48, 47, 16], [47, 46, 16], [16, 46, 45], [23, 22, 24],
        [21, 24, 22], [17, 16, 45], [20, 17, 45], [21, 25, 24], [27, 26, 28],
        [20, 72, 21], [25, 21, 72], [45, 72, 20], [25, 28, 26], [44, 73, 45],
        [72, 45, 73], [28, 25, 29], [29, 25, 31], [43, 73, 44], [73, 43, 40],
        [72, 73, 39], [72, 31, 25], [42, 40, 43], [31, 30, 29], [39, 73, 40],
        [42, 41, 40], [72, 33, 31], [32, 31, 33], [39, 38, 72], [33, 72, 38],
        [33, 38, 34], [37, 35, 38], [34, 38, 35], [35, 37, 36]])

    if not starttime:
        starttime = datetime.strptime('2001-02-11 07:14:02', '%Y-%m-%d %H:%M:%S')

    if duration:
        endtime = starttime + duration
    else:
        endtime = starttime + relativedelta(months=1)

    if not interval:
        interval = 1.0 / 24.0

    return starttime, endtime, interval, lon, lat, triangles
