import os
import numpy.testing as test
import numpy as np

from unittest import TestCase

from PyFVCOM.read import FileReader
from PyFVCOM.coordinate import utm_from_lonlat
from PyFVCOM.current import *
from PyFVCOM.utilities.grid import StubFile
from read import _prep


class Residuals_test(TestCase):

    def setUp(self):
        self.starttime, self.endtime, self.interval, self.lon, self.lat, self.triangles = _prep()
        self.model = StubFile(self.starttime,
                              self.endtime,
                              self.interval,
                              lon=self.lon,
                              lat=self.lat,
                              triangles=self.triangles,
                              zone='30N')
        self.fvcom = FileReader(self.model.ncfile.name, variables=['u', 'v', 'ua', 'va'])
        self.fvcom.grid.x, self.fvcom.grid.y, _ = utm_from_lonlat(self.fvcom.grid.lon, self.fvcom.grid.lat, zone='30N')
        self.fvcom.grid.xc, self.fvcom.grid.yc, _ = utm_from_lonlat(self.fvcom.grid.lonc, self.fvcom.grid.latc, zone='30N')

    def tearDown(self):
        self.model.ncfile.close()
        os.remove(self.model.ncfile.name)
        del(self.model)

    def test_depth_resolved_vorticity(self):
        expected_vorticity = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                       -1.69406589e-21, np.nan, np.nan, -1.64608942e-22, 6.08804931e-22, np.nan, np.nan,
                                       -6.61744490e-23, -3.97046694e-23, np.nan, np.nan, np.nan, 7.01449159e-22,
                                       1.32348898e-22, np.nan, np.nan, np.nan, -1.19114008e-22, np.nan, np.nan,
                                       4.23516474e-22, 1.32348898e-22, -1.32348898e-23, np.nan, 1.05879118e-22,
                                       4.76456033e-22, np.nan, -2.11758237e-22, np.nan, np.nan, np.nan, np.nan,
                                       2.51462906e-22, np.nan, np.nan, -1.16467030e-21, np.nan, np.nan, np.nan, np.nan,
                                       np.nan, np.nan, -4.76456033e-22, np.nan, np.nan, np.nan, -6.61744490e-24,
                                       2.64697796e-22, np.nan, np.nan, 2.11758237e-22, np.nan, -2.64697796e-23, np.nan,
                                       0.00000000e+00, 5.55865372e-22, -1.58818678e-22, np.nan, np.nan, np.nan, np.nan,
                                       3.17637355e-22, np.nan, np.nan, 4.23516474e-22, np.nan, np.nan, np.nan, np.nan])
        vort = vorticity(self.fvcom)
        # Only check the first time and vertical layer to make my life easier.
        test.assert_almost_equal(expected_vorticity, vort[0, 0, :])

    def test_depth_averaged_vorticity(self):
        expected_vorticity = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                       -1.21760986e-21, np.nan, np.nan, -1.05879118e-22, 3.27563523e-22, np.nan,
                                       np.nan, -1.32348898e-23, -4.63221143e-23, np.nan, np.nan, np.nan, 5.95570041e-22,
                                       3.97046694e-22, np.nan, np.nan, np.nan, 0.00000000e+00, np.nan, np.nan,
                                       2.38228016e-22, 2.64697796e-23, -2.97785021e-23, np.nan, 1.05879118e-22,
                                       2.11758237e-22, np.nan, 5.29395592e-23, np.nan, np.nan, np.nan, np.nan,
                                       2.64697796e-23, np.nan, np.nan, -5.82335151e-22, np.nan, np.nan, np.nan, np.nan,
                                       np.nan, np.nan, -1.85288457e-22, np.nan, np.nan, np.nan, -8.27180613e-24,
                                       2.64697796e-23, np.nan, np.nan, 0.00000000e+00, np.nan, -6.61744490e-23, np.nan,
                                       2.64697796e-23, 6.61744490e-23, -1.05879118e-22, np.nan, np.nan, np.nan, np.nan,
                                       2.64697796e-23, np.nan, np.nan, 1.58818678e-22, np.nan, np.nan, np.nan, np.nan])
        vort = vorticity(self.fvcom, depth_averaged=True)
        test.assert_almost_equal(expected_vorticity, vort[0, :])
