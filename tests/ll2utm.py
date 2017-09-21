import numpy.testing as test
import numpy as np

from unittest import TestCase

from PyFVCOM.ll2utm import *

class CoordinateToolsTest(TestCase):

    def setUp(self):
        """
        Make a set of spherical and cartesian coordinates. These have been converted from spherical to UTM zone 30N
        with the following invproj command:

        proj -f %.10f +proj=utm +ellps=WGS84 +zone=30 < test_lon_lat.txt

        """
        self.lon = np.array((-3.15921600, -3.16311600, -3.16591800, -3.16787800, -3.16893900, -3.16922600, -3.16890000, -3.16815900, -3.16722200, -3.16629900, -3.16553900, -3.16497400, -3.16450200, -3.16379300, -3.16253700, -3.17264000, -3.18184600, -3.19447000, -3.19229200, -3.20210600, -3.21187000, -3.21844500, -3.23075300, -3.24251000, -3.24355800, -3.24279400, -3.24572200, -3.25530500, -3.26385600, -3.27169900, -3.27898400, -3.28557100))
        self.lat = np.array((53.7248050, 53.7374580, 53.7505550, 53.7637230, 53.7769670, 53.7902190, 53.8034330, 53.8165890, 53.8296910, 53.8427540, 53.8557920, 53.8688170, 53.8818290, 53.8948030, 53.9076400, 53.9144000, 53.9250040, 53.9348120, 53.9464060, 53.9569400, 53.9683810, 53.9797360, 53.9828180, 53.9876820, 53.9996820, 54.0113280, 54.0244950, 54.0334370, 54.0417460, 54.0503130, 54.0591300, 54.0682910))
        self.x = np.array((489494.7558550927, 489240.6610005523, 489059.2402357359, 488933.4584639949, 488867.0218582982, 488851.6220290385, 488876.5955025404, 488928.8632107785, 488993.9868573219, 489058.1412829786, 489111.5305387266, 489152.0636311335, 489186.4575025685, 489236.3972546348, 489322.2085209083, 488660.3286343530, 488058.6693292787, 487232.6804035276, 487379.1697867822, 486738.3842831449, 486101.5057900152, 485674.0874244687, 484868.0307794868, 484098.9022559610, 484034.7774409536, 484089.2999873500, 483902.5076042408, 483278.3048510661, 482721.6876615641, 482211.7577387817, 481738.6741044795, 481311.6263004461))
        self.y = np.array((5952915.8072033720, 5954324.1044408595, 5955781.6465004245, 5957246.9665906448, 5958720.6075669480, 5960195.0210704524, 5961665.1139462870, 5963128.6924875183, 5964586.2365010530, 5966039.4476912562, 5967489.9064548314, 5968938.9524050616, 5970386.5696569690, 5971829.9264751300, 5973257.9619816393, 5974011.6393185537, 5975192.9394670632, 5976286.3678789986, 5977575.9167234059, 5978749.7152408855, 5980024.5072707040, 5981289.1715462450, 5981634.6335958010, 5982178.3772334624, 5983513.7336512301, 5984809.2988616070, 5986274.9277136009, 5987272.0420628237, 5988198.5618544118, 5989153.6773506571, 5990136.5218878873, 5991157.5058998121))

    def test_utm_from_lonlat_numpy_single_value_no_zone(self):
        eastings, northings, _ = utm_from_lonlat(self.lon[0], self.lat[0])
        test.assert_almost_equal(eastings, self.x[0])
        test.assert_almost_equal(northings, self.y[0], decimal=4)

    def test_utm_from_lonlat_numpy_single_value_fixed_zone(self):
        eastings, northings, _ = utm_from_lonlat(self.lon[0], self.lat[0], zone='30N')
        test.assert_almost_equal(eastings, self.x[0])
        test.assert_almost_equal(northings, self.y[0], decimal=5)

    def test_utm_from_lonlat_numpy_array_no_zone(self):
        eastings, northings, _ = utm_from_lonlat(self.lon, self.lat)
        test.assert_almost_equal(eastings, self.x)
        test.assert_almost_equal(northings, self.y, decimal=5)

    def test_utm_from_lonlat_numpy_array_fixed_zone(self):
        eastings, northings, _ = utm_from_lonlat(self.lon, self.lat, zone='30N')
        test.assert_almost_equal(eastings, self.x)
        test.assert_almost_equal(northings, self.y, decimal=5)

    def test_utm_from_lonlat_list_no_zone(self):
        eastings, northings, _ = utm_from_lonlat(self.lon, self.lat)
        test.assert_almost_equal(eastings, self.x)
        test.assert_almost_equal(northings, self.y, decimal=5)

    def test_utm_from_lonlat_list_fixed_zone(self):
        eastings, northings, _ = utm_from_lonlat(self.lon.tolist(), self.lat.tolist(), zone='30N')
        test.assert_almost_equal(eastings, self.x.tolist())
        test.assert_almost_equal(northings, self.y.tolist(), decimal=4)

    def test_lonlat_from_utm_numpy_single_value_fixed_zone(self):
        lon, lat = lonlat_from_utm(self.x, self.y, zone='30N')
        test.assert_almost_equal(lon, self.lon)
        test.assert_almost_equal(lat, self.lat)

    def test_lonlat_from_utm_numpy_single_value(self):
        lon, lat = lonlat_from_utm(self.x[0], self.y[0], '30N')
        test.assert_almost_equal(lon, self.lon[0])
        test.assert_almost_equal(lat, self.lat[0])

    def test_lonlat_from_utm_numpy_array_fixed_zone(self):
        lon, lat = lonlat_from_utm(self.x, self.y, '30N')
        test.assert_almost_equal(lon, self.lon)
        test.assert_almost_equal(lat, self.lat)

    def test_lonlat_from_utm_list_no_zone(self):
        lon, lat = lonlat_from_utm(self.x, self.y, '30N')
        test.assert_almost_equal(lon, self.lon)
        test.assert_almost_equal(lat, self.lat)

    def test_lonlat_from_utm_list_fixed_zone(self):
        lon, lat = lonlat_from_utm(self.x.tolist(), self.y.tolist(), '30N')
        test.assert_almost_equal(lon, self.lon.tolist())
        test.assert_almost_equal(lat, self.lat.tolist())

    def test_lonlat_from_utm_numpy_single_value_fixed_zone(self):
        lon, lat = lonlat_from_utm(self.x, self.y, '30N')
        test.assert_almost_equal(lon, self.lon)
        test.assert_almost_equal(lat, self.lat)
