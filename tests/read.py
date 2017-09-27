import os

import numpy.testing as test
import numpy as np

from unittest import TestCase
from datetime import datetime
from dateutil.relativedelta import relativedelta

from PyFVCOM.read import FileReader
from PyFVCOM.utilities import StubFile


class FileReader_test(TestCase):

    def setUp(self):
        self.starttime, self.endtime, self.interval, self.lon, self.lat, self.triangles = _prep()
        self.stub = StubFile(self.starttime, self.endtime, self.interval,
                             lon=self.lon, lat=self.lat, triangles=self.triangles, zone='30N')

    def tearDown(self):
        del(self.stub)

    def test_get_single_lon(self):
        lon = -5.78687373
        F = FileReader(self.stub.ncfile.name, dims={'node': [0]})
        test.assert_almost_equal(F.grid.lon, lon, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_single_lat(self):
        lat = 56.89470906
        F = FileReader(self.stub.ncfile.name, dims={'node': [29]})
        test.assert_almost_equal(F.grid.lat, lat, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_single_lonc(self):
        lonc = -4.67915533
        F = FileReader(self.stub.ncfile.name, dims={'nele': [0]})
        test.assert_almost_equal(F.grid.lonc, lonc, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_single_latc(self):
        latc = 52.864905897403958
        F = FileReader(self.stub.ncfile.name, dims={'nele': [29]})
        test.assert_almost_equal(F.grid.latc, latc, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_multipe_lon(self):
        lon = np.array((-5.78687373, -3.26585943))
        F = FileReader(self.stub.ncfile.name, dims={'node': [0, 5]})
        test.assert_almost_equal(F.grid.lon, lon, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_multipe_lat(self):
        lat = np.array((56.89470906, 58.49899088))
        F = FileReader(self.stub.ncfile.name, dims={'node': [29, 34]})
        test.assert_almost_equal(F.grid.lat, lat, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_multipe_lonc(self):
        lonc = np.array((-4.67915533, 0.61115498))
        F = FileReader(self.stub.ncfile.name, dims={'nele': [0, 5]})
        test.assert_almost_equal(F.grid.lonc, lonc, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_multipe_latc(self):
        latc = np.array((52.8649059 , 52.90310308))
        F = FileReader(self.stub.ncfile.name, dims={'nele': [29, 34]})
        test.assert_almost_equal(F.grid.latc, latc, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_water_column(self):
        water_column = np.array([
            8.18035714, 7.27142857, 6.3625, 5.45357143, 4.54464286,
            3.63571429, 2.72678571, 1.81785714, 0.90892857, 0.0])
        F = FileReader(self.stub.ncfile.name, dims={'node': [5], 'time': [10, 11]}, variables=['temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), water_column, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_time_series(self):
        surface_elevation = np.array([
            -1.41013891, -0.98554639, -0.3139296, 0.4363727, 1.07729949,
            1.44820437, 1.45612114, 1.09906549, 0.46653234, -0.28293574,
            -0.96148683, -1.3990441, -1.48593514, -1.20038097, -0.61395488,
            0.12635717, 0.83499817, 1.33434937, 1.49924987, 1.28836787,
            0.75456029, 0.03162443, -0.699238, -1.25483851, -1.49591749,
            -1.3620492, -0.88678734, -0.18925488, 0.55571376, 1.1613944])
        F = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': [10, 40]}, variables=['zeta'])
        test.assert_almost_equal(np.squeeze(F.data.zeta), surface_elevation, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

    def test_get_layer(self):
        vertical_velocity = np.array([
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454, 0.00377454,
            0.00377454, 0.00377454, 0.00377454, 0.00377454])
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'time': [100, 101]}, variables=['ww'])
        test.assert_almost_equal(np.squeeze(F.data.ww), vertical_velocity, decimal=5)
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)

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
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        F += FileReader(next_stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])

        test.assert_equal(F.time.datetime, all_times)
        test.assert_equal(F.data.ww, all_data)

        # Tidy up.
        self.stub.ncfile.close()
        next_stub.ncfile.close()
        os.remove(self.stub.ncfile.name)
        os.remove(next_stub.ncfile.name)


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
        [67, 66, 1], [65, 2, 66], [ 1, 66, 2], [64, 2, 65], [63, 3, 64],
        [60, 59, 57], [ 2, 64, 3], [ 3, 63, 4], [ 0, 67, 1], [62, 4, 63],
        [57, 59, 56], [59, 58, 56], [61, 60, 69], [57, 69, 60], [ 4, 62, 68],
        [ 6, 5, 9], [61, 68, 62], [69, 68, 61], [ 9, 5, 70], [ 6, 8, 7],
        [ 4, 70, 5], [ 8, 6, 9], [56, 69, 57], [69, 56, 52], [70, 10, 9],
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
