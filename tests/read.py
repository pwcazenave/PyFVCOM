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
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)
        del(self.stub)

    def test_get_single_lon(self):
        lon = -5.78687373
        F = FileReader(self.stub.ncfile.name, dims={'node': [0]})
        test.assert_almost_equal(F.grid.lon, lon, decimal=5)

    def test_get_single_lat(self):
        lat = 56.89470906
        F = FileReader(self.stub.ncfile.name, dims={'node': [29]})
        test.assert_almost_equal(F.grid.lat, lat, decimal=5)

    def test_get_single_lonc(self):
        lonc = -4.67915533
        F = FileReader(self.stub.ncfile.name, dims={'nele': [0]})
        test.assert_almost_equal(F.grid.lonc, lonc, decimal=5)

    def test_get_single_latc(self):
        latc = 52.864905897403958
        F = FileReader(self.stub.ncfile.name, dims={'nele': [29]})
        test.assert_almost_equal(F.grid.latc, latc, decimal=5)

    def test_get_multipe_lon(self):
        lon = np.array((-5.78687373, -3.26585943))
        F = FileReader(self.stub.ncfile.name, dims={'node': [0, 5]})
        test.assert_almost_equal(F.grid.lon, lon, decimal=5)

    def test_get_multipe_lat(self):
        lat = np.array((56.89470906, 58.49899088))
        F = FileReader(self.stub.ncfile.name, dims={'node': [29, 34]})
        test.assert_almost_equal(F.grid.lat, lat, decimal=5)

    def test_get_multipe_lonc(self):
        lonc = np.array((-4.67915533, 0.61115498))
        F = FileReader(self.stub.ncfile.name, dims={'nele': [0, 5]})
        test.assert_almost_equal(F.grid.lonc, lonc, decimal=5)

    def test_get_multipe_latc(self):
        latc = np.array((52.8649059, 52.90310308))
        F = FileReader(self.stub.ncfile.name, dims={'nele': [29, 34]})
        test.assert_almost_equal(F.grid.latc, latc, decimal=5)

    def test_get_bounding_box(self):
        wesn = [-5, -3, 50, 55]
        extents = [-4.9847326278686523, -3.0939722061157227,
                   50.19110107421875, 54.946651458740234]
        F = FileReader(self.stub.ncfile.name, dims={'wesn': wesn})
        test.assert_equal(F.grid.lon.min(), extents[0])
        test.assert_equal(F.grid.lon.max(), extents[1])
        test.assert_equal(F.grid.lat.min(), extents[2])
        test.assert_equal(F.grid.lat.max(), extents[3])

    def test_get_water_column(self):
        water_column = np.array([
            9.99821472, 10.90714264, 11.81607151, 12.72500038, 13.6339283,
            14.54285717, 15.45178604, 16.36071396, 17.26964378, 18.1785717
        ])
        F = FileReader(self.stub.ncfile.name, dims={'node': [5], 'time': [10, 11]}, variables=['temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), water_column, decimal=5)

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

    def test_get_single_time(self):
        surface_elevation = np.array([-1.41013891])
        F = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': [10, 11]}, variables=['zeta'])
        test.assert_almost_equal(np.squeeze(F.data.zeta), surface_elevation, decimal=5)

    def test_get_single_time_negative_index(self):
        surface_elevation = np.array([[0.22058453]])
        F = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': [-10, -9]}, variables=['zeta'])
        test.assert_almost_equal(np.squeeze(F.data.zeta), surface_elevation, decimal=5)

    def test_get_layer(self):
        vertical_velocity = np.array([
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815, 0.01509815,
            0.01509815, 0.01509815, 0.01509815, 0.01509815])
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'time': [100, 101]}, variables=['ww'])
        test.assert_almost_equal(np.squeeze(F.data.ww), vertical_velocity, decimal=5)

    def test_get_layer_no_variable(self):
        siglay = -np.tile(np.arange(0.05, 1, 0.2), [len(self.lon), 1]).T
        F = FileReader(self.stub.ncfile.name, dims={'siglay': np.arange(0, 10, 2)})
        test.assert_almost_equal(F.grid.siglay, siglay)

    def test_get_level_no_variable(self):
        siglev = -np.tile(np.arange(0, 1.2, 0.2), [len(self.lon), 1]).T
        F = FileReader(self.stub.ncfile.name, dims={'siglev': np.arange(0, 11, 2)})
        test.assert_almost_equal(F.grid.siglev, siglev)

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
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        F += FileReader(next_stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])

        test.assert_equal(F.time.datetime, all_times)
        test.assert_equal(F.data.ww, all_data)

    def test_get_time_with_string(self):
        time_dims = ['2001-02-12 09:00:00.00000', '2001-02-14 12:00:00.00000']
        returned_indices = [26, 78]

        F = FileReader(self.stub.ncfile.name, dims={'time': time_dims})
        test.assert_equal(F._dims['time'], returned_indices)

    def test_get_time_with_datetime(self):
        time_dims = [datetime.strptime('2001-02-12 09:00:00.00000', '%Y-%m-%d %H:%M:%S.%f'),
                     datetime.strptime('2001-02-14 12:00:00.00000', '%Y-%m-%d %H:%M:%S.%f')]
        returned_indices = [26, 78]

        F = FileReader(self.stub.ncfile.name, dims={'time': time_dims})
        test.assert_equal(F._dims['time'][0], returned_indices[0])

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
