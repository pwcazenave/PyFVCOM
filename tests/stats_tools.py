import numpy.testing as test
import numpy as np

from datetime import datetime
from netCDF4 import num2date, date2num

from unittest import TestCase

from PyFVCOM.stats_tools import *

class StatsToolsTest(TestCase):

    def setUp(self):
        """ Make a couple of time series """
        start = date2num(datetime.strptime('2010-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), units='days since 2010-01-01 00:00:00')
        end = date2num(datetime.strptime('2010-02-01 00:00:00', '%Y-%m-%d %H:%M:%S'), units='days since 2010-01-01 00:00:00')
        time = np.arange(start, end, 1/24/4)
        self.time = num2date(time, units='days since 2010-01-01 00:00:00')
        self.signal1 = np.sin(time) + np.cos(time)
        self.signal2 = np.cos(time) + np.tan(time)

    def test_rmse(self):
        target_rmse = 107.3423232723365
        root_mean_square_error = rmse(self.signal1, self.signal2)
        test.assert_equal(root_mean_square_error, target_rmse)