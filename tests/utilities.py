from __future__ import division

import numpy.testing as test
import numpy as np

from datetime import datetime
from netCDF4 import num2date, date2num

from unittest import TestCase

from PyFVCOM.utilities import *

class UtilitiesTest(TestCase):

    def setUp(self):
        """ Make a couple of time series """
        self.start = date2num(datetime.strptime('2010-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), units='days since 2010-01-01 00:00:00')
        self.end = date2num(datetime.strptime('2010-02-01 00:00:00', '%Y-%m-%d %H:%M:%S'), units='days since 2010-01-01 00:00:00')
        time = np.arange(self.start, self.end, 1/24/4)
        self.time = num2date(time, units='days since 2010-01-01 00:00:00')
        self.signal1 = np.sin(time) + np.cos(time)
        self.signal2 = np.cos(time) + np.tan(time)

    def test_fix_range(self):
        target_min, target_max = -100, 100
        scaled_signal = fix_range(self.signal1, target_min, target_max)
        test.assert_equal(scaled_signal.min(), target_min)
        test.assert_equal(scaled_signal.max(), target_max)

    def test_ind2sub(self):
        ind = 25
        shape = (10, 20)
        test_row, test_col = 1, 5
        rows, cols = ind2sub(shape, ind)
        test.assert_equal(rows, test_row)
        test.assert_equal(cols, test_col)

    def test_date_range(self):
        start = datetime.strptime('2010-02-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime('2010-02-03 00:00:00', '%Y-%m-%d %H:%M:%S')
        time_range = date_range(start, end, inc=0.5)  # half-day interval
        test.assert_equal(start, time_range[0])
        test.assert_equal(end, time_range[-1])
        test.assert_equal(5, len(time_range))

