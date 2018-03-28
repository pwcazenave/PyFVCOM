from __future__ import division

from datetime import datetime
from unittest import TestCase

import numpy as np
import numpy.testing as test
from dateutil.relativedelta import relativedelta
from netCDF4 import num2date, date2num

from PyFVCOM.utilities import *


class UtilitiesTest(TestCase):

    def setUp(self):
        """ Make a couple of time series """
        self.start = date2num(datetime.strptime('2010-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), units='days since 2010-01-01 00:00:00')
        self.end = date2num(datetime.strptime('2010-02-01 00:00:00', '%Y-%m-%d %H:%M:%S'), units='days since 2010-01-01 00:00:00')
        self.num_time = np.arange(self.start, self.end, 1/24/4)
        self.time = num2date(self.num_time, units='days since 2010-01-01 00:00:00')
        self.signal1 = np.sin(self.num_time) + np.cos(self.num_time)
        self.signal2 = np.cos(self.num_time) + np.tan(self.num_time)

    def test_fix_range(self):
        target_min, target_max = -100, 100
        scaled_signal = general.fix_range(self.signal1, target_min, target_max)
        test.assert_equal(scaled_signal.min(), target_min)
        test.assert_equal(scaled_signal.max(), target_max)

    def test_ind2sub(self):
        ind = 25
        shape = (10, 20)
        test_row, test_col = 1, 5
        rows, cols = general.ind2sub(shape, ind)
        test.assert_equal(rows, test_row)
        test.assert_equal(cols, test_col)

    def test_flatten_list(self):
        list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        not_list_of_lists = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        flat_list = general.flatten_list(list_of_lists)
        not_list = general.flatten_list(not_list_of_lists)
        test.assert_equal(not_list_of_lists, flat_list)
        test.assert_equal(not_list, not_list_of_lists)

    def test_split_string(self):
        test_string_spaces = 'this is a string'
        test_string_dots = 'this.is.a.string'
        string_split = ['this', 'is', 'a', 'string']
        default_split = general.split_string(test_string_spaces)
        dot_split = general.split_string(test_string_dots, separator='.')
        test.assert_equal(default_split, string_split)
        test.assert_equal(dot_split, string_split)

    def test_cleanhtml(self):
        html_string = '<a href>This is a link</a>'
        clean_string = 'This is a link'
        html_cleaned = general.cleanhtml(html_string)
        test.assert_equal(clean_string, html_cleaned)

    def test_julian_day(self):
        input_date = [2000, 7, 20, 10, 58, 12]
        actual_julian_day = 2451745.9570833333
        actual_modified_julian_day = actual_julian_day - 2400000.5
        calculated_julian_day = time.julian_day(input_date)
        calculated_modified_julian_day = time.julian_day(input_date, mjd=True)
        # Get some floating point issues here, so do an almost_equal.
        test.assert_almost_equal(actual_julian_day, calculated_julian_day)
        test.assert_almost_equal(actual_modified_julian_day, calculated_modified_julian_day)

    def test_gregorian_date(self):
        input_julian_day = 2451745.9570833333
        input_modified_julian_day = input_julian_day - 2400000.5
        actual_date = [2000, 7, 20, 10, 58, 12]
        calculated_from_julian_day = time.gregorian_date(input_julian_day)
        calculated_from_modified_julian_day = time.gregorian_date(input_modified_julian_day, mjd=True)
        # Get some floating point issues here, so do an almost_equal.
        test.assert_almost_equal(actual_date, calculated_from_julian_day, decimal=5)
        test.assert_almost_equal(actual_date, calculated_from_modified_julian_day, decimal=5)

    def test_date_range(self):
        start = datetime.strptime('2010-02-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime('2010-02-03 00:00:00', '%Y-%m-%d %H:%M:%S')
        time_range = time.date_range(start, end, inc=0.5)  # half-day interval
        test.assert_equal(start, time_range[0])
        test.assert_equal(end, time_range[-1])
        test.assert_equal(5, len(time_range))

    def test_overlap(self):
        test_first_start = datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        test_first_end = datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        test_second_start = datetime.strptime('1999-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        test_second_end = datetime.strptime('2000-01-02 00:00:00', '%Y-%m-%d %H:%M:%S')
        test_third_start = datetime.strptime('1990-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        test_third_end = datetime.strptime('1991-01-02 00:00:00', '%Y-%m-%d %H:%M:%S')
        overlapping = time.overlap(test_first_start, test_first_end, test_second_start, test_second_end)
        not_overlapping = time.overlap(test_first_start, test_first_end, test_third_start, test_third_end)
        test.assert_equal(overlapping, True)
        test.assert_equal(not_overlapping, False)

    def test_common_time(self):
        test_first_start = datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        test_second_start = datetime.strptime('1999-12-31 00:00:00', '%Y-%m-%d %H:%M:%S')
        test_third_start = datetime.strptime('2010-07-10 12:00:00', '%Y-%m-%d %H:%M:%S')
        test_first = [test_first_start + relativedelta(days=i) for i in range(3)]
        test_second = [test_second_start + relativedelta(days=i) for i in range(3)]
        test_third = [test_third_start + relativedelta(days=i) for i in range(3)]
        time_in_common_first_second = time.common_time(test_first, test_second)
        time_in_common_first_third = time.common_time(test_first, test_third)
        common_time_first_second = test_first[:2]
        common_time_first_third = [False, False]
        test.assert_equal(common_time_first_second, time_in_common_first_second)
        test.assert_equal(common_time_first_third, time_in_common_first_third)

    def test_make_signal(self):
        # This should be a much better test.
        amplitude = 2
        period = 4
        phase = 0
        signal = time.make_signal(self.num_time, amplitude, phase, period)
        test.assert_equal(np.max(signal), amplitude)
        test.assert_equal(np.min(signal), -amplitude)
