import numpy.testing as test
import numpy as np

from unittest import TestCase
from datetime import datetime

from PyFVCOM.tide import *
from PyFVCOM.utilities import date_range, make_signal

from matplotlib.pyplot import *

class TideTest(TestCase):

    def setUp(self):
        """ Make a set of data for the various tide functions """
        self.start = datetime.strptime('2010-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.end = datetime.strptime('2010-01-10 00:00:01', '%Y-%m-%d %H:%M:%S')
        self.interval = 2 / 24  # 2-hourly interval in days
        self.amplitude = 2  # base signal amplitude
        self.phase = 90  # base signal phase (degrees)
        self.period = (1 / (12 + (25 / 60))) * 24  # base signal period in days (~M2 tide)

        self.time = date_range(self.start, self.end, inc=self.interval)
        self.num_time = [int(d.strftime('%s')) / 60 / 60 / 24 for d in self.time]  # float days
        # Make a somewhat complicated time series to filter. It's characteristics are a strong sub-daily signal with
        # lower amplitude signals with longer periods added.
        self.signal = make_signal(self.num_time, self.amplitude, self.phase * 2, self.period / 2)
        self.signal += make_signal(self.num_time, self.amplitude, self.phase, self.period)
        self.signal += make_signal(self.num_time, self.amplitude, self.phase * 0.75, self.period * 2)

    def test_lanczos(self):
        """ Lanczos cosine time series filter """
        test_signal = np.array([2.11587658e-01, 9.24698472e-01, 1.48269889e+00,
                                1.84482586e+00, 2.02441756e+00, 2.05871180e+00,
                                1.97542688e+00, 1.77341612e+00, 1.42680574e+00,
                                9.09931890e-01, 2.29857208e-01, -5.57133414e-01,
                                -1.31757369e+00, -1.90262754e+00, -2.18930653e+00,
                                -2.11623963e+00, -1.70364091e+00, -1.05151044e+00,
                                -3.16220468e-01, 3.28259798e-01, 7.33903037e-01,
                                8.14980942e-01, 5.67328778e-01, 6.54416636e-02,
                                -5.61417401e-01, -1.16702631e+00, -1.62900938e+00,
                                -1.88100583e+00, -1.92718247e+00, -1.83525584e+00,
                                -1.71044824e+00, -1.65833568e+00, -1.74776321e+00,
                                -1.98498085e+00, -2.30689658e+00, -2.59578873e+00,
                                -2.71156378e+00, -2.53248775e+00, -1.99274639e+00,
                                -1.10594889e+00, 3.24249696e-02, 1.26472561e+00,
                                2.40782591e+00, 3.29684275e+00, 3.82393487e+00,
                                3.96161683e+00, 3.76440640e+00, 3.34873669e+00,
                                2.85716918e+00, 2.41729990e+00, 2.10712132e+00,
                                1.93660426e+00, 1.85048155e+00, 1.75102687e+00,
                                1.53380943e+00, 1.12563901e+00, 5.13277546e-01,
                                -2.45832283e-01, -1.03435336e+00, -1.70462559e+00,
                                -2.11878719e+00, -2.18804830e+00, -1.89981517e+00,
                                -1.32470208e+00, -6.01101984e-01, 9.86863384e-02,
                                6.11563946e-01, 8.24147077e-01, 6.99603730e-01,
                                2.83594818e-01, -3.11993390e-01, -9.42588563e-01,
                                -1.47199120e+00, -1.80919032e+00, -1.93053223e+00,
                                -1.88116766e+00, -1.75575149e+00, -1.66439732e+00,
                                -1.69416331e+00, -1.87763766e+00, -2.17813138e+00,
                                -2.49616380e+00, -2.69572919e+00, -2.64304240e+00,
                                -2.24674529e+00, -1.48798807e+00, -4.31555065e-01,
                                7.85497750e-01, 1.98489138e+00, 2.98949366e+00,
                                3.66544057e+00, 3.95250833e+00, 3.87455897e+00,
                                3.52751003e+00, 3.04860674e+00, 2.57599603e+00,
                                2.21023260e+00, 1.98864289e+00, 1.87964726e+00,
                                1.79829928e+00, 1.63812574e+00, 1.29086073e+00,
                                7.25069526e-01, -1.92256509e-03, -7.55653367e-01,
                                -1.36583010e+00, -1.68263133e+00, -1.62693065e+00,
                                -1.21544662e+00, -5.51819064e-01])
        filtered_signal, _, _, _, _ = lanczos(self.signal,
                                              dt=self.interval * 24 * 60,
                                              samples=10,
                                              cutoff=1/(60*24), # daily filter
                                              passtype='low')
        test.assert_almost_equal(filtered_signal, test_signal)


