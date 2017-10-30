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
        test_signal = np.array([0.14126143, 0.80711636, 1.37180911, 1.76993207, 1.99026636,
                                2.05381835, 1.98330516, 1.78151062, 1.43019889, 0.90993189,
                                0.22985721, -0.55713341, -1.31757369, -1.90262754, -2.18930653,
                                -2.11623963, -1.70364091, -1.05151044, -0.31622047, 0.3282598,
                                0.73390304, 0.81498094, 0.56732878, 0.06544166, -0.5614174,
                                -1.16702631, -1.62900938, -1.88100583, -1.92718247, -1.83525584,
                                -1.71044824, -1.65833568, -1.74776321, -1.98498085, -2.30689658,
                                -2.59578873, -2.71156378, -2.53248775, -1.99274639, -1.10594889,
                                0.03242497, 1.26472561, 2.40782591, 3.29684275, 3.82393487,
                                3.96161683, 3.7644064, 3.34873669, 2.85716918, 2.4172999,
                                2.10712132, 1.93660426, 1.85048155, 1.75102687, 1.53380943,
                                1.12563901, 0.51327755, -0.24583228, -1.03435336, -1.70462559,
                                -2.11878719, -2.1880483, -1.89981517, -1.32470208, -0.60110198,
                                0.09868634, 0.61156395, 0.82414708, 0.69960373, 0.28359482,
                                -0.31199339, -0.94258856, -1.4719912, -1.80919032, -1.93053223,
                                -1.88116766, -1.75575149, -1.66439732, -1.69416331, -1.87763766,
                                -2.17813138, -2.4961638, -2.69572919, -2.6430424, -2.24674529,
                                -1.48798807, -0.43155507, 0.78549775, 1.98489138, 2.98949366,
                                3.66544057, 3.95250833, 3.87455897, 3.52751003, 3.04860674,
                                2.57599603, 2.2102326, 1.98864289, 1.87964726, 1.79829928,
                                1.61445049, 1.24543605, 0.68551425, 0.01597098, -0.6191033,
                                -1.06238141, -1.19875297, -0.99478846, -0.50722723])
        filtered_signal, _, _, _, _ = lanczos(self.signal,
                                              dt=self.interval * 24 * 60,
                                              samples=10,
                                              cutoff=1/(60*24), # daily filter
                                              passtype='low')
        test.assert_almost_equal(filtered_signal, test_signal)


