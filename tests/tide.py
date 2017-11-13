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

        # A known "good" result.
        self.test_signal = np.array([-1.07121179, -0.11962379, 1.0109278, 2.12490737, 3.01282377,
                                     3.49579161, 3.46537306, 2.90922325, 1.91622578, 0.65954418,
                                     -0.63884346, -1.75387988, -2.50406992, -2.78720037, -2.59801604,
                                     -2.02424033, -1.22256417, -0.38084462, 0.3242335, 0.76406818,
                                     0.88136377, 0.69505071, 0.28691516, -0.22559774, -0.7227697,
                                     -1.11339619, -1.35528269, -1.46075209, -1.48654153, -1.51155773,
                                     -1.60902271, -1.82082188, -2.14108374, -2.51342712, -2.84262041,
                                     -3.0175824, -2.93970513, -2.549137, -1.84224735, -0.875813,
                                     0.24310352, 1.37897976, 2.39585577, 3.18460594, 3.68287738,
                                     3.88314843, 3.82750929, 3.5911893, 3.25965929, 2.90560853,
                                     2.57187833, 2.26462103, 1.95805233, 1.60896251, 1.17650241,
                                     0.64136039, 0.01865768, -0.63935989, -1.25194082, -1.72774336,
                                     -1.98743981, -1.9850287, -1.72209068, -1.25097773, -0.66575916,
                                     -0.08291716, 0.38354377, 0.64643984, 0.66319267, 0.44273154,
                                     0.03999982, -0.46024836, -0.96592895, -1.40038322, -1.71942479,
                                     -1.91830516, -2.02699675, -2.0953945, -2.17281394, -2.28776731,
                                     -2.43398777, -2.56702771, -2.61289924, -2.48690246, -2.11788425,
                                     -1.47147605, -0.56585564, 0.52469358, 1.68010716, 2.75684293,
                                     3.61706376, 4.15764995, 4.33182789, 4.15791024, 3.71282023,
                                     3.11192078, 2.48019913, 1.92213983, 1.49803406, 1.21287174,
                                     1.02071083, 0.84353814, 0.59888118, 0.22995842, -0.27103678,
                                     -0.85306816, -1.41322768, -1.81958148, -1.9446488, -1.70121962])

    def test_lanczos(self):
        """ Lanczos cosine time series filter """
        filtered_signal, _, _, _, _ = lanczos(self.signal,
                                              dt=self.interval * 24 * 60,
                                              samples=100,
                                              cutoff=1/(60*24), # daily filter
                                              passtype='low')
        test.assert_almost_equal(filtered_signal, self.test_signal)

    def test_Lanczos(self):
        """ Lanczos cosine time series filter using the class """
        filt = Lanczos(dt=self.interval * 24 * 60, samples=100, cutoff=1/(60*24), passtype='low')
        filtered = filt.filter(self.signal)
        test.assert_almost_equal(filtered, self.test_signal)


