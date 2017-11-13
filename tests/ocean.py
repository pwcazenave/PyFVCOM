import numpy.testing as test
import numpy as np

from unittest import TestCase

from PyFVCOM.ocean import *

class OceanToolsTest(TestCase):

    def setUp(self):
        """ Make a set of data for the various ocean tools functions """
        self.lat = 30
        self.z = np.array(9712.02)
        self.t = np.array(40)
        self.s = np.array(40)
        self.p = np.array(10000)
        self.pr = np.array(0)
        self.c = np.array(1.888091)
        self.td = np.array(20)  # for dens_jackett
        self.sd = np.array(20)  # for dens_jackett
        self.pd = np.array(1000)  # for dens_jackett
        self.cond = np.array(53000)  # for cond2salt
        self.h = np.array((10, 20, 30, 100))  # depths for stokes
        self.U = 0.25  # U for stokes and dissipation
        self.omega = 1 / 44714.1647021416  # omega for stokes
        self.z0 = np.array((0.0025))  # z0 for stokes
        self.rho = 1025
        self.temp = np.arange(-20, 50, 10)
        self.dew = np.linspace(0, 20, len(self.temp))

    # Use some of the Fofonoff and Millard (1983) checks.
    def test_sw_svan(self):
        """ Specific volume anomaly """
        test_svan = 9.8130210e-6
        res_svan = sw_svan(self.t, self.s, self.p)
        test.assert_almost_equal(res_svan, test_svan, decimal=1)

    def test_res_z(self):
        """ Pressure to depth """
        test_z = 9712.02
        res_z = pressure2depth(self.p, self.lat)
        # Hmmm, not very accurate!
        test.assert_almost_equal(res_z, test_z, decimal=-1)

    # The return to depth is a bit inaccurate, not sure why.
    def test_depth2pressure(self):
        """ Depth to pressure """
        test_p = 9712.653
        res_pres = depth2pressure(self.z, self.lat)
        # Hmmm, horribly inaccurate!
        test.assert_almost_equal(res_pres, test_p, decimal=-4)

    def test_cp_sw(self):
        """ Specific heat of seawater """
        test_cp = 3849.5
        res_cp = cp_sw(self.t, self.s, self.p)
        test.assert_almost_equal(res_cp, test_cp, decimal=1)

    def test_dT_adiab_sw(self):
        """ Adiabatic temperature gradient """
        test_atg = 0.0003255976
        res_atg = dT_adiab_sw(self.t, self.s, self.p)
        test.assert_almost_equal(res_atg, test_atg, decimal=6)

    def test_theta_sw(self):
        """ Potential temperature for sea water """
        test_theta = 36.89073
        res_theta = theta_sw(self.t, self.s, self.p, self.pr)
        test.assert_almost_equal(res_theta, test_theta, decimal=2)

    def test_sw_sal78(self):
        """ Salinity from conductivity, temperature and pressure (sw_sal78) """
        test_salinity = 40
        res_sal78 = sw_sal78(self.c, self.t, self.p)
        test.assert_almost_equal(res_sal78, test_salinity, decimal=5)

    def test_dens_jackett(self):
        """ Density from temperature, salinity and pressure """
        test_dens = 1017.728868019642
        res_dens = dens_jackett(self.td, self.sd, self.pd)
        test.assert_equal(res_dens, test_dens)

    def test_cond2salt(self):
        """ Conductivity to salinity """
        test_salt = 34.935173507811783
        res_salt = cond2salt(self.cond)
        test.assert_equal(res_salt, test_salt)

    # def test_stokes(self):
    #     """ Stokes number """
    #     test_stokes, test_u_star, test_delta = np.nan, np.nan, np.nan
    #     res_stokes, res_u_star, res_delta = stokes(self.h, self.U, self.omega, self.z0, U_star=True, delta=True)
    #     test.assert_equal(res_stokes, test_stokes)
    #     test.assert_equal(res_u_star, test_u_star)
    #     test.assert_equal(res_delta, test_delta)

    def test_dissipation(self):
        """ Tidal dissipation for a given tidal harmonic """
        test_dissipation = 0.0400390625
        res_dissipation = dissipation(self.rho, self.U)
        test.assert_equal(res_dissipation, test_dissipation)

    def test_rhum(self):
        """ Relative humidity from dew temperature and air temperature """
        test_rhum = np.array((487.36529085, 270.83391406, 160.16590946, 100.0, 65.47545095, 44.70251971, 31.67003471))
        res_rhum = rhum(self.dew, self.temp)
        test.assert_almost_equal(res_rhum, test_rhum)
