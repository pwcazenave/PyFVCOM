"""
Some useful ocean functions taken from ocean_funcs.ncl, which in turn has taken
them from the CSIRO SEAWATER (now GSW) MATLAB toolbox.

NCL code can be found at:
    http://www.ncl.ucar.edu/Support/talk_archives/2013/att-1501/ocean_funcs.ncl__size_15540__creation-date_

See Fofonoff, P. & Millard, R.C. Unesco 1983. Algorithms for computation of
fundamental properties of seawater, 1983. Unesco Tech. Pap. in Mar. Sci., No.
44.

Provides functions:
    - pressure2depth : convert pressure (decibars) to depth in metres
    - depth2pressure : convert depth in metres to pressure in decibars
    - dT_adiab_sw : calculate adiabatic temperature gradient
    - theta_sw : calculate potential temperature for sea water
    - cp_sw : calculate constant pressure specific heat for seawater
    - sw_smow : calculate density of Standard Mean Ocean Water
    - sw_dens0 : calculate seawater density at atmospheric surface pressure
    - sw_seck : calculate Secant Bulk Modulus (K) of seawater
    - sw_dens : calculate density from temperature, salinity and pressure
    - sw_svan : calculate specific volume anomaly (only use if you don't
      already have density)

Pierre Cazenave (Plymouth Marine Laboratory) 2013/06/14

"""

import numpy as np

# Define some commonly used constants.
c68 = 1.00024   # conversion constant to 1968 temperature scale.


def pressure2depth(p, lat):
    """
    Convert from pressure in decibars to depth in metres.

    Parameters
    ----------

    p : ndarray
        Pressure (1D array) in decibars.
    lat : ndarray
        Latitudes for samples in p.

    Returns
    -------

    z : ndarray
        Water depth in metres.

    """

    c1  =  9.72659
    c2  = -2.1512e-5
    c3  =  2.279e-10
    c4  = -1.82e-15
    gam =  2.184e-6

    y = np.abs(lat)
    rad = np.sin(np.deg2rad(y))**2

    gy = 9.780318 * (1.0 + (rad * 5.2788e-3) + (rad**2 * 2.36e-5))

    bline = gy + (gam * 0.5 * p)

    tline = (c1 * p) + (c2 * p**2.0) + (c3 * p**3.0) + (c4 * p**4.0)
    z = tline / bline

    return z

def depth2pressure(z, lat):
    """
    Convert from depth in metres to pressure in decibars.

    Parameters
    ----------

    z : ndarray
        Depth (1D array) in metres.
    lat : ndarray
        Latitudes for samples in z.

    Returns
    -------

    p : ndarray
        Pressure in decibars.

    """

    # Convert depths to positive values only - should this be more robust? When
    # will we have both positive and negative depth values? Wetting and drying
    # springs to mind, but not sure what I can do about that here. The data
    # should probably be sanitised before coming to here.
    pz = np.abs(z)

    c2 = 2.21e-6
    Y = np.sin(np.deg2rad(np.abs(lat)))
    c1 = (5.92 + (5.25 * Y**2.0)) * 1.e-3

    p = ((1.0 - c1) - np.sqrt((1.0 - c1)**2.0 - (4.0 * c2 * pz))) / (2.0 * c2)

    return p

def dT_adiab_sw(t, s, p):
    """
    Calculate adiabatic temperature gradient (degrees Celcius dbar^{-1})

    Parameters
    ----------

    t : ndarray
        Temperature (Celcius)
    s : ndarray
        Salinity (PSU)
    p : ndarray
        Pressure (decibars)

    All three arrays must have the same shape.

    Returns
    -------

    atg : ndarray
        Adiabatic temperature gradient

    """

    # Constants
    a0 =  3.5803e-5
    a1 =  8.5258e-6
    a2 = -6.836e-8
    a3 =  6.6228e-10

    b0 =  1.8932e-6
    b1 = -4.2393e-8

    c0 =  1.8741e-8
    c1 = -6.7795e-10
    c2 =  8.733e-12
    c3 = -5.4481e-14

    d0 = -1.1351e-10
    d1 =  2.7759e-12

    e0 = -4.6206e-13
    e1 =  1.8676e-14
    e2 = -2.1687e-16

    T68 = t * c68 # convert to 1968 temperature scale


    atg = a0 + (a1 + (a2 + a3 * T68) * T68) * T68 + (b0 + b1 * T68) * (s - 35) + \
            ((c0 + (c1 + (c2 + c3 * T68) * T68) * T68) + (d0 + d1 * T68) * \
            (s - 35)) * p + (e0 + (e1 + e2 * T68) * T68) * p * p

    return atg

def theta_sw(t, s, p, pr):
    """
    Calculate potential temperature for seawater from temperature, salinity and
    pressure.

    Parameters
    ----------

    t : ndarray
        Temperature (1D array) in degrees Celcius.
    s : ndarray
        Salinity (1D array) in practical salinity units (unitless). Must be the
        same shape as t.
    p : ndarray
        Pressure (1D array) in decibars. Must be the same shape as t.
    pr : ndarray
        Reference pressure (decibars) either a scalar or the same shape as t.

    Returns
    -------

    th : ndarray
        Potential temperature (Celcius)

    """

    dP = pr - p # pressure difference.

    # 1st iteration
    dth = dP * dT_adiab_sw(t, s, p)
    th  = (t * c68) + (0.5 * dth)
    q   = dth

    # 2nd interation
    dth = dP * dT_adiab_sw(th / c68, s, (p + (0.5 * dP)))
    th  = th + ((1 - (1 / np.sqrt(2))) * (dth - q))
    q   = ((2 - np.sqrt(2)) * dth) + (((3 / np.sqrt(2)) - 2) * q)

    # 3rd iteration
    dth = dP * dT_adiab_sw(th / c68, s, (p + (0.5 * dP)))
    th  = th + ((1 + (1 / np.sqrt(2))) * (dth - q))
    q   = ((2 + np.sqrt(2)) * dth) + (((-3 / np.sqrt(2)) - 2) * q)

    # 4th interation
    dth = dP * dT_adiab_sw(th / c68, s, (p + dP))
    th  = (th + (dth - (2 * q)) / 6) / c68

    return th

def cp_sw(t, s, p):
    """
    Calculate constant pressure specific heat (cp) for seawater, from
    temperature, salinity and pressure.

    Parameters
    ----------

    t : ndarray
        Temperature (1D array) in degrees Celcius.
    s : ndarray
        Salinity (1D array) in practical salinity units (unitless). Must be the
        same shape as t.
    p : ndarray
        Pressure (1D array) in decibars. Must be the same shape as t.

    Returns
    -------

    cp : ndarray
        Constant pressure specific heat (Celcius).

    Notes
    -----

    Valid temperature range is -2 to 40C and salinity is 0-42 PSU. Warnings are
    issued if the data fall outside these ranges.

    """

    # Check for values outside the valid ranges.
    if t.min() < -2:
        n = np.sum(t < -2)
        print('WARNING: {} values below minimum value temperature (-2C)'.format(n))

    if t.max() > 40:
        n = np.sum(t > 40)
        print('WARNING: {} values above maximum value temperature (40C)'.format(n))

    if s.min() < 0:
        n = np.sum(s < 0)
        print('WARNING: {} values below minimum salinity value (0 PSU)'.format(n))

    if s.max() > 42:
        n = np.sum(s > 42)
        print('WARNING: {} values above maximum salinity value (42C)'.format(n))

    # Convert from decibar to bar and temperature to the 1968 temperature scale.
    pbar = p / 10.0
    T1 = t * c68

    # Specific heat at p = 0

    # Temperature powers
    T2 = T1**2
    T3 = T1**3
    T4 = T1**4

    # Empirical constants
    c0 =  4217.4
    c1 = -3.720283
    c2 =  0.1412855
    c3 = -2.654387e-3
    c4 =  2.093236e-5

    a0 = -7.643575
    a1 =  0.1072763
    a2 = -1.38385e-3

    b0 =  0.1770383
    b1 = -4.07718e-3
    b2 =  5.148e-5

    cp_0t0 = c0 + (c1 * T1) + (c2 * T2) + (c3 * T3) + (c4 * T4)

    A = a0 + (a1 * T1) + (a2 * T2)
    B = b0 + (b1 * T1) + (b2 * T2)

    cp_st0 = cp_0t0 + (A * s) + (B * s**1.5)

    # Pressure dependance
    a0 = -4.9592e-1
    a1 =  1.45747e-2
    a2 = -3.13885e-4
    a3 =  2.0357e-6
    a4 =  1.7168e-8

    b0 =  2.4931e-4
    b1 = -1.08645e-5
    b2 =  2.87533e-7
    b3 = -4.0027e-9
    b4 =  2.2956e-11

    c0 = -5.422e-8
    c1 =  2.6380e-9
    c2 = -6.5637e-11
    c3 =  6.136e-13

    d1_cp = (pbar * (a0 + (a1 * T1) + (a2 * T2) + (a3 * T3) + (a4 * T4))) + \
            (pbar**2 * (b0 + (b1 * T1) + (b2 * T2) + (b3 * T3) + (b4 * T4))) + \
            (pbar**3 * (c0 + (c1 * T1) + (c2 * T2) + (c3 * T3)))

    d0 =  4.9247e-3
    d1 = -1.28315e-4
    d2 =  9.802e-7
    d3 =  2.5941e-8
    d4 = -2.9179e-10

    e0 = -1.2331e-4
    e1 = -1.517e-6
    e2 =  3.122e-8

    f0 = -2.9558e-6
    f1 =  1.17054e-7
    f2 = -2.3905e-9
    f3 =  1.8448e-11

    g0 =  9.971e-8

    h0 =  5.540e-10
    h1 = -1.7682e-11
    h2 =  3.513e-13

    j1 = -1.4300e-12

    d2_cp = pbar * \
            ((s * (d0 + (d1 * T1) + (d2 * T2) + (d3 * T3) + (d4 * T4))) + \
            (s**1.5 *(e0 + (e1 * T1) + (e2 * T2)))) + \
            (pbar**2 * ((s * (f0 + (f1 * T1) + (f2 * T2) + (f3 * T3))) + \
            (g0 * s**1.5))) + (pbar**3 * ((s * (h0 + (h1 * T1) + (h2 * T2))) + \
            (j1 *T1 * s**1.5)))

    cp = cp_st0 + d1_cp + d2_cp

    return(cp)

def sw_smow(t):
    """
    Calculate the density of Standard Mean Ocean Water (pure water).

    Parameters
    ----------

    t : ndarray
        Temperature (1D array) in degrees Celcius.

    Returns
    -------

    rho : ndarray
        Density in kg m^{-3}.

    """

    # Coefficients
    a0 =  999.842594
    a1 =  6.793952e-2
    a2 = -9.095290e-3
    a3 =  1.001685e-4
    a4 = -1.120083e-6
    a5 =  6.536332e-9

    T68 = t * c68

    dens = a0 + (a1 * T68) + (a2 * T68**2) + (a3 * T68**3) + (a4 * T68**4) + (a5 * T68**5)

    return dens

def sw_dens0(t, s):
    """
    Calculate sea water density at atmospheric surface pressure.

    Parameters
    ----------

    t : ndarray
        Temperature (1D array) in degrees Celcius.
    s: ndarray
        Salinity (PSU). Must be the same size as t.

    Returns
    -------

    dens : ndarray
        Seawater density at atmospheric surface pressure (kg m^{-1}).

    """

    b0 =  8.24493e-1
    b1 = -4.0899e-3
    b2 =  7.6438e-5
    b3 = -8.2467e-7
    b4 =  5.3875e-9

    c0 = -5.72466e-3
    c1 =  1.0227e-4
    c2 = -1.6546e-6

    d0 =  4.8314e-4

    t68 = t * c68

    dens = s * (b0 + (b1 * t68) + (b2 * t68**2) + (b3 * t68**3) + (b4 * t68**4)) + \
            s**1.5 * (c0 + (c1 * t68) + (c2 * t68**2)) + (d0 * s**2)

    dens = dens + sw_smow(t68)

    return dens

def sw_seck(t, s, p):
    """
    Calculate Secant Bulk Modulus (K) of seawater.

    Parameters
    ----------

    t : ndarray
        Temperature (1D array) in degrees Celcius.
    s : ndarray
        Salinity (1D array) in practical salinity units (unitless). Must be the
        same shape as t.
    p : ndarray
        Pressure (1D array) in decibars. Must be the same shape as t.

    Returns
    -------

    k : ndarray
        Secant Bulk Modulus of seawater.

    """

    # Compression terms

    T68 = t * c68
    Patm = p / 10.0 # convert to bar

    h3 = -5.77905e-7
    h2 =  1.16092e-4
    h1 =  1.43713e-3
    h0 =  3.239908

    AW = h0 + (h1 * T68) + (h2 * T68**2) + (h3 * T68**3)

    k2 =  5.2787e-8
    k1 = -6.12293e-6
    k0 =  8.50935e-5

    BW = k0 + (k1 + k2 * T68) * T68

    e4 = -5.155288e-5
    e3 =  1.360477e-2
    e2 = -2.327105
    e1 =  148.4206
    e0 =  19652.21

    KW = e0 + (e1 + (e2 + (e3 + e4 * T68) * T68) * T68) * T68

    # K at atmospheric pressure

    j0 =  1.91075e-4

    i2 = -1.6078e-6
    i1 = -1.0981e-5
    i0 =  2.2838e-3

    A = AW + s * (i0 + (i1 * T68) + (i2 * T68**2) ) + (j0 * s**1.5)

    m2 =  9.1697e-10
    m1 =  2.0816e-8
    m0 = -9.9348e-7

    # Equation 18
    B = BW + (m0 + (m1 * T68) + (m2 * T68**2)) * s

    f3 = -6.1670e-5
    f2 =  1.09987e-2
    f1 = -0.603459
    f0 =  54.6746

    g2 = -5.3009e-4
    g1 =  1.6483e-2
    g0 =  7.944e-2

    # Equation 16
    K0 = KW + s * (f0 + (f1 * T68)+ (f2 * T68**2) + (f3 * T68**3)) + \
            s**1.5 * (g0 + (g1 * T68) + (g2 * T68**2))

    # K at t, s, p
    K = K0 + (A * Patm) + (B * Patm**2) # Equation 15

    return K

def sw_dens(t, s, p):
    """
    Convert temperature, salinity and pressure to density.

    Parameters
    ----------

    t : ndarray
        Temperature (1D array) in degrees Celcius.
    s : ndarray
        Salinity (1D array) in practical salinity units (unitless). Must be the
        same shape as t.
    p : ndarray
        Pressure (1D array) in decibars. Must be the same shape as t.

    Returns
    -------

    rho : ndarray
        Density in kg m^{-3}.

    Notes
    -----

    Valid temperature range is -2 to 40C, salinity is 0-42 and pressure is
    0-10000 decibars. Warnings are issued if the data fall outside these
    ranges.

    """

    # Check for values outside the valid ranges.
    if t.min() < -2:
        n = np.sum(t < -2)
        print('WARNING: {} values below minimum value temperature (-2C)'.format(n))

    if t.max() > 40:
        n = np.sum(t > 40)
        print('WARNING: {} values above maximum value temperature (40C)'.format(n))

    if s.min() < 0:
        n = np.sum(s < 0)
        print('WARNING: {} values below minimum salinity value (0 PSU)'.format(n))

    if s.max() > 42:
        n = np.sum(s > 42)
        print('WARNING: {} values above maximum salinity value (42C)'.format(n))

    if p.min() < 0:
        n = np.sum(p < 0)
        print('WARNING: {} values below minimum pressure value (0 decibar)'.format(n))

    if p.max() > 10000:
        n = np.sum(p > 10000)
        print('WARNING: {} values above maximum pressure value (10000 decibar)'.format(n))

    dens0 = sw_dens0(t, s)
    k = sw_seck(t, s, p)
    Patm = p / 10.0 # pressure in bars
    rho = dens0 / (1 - Patm / k)

    return rho

def sw_svan(t, s, p):
    """
    Calculate the specific volume (steric) anomaly.

    t : ndarray
        Temperature (1D array) in degrees Celcius.
    s : ndarray
        Salinity (1D array) in practical salinity units (unitless). Must be the
        same shape as t.
    p : ndarray
        Pressure (1D array) in decibars. Must be the same shape as t.

    Returns
    -------

    svan : ndarray
        Specific Volume Anomaly in kg m^{-3}.

    """

    rho  = sw_dens(t, s, p)
    rho0 = sw_dens(np.zeros(p.shape), np.ones(p.shape) * 35.0, p)
    svan = (1 / rho) - (1 / rho0)

    return svan

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Put some unit tests in here to make sure the functions work as expected.

    test_lat = 30

    test_z = np.logspace(0.1, 4, 50) # log depth distribution
    test_p = np.logspace(0.1, 4, 50) # log pressure distribution

    res_p1 = depth2pressure(test_z, test_lat)
    res_z1 = pressure2depth(res_p1, test_lat)

    res_z2 = pressure2depth(test_p, test_lat)
    res_p2 = depth2pressure(res_z2, test_lat)

    # Graph the differences
    fig0 = plt.figure(figsize=(12, 10))
    ax0 = fig0.add_subplot(1, 2, 1)
    ax0.loglog(test_z, res_z1 - test_z, '.')
    ax0.set_xlabel('Depth (m)')
    ax0.set_ylabel('Difference (m)')
    ax0.set_title('depth2pressure <-> pressure2depth')

    ax1 = fig0.add_subplot(1, 2, 2)
    ax1.loglog(test_p, res_p2 - test_p, '.')
    ax1.set_xlabel('Pressure (dbar)')
    ax1.set_ylabel('Difference (dbar)')
    ax1.set_title('pressure2depth <-> depth2pressure ')

    fig0.show()


    # Use some of the Fofonoff and Millard (1983) checks.
    test_s, test_t, test_p = np.array(40), np.array(40), np.array(10000)
    res_svan = sw_svan(test_t, test_s, test_p)
    print('Steric anomaly\nFofonoff and Millard (1983):\t9.8130210e-6\nsw_svan:\t\t\t{}\n'.format(res_svan))

    test_p = np.array(10000)
    res_z = pressure2depth(test_p, test_lat)
    print('Pressure to depth\nFofonoff and Millard (1983):\t9712.653\npressure2depth:\t\t\t{}\n'.format(res_z))

    test_s, test_t, test_p = np.array(40), np.array(40), np.array(10000)
    res_cp = cp_sw(test_t, test_s, test_p)
    print('Specific heat of seawater\nFofonoff and Millard (1983):\t3849.500\ncp_sw:\t\t\t\t{}\n'.format(res_cp))

    test_s, test_t, test_p = np.array(40), np.array(40), np.array(10000)
    res_atg = dT_adiab_sw(test_t, test_s, test_p)
    print('Adiabatic temperature gradient\nFofonoff and Millard (1983):\t0.0003255976\ndT_adiab_sw:\t\t\t{}\n'.format(res_atg))

    test_s, test_t, test_p, test_pr = np.array(40), np.array(40), np.array(10000), np.array(0)
    res_theta = theta_sw(test_t, test_s, test_p, test_pr)
    print('Potential temperature\nFofonoff and Millard (1983):\t36.89073\ntheta_sw:\t\t\t{}\n'.format(res_theta))
