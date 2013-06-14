"""
Some useful ocean functions taken from ocean_funcs.ncl, which in turn has taken them from the CSIRO SEAWATER (now GSW) MATLAB toolbox.

NCL code can be found at:
    http://www.ncl.ucar.edu/Support/talk_archives/2013/att-1501/ocean_funcs.ncl__size_15540__creation-date_

See Fofonoff, P. & Millard, R.C. Unesco 1983. Algorithms for computation of fundamental properties of seawater, 1983. Unesco Tech. Pap. in Mar. Sci., No. 44.

Provides functions:
    - pressure2depth : convert pressure (decibars) to depth in metres
    - depth2pressure : convert depth in metres to pressure in decibars
    - tsp2rho : calculate density from temperature, salinity and pressure
    - sw_dens0 : calculate seawater density at atmospheric surface pressure
    - sw_smow : calculate density of Standard Mean Ocean Water
    - sw_seck : calculate Secant Bulk Modulus (K) of seawater
    - sw_svan : calculate specific volume anomaly (only use if you don't
      already have density)
    - theta_sw : calculate potential temperature for sea water

"""

import numpy as np

# Define some commonly used constants.
c68 = 1.00024   # conversion constant to 1968 temperature scale.
Re = 6371220.0  # mean radius of earth (m)
g = 9.80665     # standard gravity


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

    c2 = 2.21e-6
    Y = np.sin(np.deg2rad(np.abs(lat)))
    c1 = (5.92 + (5.25 * Y**2.0)) * 1.e-3

    p = ((1.0 - c1) - np.sqrt((1.0 - c1)**2.0 - (4.0 * c2 * z))) / (2.0 * c2)

    return p

def dT_adiab_sw(t, s, p):
    """
    Calculate adiabatic temperature gradient.

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

    # constants
    a0 =  3.5803E-5
    a1 =  8.5258E-6
    a2 = -6.836E-8
    a3 =  6.6228E-10

    b0 =  1.8932E-6
    b1 = -4.2393E-8

    c0 =  1.8741E-8
    c1 = -6.7795E-10
    c2 =  8.733E-12
    c3 = -5.4481E-14

    d0 = -1.1351E-10
    d1 =  2.7759E-12

    e0 = -4.6206E-13
    e1 =  1.8676E-14
    e2 = -2.1687E-16

    T68 = T * 1.00024 # convert to 1968 temperature scale


    out = a0 + (a1 + (a2 + a3*T68)*T68)*T68 + (b0 + b1*T68)*(S-35) + ((c0 + (c1 + (c2 + c3*T68)*T68)*T68) + (d0 + d1*T68)*(S-35) )*P + (e0 + (e1 + e2*T68)*T68 )*P*P

    return(out)

def tsp2rho(t, s, p):
    """
    Convert temperature, salinity and temperature to density.

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
        prin('WARNING: {} values below minimum value temperature (-2C)'.format(n))

    if t.max() > 40:
        n = np.sum(t > 40)
        prin('WARNING: {} values above maximum value temperature (40C)'.format(n))

    if s.min() < 0:
        n = np.sum(s < 0)
        prin('WARNING: {} values below minimum salinity value (0 PSU)'.format(n))

    if s.max() > 42:
        n = np.sum(s > 42)
        prin('WARNING: {} values above maximum salinity value (42C)'.format(n))

    if p.min() < 0:
        n = np.sum(p < 0)
        prin('WARNING: {} values below minimum pressure value (0 decibar)'.format(n))

    if p.max() > 10000:
        n = np.sum(p > 10000)
        prin('WARNING: {} values above maximum pressure value (10000 decibar)'.format(n))

    dens0 = sw_dens0(t, s)
    K = sw_seck(T, S, P)
    Patm = p / 10 # pressure in millibars
    rho = dens0 / (1 - Patm / K)

    return rho

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

    b0 = 8.24493e-1
    b1 = -4.0899e-3
    b2 = 7.6438e-5
    b3 = -8.2467e-7
    b4 = 5.3875e-9

    c0 = -5.72466e-3
    c1 = 1.0227e-4
    c2 = -1.6546e-6

    d0 = 4.8314e-4

    t68 = t * 1.00024

    dens = s * (b0 + (b1*t68) + (b2 * t68**2) + (b3 * t68**3) + (b4 * t68**4)) + \
            s**1.5 * (c0 + (c1 * t68) + (c2 * t68**2)) + (d0 * s**2)

    dens = dens + sw_smow(t68)

    return dens

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

    t : ndarray
        Potential temperature (Celcius)

    """

    dP = pr - p # pressure difference.

    # 1st iteration
    dth = dP * dT_adiab_sw(t, s, p)
    th  = (t * c68) + (0.5 * dth)
    q   = dth

    # 2nd interation
    dth = dP * dT_adiab_sw(th / c68, S, (P + (0.5 * dP)))
    th  = th + ((1 - (1 / np.sqrt(2))) * (dth - q))
    q   = ((2 - np.sqrt(2)) * dth) + (((3 / np.sqrt(2)) - 2) * q)

    # 3rd iteration
    dth = dP * dT_adiab_sw(th / c68, S, (P + (0.5 * dP)))
    th  = th + ((1 + (1 / np.sqrt(2))) * (dth - q))
    q   = ((2+ sqrt(2))*dth) + (((-3/sqrt(2)) - 2) * q)

    # 4th interation
    dth = dP * dT_adiab_sw(th/c68, S, (P + dP))
    th  = (th + (dth - (2*q))/6)/ c68

    return th



if __name__ == '__main__':

    # Put some unit tests in here to make sure the functions work as expected.

