"""Copyright 2000, 2001 William McClain

    This file is part of Astrolabe.

    Astrolabe is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Astrolabe is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astrolabe; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Calculate the times of solstice and equinox events for Earth
    """

from math import pi, cos
from astronomia.calendar import jd_to_jcent
from astronomia.constants import pi2
from astronomia.nutation import nut_in_lon
from astronomia.sun import aberration_low, Sun
from astronomia.util import d_to_r, polynomial, diff_angle
from astronomia.vsop87d import vsop_to_fk5
import astronomia.globals

class Error(Exception):
    """local exception class"""
    pass

#
# Meeus-1998 Table 27.A
#
_approx_1000 = {
    "spring": (1721139.29189, 365242.13740,  0.06134,  0.00111, -0.00071),
    "summer": (1721233.25401, 365241.72562, -0.05323,  0.00907, -0.00025),
    "autumn": (1721325.70455, 365242.49558, -0.11677, -0.00297,  0.00074),
    "winter": (1721414.39987, 365242.88257, -0.00769, -0.00933, -0.00006)}

#
# Meeus-1998 Table 27.B
#
_approx_3000 = {
    "spring": (2451623.80984, 365242.37404,  0.05169, -0.00411, -0.00057),
    "summer": (2451716.56767, 365241.62603,  0.00325,  0.00888, -0.00030),
    "autumn": (2451810.21715, 365242.01767, -0.11575,  0.00337,  0.00078),
    "winter": (2451900.05952, 365242.74049, -0.06223, -0.00823,  0.00032)}

#
# Meeus-1998 Table 27.C
#
_terms = [
    (485, d_to_r(324.96),  d_to_r(  1934.136)),
    (203, d_to_r(337.23),  d_to_r( 32964.467)),
    (199, d_to_r(342.08),  d_to_r(    20.186)),
    (182, d_to_r( 27.85),  d_to_r(445267.112)),
    (156, d_to_r( 73.14),  d_to_r( 45036.886)),
    (136, d_to_r(171.52),  d_to_r( 22518.443)),
    ( 77, d_to_r(222.54),  d_to_r( 65928.934)),
    ( 74, d_to_r(296.72),  d_to_r(  3034.906)),
    ( 70, d_to_r(243.58),  d_to_r(  9037.513)),
    ( 58, d_to_r(119.81),  d_to_r( 33718.147)),
    ( 52, d_to_r(297.17),  d_to_r(   150.678)),
    ( 50, d_to_r( 21.02),  d_to_r(  2281.226)),
    ( 45, d_to_r(247.54),  d_to_r( 29929.562)),
    ( 44, d_to_r(325.15),  d_to_r( 31555.956)),
    ( 29, d_to_r( 60.93),  d_to_r(  4443.417)),
    ( 18, d_to_r(155.12),  d_to_r( 67555.328)),
    ( 17, d_to_r(288.79),  d_to_r(  4562.452)),
    ( 16, d_to_r(198.04),  d_to_r( 62894.029)),
    ( 14, d_to_r(199.76),  d_to_r( 31436.921)),
    ( 12, d_to_r( 95.39),  d_to_r( 14577.848)),
    ( 12, d_to_r(287.11),  d_to_r( 31931.756)),
    ( 12, d_to_r(320.81),  d_to_r( 34777.259)),
    (  9, d_to_r(227.73),  d_to_r(  1222.114)),
    (  8, d_to_r( 15.45),  d_to_r( 16859.074))]
    
def equinox_approx(yr, season):
    """Returns the approximate time of a solstice or equinox event.
    
    The year must be in the range -1000...3000. Within that range the
    the error from the precise instant is at most 2.16 minutes.
    
    Parameters:
        yr     : year
        season : one of ("spring", "summer", "autumn", "winter")
    
    Returns:
        Julian Day of the event in dynamical time
    
    """
    if not (-1000 <= yr <= 3000):
        raise Error, "year is out of range"
    if season not in astronomia.globals.season_names:
        raise Error, "unknown season =" + season
        
    yr = int(yr)
    if -1000 <= yr <= 1000:
        Y = yr / 1000.0
        tbl = _approx_1000
    else:
        Y = (yr - 2000) / 1000.0
        tbl = _approx_3000

    jd = polynomial(tbl[season], Y)
    T = jd_to_jcent(jd)
    W = d_to_r(35999.373 * T - 2.47)
    delta_lambda = 1 + 0.0334 * cos(W) + 0.0007 * cos(2 * W)

    jd += 0.00001 * sum([A * cos(B + C * T) for A, B, C in _terms]) / delta_lambda

    return jd

_circle = {
    "spring": 0.0,
    "summer": pi * 0.5,
    "autumn": pi,
    "winter": pi * 1.5}

_k_sun_motion = 365.25 / pi2

def equinox(jd, season, delta):
    """Return the precise moment of an equinox or solstice event on Earth.
    
    Parameters:
        jd     : Julian Day of an approximate time of the event in dynamical time
        season : one of ("spring", "summer", "autumn", "winter")
        delta  : the required precision in days. Times accurate to a second are
            reasonable when using the VSOP model.
        
    Returns:
        Julian Day of the event in dynamical time

    """
    #
    # If we knew that the starting approximate time was close enough
    # to the actual time, we could pull nut_in_lon() and the aberration
    # out of the loop and save some calculating.
    #
    circ = _circle[season]
    sun = Sun()
    for i in range(20):
        jd0 = jd
        L, B, R = sun.dimension3(jd)
        L += nut_in_lon(jd) + aberration_low(R)
        L, B = vsop_to_fk5(jd, L, B)
        # Meeus uses jd + 58 * sin(diff(...))
        jd += diff_angle(L, circ) * _k_sun_motion
        if abs(jd - jd0) < delta: 
            return jd
    raise Error, "bailout"

