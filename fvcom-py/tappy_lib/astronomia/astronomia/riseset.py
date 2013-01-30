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
    """

"""Compute Rise, Set, and Transit times.

Each of the routines requires three equatorial coordinates for the
object: yesterday, today and tomorrow, all at 0hr UT.

This approach is inadequate for the Moon, which moves too fast to
be accurately interpolated from three daily positions.

Bug: each of the routines drops some events which occur near 0hr UT.

"""

from math import *

from astronomia.calendar import sidereal_time_greenwich
from astronomia.constants import seconds_per_day, pi2, earth_equ_radius, standard_rst_altitude
from astronomia.dynamical import deltaT_seconds
from astronomia.util import d_to_r, interpolate_angle3, diff_angle, r_to_d, modpi2, interpolate3, equ_to_horiz
import astronomia.globals

class Error(Exception):
    """local exception class"""
    pass

_k1 = d_to_r(360.985647)

def rise(jd, raList, decList, h0, delta):
    """Return the Julian Day of the rise time of an object.
    
    Parameters:
        jd      : Julian Day number of the day in question, at 0 hr UT
        raList  : a sequence of three right accension values, in radians,
            for (jd-1, jd, jd+1)
        decList : a sequence of three right declination values, in radians,
            for (jd-1, jd, jd+1)
        h0      : the standard altitude in radians
        delta   : desired accuracy in days. Times less than one minute are
            infeasible for rise times because of atmospheric refraction.
            
    Returns:
        Julian Day of the rise time
    
    """
    longitude = astronomia.globals.longitude
    latitude = astronomia.globals.latitude
    THETA0 = sidereal_time_greenwich(jd)
    deltaT_days = deltaT_seconds(jd) / seconds_per_day

    cosH0 = (sin(h0) - sin(latitude) * sin(decList[1])) / (cos(latitude) * cos(decList[1]))
    #
    # future: return some indicator when the object is circumpolar or always
    # below the horizon.
    #
    if cosH0 < -1.0: # circumpolar
        return None
    if cosH0 > 1.0:  # never rises
        return None

    H0 = acos(cosH0)    
    m0 = (raList[1] + longitude - THETA0) / pi2
    m = m0 - H0 / pi2  # this is the only difference between rise() and set()
    if m < 0:
        m += 1
    elif m > 1:
        m -= 1
    if not 0 <= m <= 1:
        raise Error, "m is out of range = " + str(m)
    for bailout in range(20):
        m0 = m
        theta0 = modpi2(THETA0 + _k1 * m)
        n = m + deltaT_days
        if not -1 < n < 1:
            return None # Bug: this is where we drop some events
        ra = interpolate_angle3(n, raList)
        dec = interpolate3(n, decList)
        H = theta0 - longitude - ra
#        if H > pi:
#            H = H - pi2
        H = diff_angle(0.0, H)
        A, h = equ_to_horiz(H, dec)
        dm = (h - h0) / (pi2 * cos(dec) * cos(latitude) * sin(H))
        m += dm
        if abs(m - m0) < delta:
            return jd + m

    raise Error, "bailout"
    

def set(jd, raList, decList, h0, delta):
    """Return the Julian Day of the set time of an object.
    
    Parameters:
        jd      : Julian Day number of the day in question, at 0 hr UT
        raList  : a sequence of three right accension values, in radians,
            for (jd-1, jd, jd+1)
        decList : a sequence of three right declination values, in radians,
            for (jd-1, jd, jd+1)
        h0      : the standard altitude in radians
        delta   : desired accuracy in days. Times less than one minute are
            infeasible for set times because of atmospheric refraction.
            
    Returns:
        Julian Day of the set time
    
    """
    longitude = astronomia.globals.longitude
    latitude = astronomia.globals.latitude
    THETA0 = sidereal_time_greenwich(jd)
    deltaT_days = deltaT_seconds(jd) / seconds_per_day

    cosH0 = (sin(h0) - sin(latitude) * sin(decList[1])) / (cos(latitude) * cos(decList[1]))
    #
    # future: return some indicator when the object is circumpolar or always
    # below the horizon.
    #
    if cosH0 < -1.0: # circumpolar
        return None
    if cosH0 > 1.0:  # never rises
        return None

    H0 = acos(cosH0)    
    m0 = (raList[1] + longitude - THETA0) / pi2
    m = m0 + H0 / pi2  # this is the only difference between rise() and set()
    if m < 0:
        m += 1
    elif m > 1:
        m -= 1
    if not 0 <= m <= 1:
        raise Error, "m is out of range = " + str(m)
    for bailout in range(20):
        m0 = m
        theta0 = modpi2(THETA0 + _k1 * m)
        n = m + deltaT_days
        if not -1 < n < 1:
            return None # Bug: this is where we drop some events
        ra = interpolate_angle3(n, raList)
        dec = interpolate3(n, decList)
        H = theta0 - longitude - ra
#        if H > pi:
#            H = H - pi2
        H = diff_angle(0.0, H)
        A, h = equ_to_horiz(H, dec)
        dm = (h - h0) / (pi2 * cos(dec) * cos(latitude) * sin(H))
        m += dm
        if abs(m - m0) < delta:
            return jd + m

    raise Error, "bailout"
    

def transit(jd, raList, delta):
    """Return the Julian Day of the transit time of an object.
    
    Parameters:
        jd      : Julian Day number of the day in question, at 0 hr UT
        raList  : a sequence of three right accension values, in radians,
            for (jd-1, jd, jd+1)
        delta   : desired accuracy in days. 
            
    Returns:
        Julian Day of the transit time
    
    """
    #
    # future: report both upper and lower culmination, and transits of objects below 
    # the horizon
    # 
    longitude = astronomia.globals.longitude
    THETA0 = sidereal_time_greenwich(jd)
    deltaT_days = deltaT_seconds(jd) / seconds_per_day
    
    m = (raList[1] + longitude - THETA0) / pi2
    if m < 0:
        m += 1
    elif m > 1:
        m -= 1
    if not 0 <= m <= 1:
        raise Error, "m is out of range = " + str(m)
    for bailout in range(20):
        m0 = m
        theta0 = modpi2(THETA0 + _k1 * m)
        n = m + deltaT_days
        if not -1 < n < 1:
            return None # Bug: this is where we drop some events
        ra = interpolate_angle3(n, raList)
        H = theta0 - longitude - ra
#        if H > pi:
#            H = H - pi2
        H = diff_angle(0.0, H)
        dm = -H/pi2
        m += dm
        if abs(m - m0) < delta:
            return jd + m

    raise Error, "bailout"
    

def moon_rst_altitude(r):
    """ Returnn the standard altitude of the Moon.
    
    Parameters:
        r : Distance between the centers of the Earth and Moon, in km.
            
    Returns:
        Standard altitude in radians.
    
    """
    # horizontal parallax
    parallax = asin(earth_equ_radius / r)
    
    return 0.7275 * parallax + standard_rst_altitude

