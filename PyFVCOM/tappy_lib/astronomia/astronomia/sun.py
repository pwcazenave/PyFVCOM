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

""" Geocentric solar position and radius, both low and high precision.

"""
from math import *
from astronomia.calendar import jd_to_jcent
from astronomia.util import polynomial, d_to_r, modpi2, dms_to_d
from astronomia.vsop87d import VSOP87d

class Error(Exception):
    """Local exception class"""
    pass

class Sun:
    """High precision position calculations.
    
    This is a very light wrapper around the VSOP87d class. The geocentric
    longitude of the Sun is simply the heliocentric longitude of the Earth +
    180 degrees. The geocentric latitude of the Sun is the negative of the
    heliocentric latitude of the Earth. The radius is of course the same in
    both coordinate systems.
    
    """
    def __init__(self):
        self.vsop = VSOP87d()

    def mean_longitude(self, jd):
        """Return mean longitude.
        
        Parameters:
            jd : Julian Day in dynamical time

        Returns:
            Longitude in radians
                
        """
        T = jd_to_jcent(jd)

        # From astrolabe
        #X = polynomial((d_to_r(100.466457), d_to_r(36000.7698278), d_to_r(0.00030322), d_to_r(0.000000020)), T)

        # From AA, Naughter
        # Takes T/10.0
        X = polynomial((d_to_r(100.4664567), d_to_r(360007.6982779), d_to_r(0.03032028), d_to_r(1.0/49931), d_to_r(-1.0/15300), d_to_r(-1.0/2000000)), T/10.0)

        X = modpi2(X + pi)
        return X


    def mean_longitude_perigee(self, jd):
        """Return mean longitude of solar perigee.
        
        Parameters:
            jd : Julian Day in dynamical time

        Returns:
            Longitude of solar perigee in radians
                
        """
        T = jd_to_jcent(jd)

        X = polynomial((1012395.0, 
                        6189.03  ,
                        1.63     , 
                        0.012    ), (T + 1))/3600.0
        X = d_to_r(X)


        X = modpi2(X)
        return X

    def dimension(self, jd, dim):
        """Return one of geocentric ecliptic longitude, latitude and radius.
        
        Parameters:
            jd : Julian Day in dynamical time
            dim : one of "L" (longitude) or "B" (latitude) or "R" (radius).

        Returns:
            Either longitude in radians, or
            latitude in radians, or
            radius in au.
                
        """
        X = self.vsop.dimension(jd, "Earth", dim)
        if dim == "L":
            X = modpi2(X + pi)
        elif dim == "B":
            X = -X
        return X

    def dimension3(self, jd):
        """Return geocentric ecliptic longitude, latitude and radius.
        
        Parameters:
            jd : Julian Day in dynamical time

        Returns:
            longitude in radians
            latitude in radians
            radius in au
        
        """
        L = self.dimension(jd, "L")
        B = self.dimension(jd, "B")
        R = self.dimension(jd, "R")
        return L, B, R

#
# Constant terms
#
_kL0 = (d_to_r(280.46646),  d_to_r(36000.76983),  d_to_r( 0.0003032))
_kM  = (d_to_r(357.52911),  d_to_r(35999.05029),  d_to_r(-0.0001537))
_kC  = (d_to_r(  1.914602), d_to_r(   -0.004817), d_to_r(-0.000014))

_ck3 = d_to_r( 0.019993)
_ck4 = d_to_r(-0.000101)
_ck5 = d_to_r( 0.000289)

def longitude_radius_low(jd):
    """Return geometric longitude and radius vector. 
    
    Low precision. The longitude is accurate to 0.01 degree.  The latitude
    should be presumed to be 0.0. [Meeus-1998: equations 25.2 through 25.5
    
    Parameters:
        jd : Julian Day in dynamical time

    Returns:
        longitude in radians
        radius in au

    """
    T = jd_to_jcent(jd)
    L0 = polynomial(_kL0, T)
    M = polynomial(_kM, T)
    e = polynomial((0.016708634, -0.000042037, -0.0000001267), T)
    C = polynomial(_kC, T) * sin(M) \
        + (_ck3 - _ck4 * T) * sin(2 * M) \
        + _ck5 * sin(3 * M)
    L = modpi2(L0 + C)
    v = M + C
    R = 1.000001018 * (1 - e * e) / (1 + e * cos(v))
    return L, R


#
# Constant terms
#
_lk0 = d_to_r(125.04)
_lk1 = d_to_r(1934.136)
_lk2 = d_to_r(0.00569)
_lk3 = d_to_r(0.00478)

def apparent_longitude_low(jd, L):
    """Correct the geometric longitude for nutation and aberration.
    
    Low precision. [Meeus-1998: pg 164]
    
    Parameters:
        jd : Julian Day in dynamical time
        L : longitude in radians

    Returns:
        corrected longitude in radians

    """    
    T = jd_to_jcent(jd)
    omega = _lk0 - _lk1 * T
    return modpi2(L - _lk2 - _lk3 * sin(omega))
    

#
# Constant terms
#
_lk4 = d_to_r(dms_to_d(0, 0, 20.4898))

def aberration_low(R):
    """Correct for aberration; low precision, but good enough for most uses. 
    
    [Meeus-1998: pg 164]
    
    Parameters:
        R : radius in au

    Returns:
        correction in radians

    """
    return -_lk4 / R
    
