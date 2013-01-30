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

    Collection of miscellaneous functions
    """

from math import modf, cos, sin, asin, tan, atan2, pi
import os
import shlex
import astronomia.globals
from astronomia.constants import pi2, minutes_per_day, seconds_per_day


class Error(Exception):
    """Local exception class"""
    pass


def d_to_dms(x):
    """Convert an angle in decimal degrees to degree components.
    
    Return a tuple (degrees, minutes, seconds). Degrees and minutes
    will be integers, seconds may be floating.
    
    If the argument is negative: 
        The return value of degrees will be negative. 
        If degrees is 0, minutes will be negative. 
        If minutes is 0, seconds will be negative.
        
    Parameters:
        x : degrees
    
    Returns:
        degrees
        minutes
        seconds
        
    """
    frac, degrees = modf(x)
    seconds, minutes = modf(frac*60)
    return int(degrees), int(minutes), seconds*60


#
# Local constants
#
_DtoR = pi / 180.0

def d_to_r(d):
    """Convert degrees to radians.
    
    Parameters:
        d : degrees
        
    Returns:
        radians
    
    """
    return d * _DtoR


def diff_angle(a, b):
    """Return angle b - a, accounting for circular values.
    
    Parameters a and b should be in the range 0..pi*2. The
    result will be in the range -pi..pi.
    
    This allows us to directly compare angles which cross through 0:
    
        359 degress... 0 degrees... 1 degree... etc
        
    Parameters:
        a : first angle, in radians
        b : second angle, in radians
        
    Returns:
        b - a, in radians
    
    """
    if b < a:
        result = b + pi2 - a
    else:
        result = b - a
    if result > pi:
        result -= pi2
    return result


def dms_to_d(deg, minute, sec):
    """Convert an angle in degree components to decimal degrees. 
    
    If any of the components are negative the result will also be negative.
    
    Parameters:
        deg : degrees
        minute : minutes
        sec : seconds
        
    Returns:
        decimal degrees
    
    """
    result = abs(deg) + abs(minute)/60.0 + abs(sec)/3600.0
    if deg < 0 or minute < 0 or sec < 0:
        result = -result
    return result


def ecl_to_equ(longitude, latitude, obliquity):
    """Convert ecliptic to equitorial coordinates. 
    
    [Meeus-1998: equations 13.3, 13.4]
    
    Parameters:
        longitude : ecliptic longitude in radians
        latitude : ecliptic latitude in radians
        obliquity : obliquity of the ecliptic in radians
    
    Returns:
        Right accension in radians
        Declination in radians
    
    """
    cose = cos(obliquity)
    sine = sin(obliquity)
    sinl = sin(longitude)
    ra = modpi2(atan2(sinl * cose - tan(latitude) * sine, cos(longitude)))
    dec = asin(sin(latitude) * cose + cos(latitude) * sine * sinl)
    return ra, dec


def equ_to_horiz(H, decl):
    """Convert equitorial to horizontal coordinates.
    
    [Meeus-1998: equations 13.5, 13.6]

    Note that azimuth is measured westward starting from the south.
    
    This is not a good formula for using near the poles.
    
    Parameters:
        H : hour angle in radians
        decl : declination in radians
        
    Returns:
        azimuth in radians
        altitude in radians
    
    """
    cosH = cos(H)
    sinLat = sin(astronomia.globals.latitude)
    cosLat = cos(astronomia.globals.latitude)
    A = atan2(sin(H), cosH * sinLat - tan(decl) * cosLat)
    h = asin(sinLat * sin(decl) + cosLat * cos(decl) * cosH)
    return A, h
    
    
def equ_to_ecl(ra, dec, obliquity):
    """Convert equitorial to ecliptic coordinates. 
    
    [Meeus-1998: equations 13.1, 13.2]
    
    Parameters:
        ra : right accension in radians
        dec : declination in radians
        obliquity : obliquity of the ecliptic in radians
        
    Returns:
        ecliptic longitude in radians
        ecliptic latitude in radians
    
    """
    cose = cos(obliquity)
    sine = sin(obliquity)
    sina = sin(ra)
    longitude = modpi2(atan2(sina * cose + tan(dec) * sine, cos(ra)))
    latitude = modpi2(asin(sin(dec) * cose - cos(dec) * sine * sina))
    return longitude, latitude
    

def fday_to_hms(day):
    """Convert fractional day (0.0..1.0) to integral hours, minutes, seconds.

    Parameters:
        day : a fractional day in the range 0.0..1.0
        
    Returns:
        hour : 0..23
        minute : 0..59
        seccond : 0..59
    
    """
    frac, hours = modf(day * 24)
    seconds, minutes = modf(frac * 60)
    return int(hours), int(minutes), int(seconds * 60)


def hms_to_fday(hr, mn, sec):
    """Convert hours-minutes-seconds into a fractional day 0.0..1.0.
    
    Parameters:
        hr : hours, 0..23
        mn : minutes, 0..59
        sec : seconds, 0..59
        
    Returns:
        fractional day, 0.0..1.0
    
    """
    return ((hr / 24.0) + (mn / minutes_per_day) + (sec / seconds_per_day))
          

def interpolate3(n, y):
    """Interpolate from three equally spaced tabular values.
    
    [Meeus-1998; equation 3.3]
    
    Parameters:
        n : the interpolating factor, must be between -1 and 1
        y : a sequence of three values 
    
    Results:
        the interpolated value of y
        
    """
    if not -1 < n < 1:
        raise Error, "interpolating factor out of range: " + str(n)
        
    a = y[1] - y[0]
    b = y[2] - y[1]
    c = b - a
    return y[1] + n/2 * (a + b + n*c)


def interpolate_angle3(n, y):
    """Interpolate from three equally spaced tabular angular values.
    
    [Meeus-1998; equation 3.3]
    
    This version is suitable for interpolating from a table of
    angular values which may cross the origin of the circle, 
    for example: 359 degrees...0 degrees...1 degree.
    
    Parameters:
        n : the interpolating factor, must be between -1 and 1
        y : a sequence of three values 
    
    Results:
        the interpolated value of y
        
    """
    if not -1 < n < 1:
        raise Error, "interpolating factor out of range: " + str(n)

    a = diff_angle(y[0], y[1])
    b = diff_angle(y[1], y[2])
    c = diff_angle(a, b)
    return y[1] + n/2 * (a + b + n*c)


def load_params():
    """Read a parameter file and assign global values.
    
    Parameters:
        none

    Returns: 
        nothing
    
    """
    try:
        f = file(os.environ.get("ASTROLABE_PARAMS", "astronomia_params.txt"))
    except IOError, value:
        print value
        raise Error, "Unable to open param file. Either set ASTROLABE_PARAMS correctly or create astronomia_params.txt in the current directory"

    lex = shlex.shlex(f)
    lex.wordchars = lex.wordchars + '.-/\\:'   # tokens and values can have dots, dashes, slashes, colons
    token = lex.get_token()
    while token:
        if token == "standard_timezone_name":
            astronomia.globals.standard_timezone_name = lex.get_token()
        elif token == "standard_timezone_offset":
            offset = float(lex.get_token())
            unit = lex.get_token().lower()
            if unit not in ("day", "days", "hour", "hours", "minute", "minutes", "second", "seconds"):
                raise Error, 'bad value for standard_timezone_offset units'
            if unit in ("hour", "hours"):
                offset /= 24.0
            elif unit in ("minute", "minutes"):
                offset /= minutes_per_day
            elif unit in ("second", "seconds"):
                offset /= seconds_per_day
            astronomia.globals.standard_timezone_offset = offset                
        elif token == "daylight_timezone_name":
            astronomia.globals.daylight_timezone_name = lex.get_token()
        elif token == "daylight_timezone_offset":
            offset = float(lex.get_token())
            unit = lex.get_token().lower()
            if unit not in ("day", "days", "hour", "hours", "minute", "minutes", "second", "seconds"):
                raise Error, 'bad value for standard_timezone_offset units'
            if unit in ("hour", "hours"):
                offset /= 24.0
            elif unit in ("minute", "minutes"):
                offset /= minutes_per_day
            elif unit in ("second", "seconds"):
                offset /= seconds_per_day
            astronomia.globals.daylight_timezone_offset = offset
        elif token == "longitude":
            longitude = float(lex.get_token())
            direction = lex.get_token().lower()
            if direction not in ("east","west"):
                raise Error, 'longitude direction must be "west" or "east"'
            if direction == "east":
                longitude = -longitude
            astronomia.globals.longitude = d_to_r(longitude)
        elif token == "latitude":
            latitude = float(lex.get_token())
            direction = lex.get_token().lower()
            if direction not in ("north","south"):
                raise Error, 'latitude direction must be "north" or "south"'
            if direction == "south":
                latitude = -latitude
            astronomia.globals.latitude = d_to_r(latitude)
        elif token == "vsop87d_text_path":
            astronomia.globals.vsop87d_text_path = lex.get_token()
        elif token == "vsop87d_binary_path":
            astronomia.globals.vsop87d_binary_path = lex.get_token()
        else:
            raise Error, "unknown token %s at line %d in param file" % (token, lex.lineno)
        token = lex.get_token()

    f.close()


def modpi2(x):
    """Reduce an angle in radians to the range 0..2pi.
    
    Parameters:
        x : angle in radians
        
    Returns:
        angle in radians in the range 0..2pi
    
    """
    return x % pi2
    

def polynomial(terms, x):
    """Evaluate a simple polynomial.
    
    Where: terms[0] is constant, terms[1] is for x, terms[2] is for x^2, etc.
    
    Example:
        y = polynomial((1.1, 2.2, 3.3, 4.4), t)
        
        returns the value of:
        
            1.1 + 2.2 * t + 3.3 * t^2 + 4.4 * t^3
    
    Parameters:
        terms : sequence of coefficients
        x : variable value
        
    Results:
        value of the polynomial
    
    """

    result = 0.0
    for index, i in enumerate(terms):
        result = result + i*x**index
    return result
    
#
# Local constants
#
_RtoD = 180.0 / pi

def r_to_d(r):
    """Convert radians to degrees.
    
    Parameters:
        r : radians
        
    Returns:
        degrees
    
    """
    return r * _RtoD
    
