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

"""A collection of date and time functions.

The functions which use Julian Day Numbers are valid only for positive values,
i.e., for dates after -4712 (4713BC).

Unless otherwise specified, Julian Day Numbers may be fractional values.

Numeric years use the astronomical convention of a year 0: 0 = 1BC, -1 = 2BC,
etc.

Numeric months are 1-based: Jan = 1...Dec = 12.

Numeric days are the same as the calendar value.

Reference: Jean Meeus, _Astronomical Algorithms_, second edition, 1998,
Willmann-Bell, Inc.

"""
from math import modf
from astronomia.util import d_to_r, modpi2, fday_to_hms, hms_to_fday

import astronomia.globals


class Error(Exception):
    """local exception class"""
    pass


def cal_to_jd(yr, mo = 1, day = 1, gregorian = True):
    """Convert a date in the Julian or Gregorian calendars to the Julian Day
    Number (Meeus 7.1).

    Parameters:
        yr        : year
        mo        : month (default: 1)
        day       : day, may be fractional day (default: 1)
        gregorian : If True, use Gregorian calendar, else use Julian calendar
        (default: True)

    Return:
        jd        : julian day number

    """
    if mo <= 2:
        yr -= 1
        mo += 12
    if gregorian:
        A = int(yr / 100)
        B = 2 - A + (A / 4)
    else:
        B = 0
    return int(365.25 * (yr + 4716)) + int(30.6001 * (mo + 1)) + day + B - 1524.5


def cal_to_jde(yr, mo = 1, day = 1, hr = 0, mn = 0, sc = 0.0, gregorian = True):
    """Convert a date in the Julian or Gregorian calendars to the Julian Day
    Ephemeris (Meeus 22.1).

    Parameters:
        yr        : year
        mo        : month (default: 1)
        day       : day, may be fractional day (default: 1)
        hr        : hour
        mn        : minute
        sc        : second
        gregorian : If True, use Gregorian calendar, else use Julian calendar
        (default: True)

    Return:
        jde        : julian day ephemeris

    """
    jde = cal_to_jd(yr, mo, day, gregorian)
    return jde + hms_to_fday(hr, mn, sc)


def cal_to_day_of_year(yr, mo, dy, gregorian = True):
    """Convert a date in the Julian or Gregorian calendars to day of the year
    (Meeus 7.1).

    Parameters:
        yr        : year
        mo        : month
        day       : day
        gregorian : If True, use Gregorian calendar, else use Julian calendar
        (default: True)

    Return:
        day number : 1 = Jan 1...365 (or 366 for leap years) = Dec 31.

    """
    if is_leap_year(yr, gregorian):
        K = 1
    else:
        K = 2
    dy = int(dy)
    return int(275 * mo / 9.0) - (K * int((mo + 9) / 12.0)) + dy - 30


def day_of_year_to_cal(yr, N, gregorian = True):
    """Convert a day of year number to a month and day in the Julian or
    Gregorian calendars.

    Parameters:
        yr        : year
        N         : day of year, 1..365 (or 366 for leap years)
        gregorian : If True, use Gregorian calendar, else use Julian calendar
        (default: True)

    Return:
        month
        day

    """
    if is_leap_year(yr, gregorian):
        K = 1
    else:
        K = 2
    if (N < 32):
        mo = 1
    else:
        mo = int(9 * (K+N) / 275.0 + 0.98)
    dy = int(N - int(275 * mo / 9.0) + K * int((mo + 9) / 12.0) + 30)
    return mo, dy


def easter(yr, gregorian = True):
    """Return the date of Western ecclesiastical Easter for a year in the
    Julian or Gregorian calendars.

    Parameters:
        yr        : year
        gregorian : If True, use Gregorian calendar, else use Julian calendar
        (default: True)

    Return:
        month
        day

    """
    yr = int(yr)
    if gregorian:
        a = yr % 19
        b = yr / 100
        c = yr % 100
        d = b / 4
        e = b % 4
        f = (b + 8) / 25
        g = (b - f + 1) / 3
        h = (19 * a + b - d - g + 15) % 30
        i = c / 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) / 451
        tmp = h + l - 7 * m + 114
    else:
        a = yr % 4
        b = yr % 7
        c = yr % 19
        d = (19 * c + 15) % 30
        e = (2 * a + 4 * b - d + 34) % 7
        tmp = d + e + 114
    mo = tmp / 31
    dy = (tmp % 31) + 1
    return mo, dy


def is_dst(jd):
    """Is this instant within the Daylight Savings Time period as used in the
    US?

    If astronomia.globals.daylight_timezone_name is None, the function always
    returns False.

    Parameters:
        jd : Julian Day number representing an instant in Universal Time

    Return:
        True if Daylight Savings Time is in effect, False otherwise.

    """
    if not astronomia.globals.daylight_timezone_name:
        return False

    #
    # What year is this?
    #
    yr, mon, day = jd_to_cal(jd)
    #
    # First day in April
    #
    start = cal_to_jd(yr, 4, 1)

    #
    # Advance to the first Sunday
    #
    dow = jd_to_day_of_week(start)
    if dow:
        start += 7 - dow

    #
    # Advance to 2AM
    #
    start += 2.0 / 24

    #
    # Convert to Universal Time
    #
    start += astronomia.globals.standard_timezone_offset

    if jd < start:
        return False

    #
    # Last day in October
    #
    stop = cal_to_jd(yr, 10, 31)

    #
    # Backup to the last Sunday
    #
    dow = jd_to_day_of_week(stop)
    stop -= dow

    #
    # Advance to 2AM
    #
    stop += 2.0 / 24

    #
    # Convert to Universal Time
    #
    stop += astronomia.globals.daylight_timezone_offset

    if jd < stop:
        return True

    return False


def is_leap_year(yr, gregorian = True):
    """Return True if this is a leap year in the Julian or Gregorian calendars

    Parameters:
        yr        : year
        gregorian : If True, use Gregorian calendar, else use Julian calendar
        (default: True)

    Return:
        True is this is a leap year, else False.

    """
    yr = int(yr)
    if gregorian:
        return (yr % 4 == 0) and ((yr % 100 != 0) or (yr % 400 == 0))
    else:
        return yr % 4 == 0


def jd_to_cal(jd, gregorian = True):
    """Convert a Julian day number to a date in the Julian or Gregorian
    calendars.

    Parameters:
        jd        : Julian Day number
        gregorian : If True, use Gregorian calendar, else use Julian calendar
        (default: True)

    Return:
        year
        month
        day (may be fractional)

    Return a tuple (year, month, day).

    """
    F, Z = modf(jd + 0.5)
    if gregorian:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)
    else:
        A = Z
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day = B - D - int(30.6001 * E) + F
    if E < 14:
        mo = E - 1
    else:
        mo = E - 13
    if mo > 2:
        yr = C - 4716
    else:
        yr = C - 4715
    return yr, mo, day


def jd_to_day_of_week(jd):
    """Return the day of week for a Julian Day Number.

    The Julian Day Number must be for 0h UT.

    Parameters:
        jd : Julian Day number

    Return:
        day of week: 0 = Sunday...6 = Saturday.

    """
    i = jd + 1.5
    return int(i) % 7


def jd_to_jcent(jd):
    """Return the number of Julian centuries since J2000.0

    Parameters:
        jd : Julian Day number

    Return:
        Julian centuries

    """
    return (jd - 2451545.0) / 36525.0


def lt_to_str(jd, zone = "", level = "second"):
    """Convert local time in Julian Days to a formatted string.

    The general format is:

        YYYY-MMM-DD HH:MM:SS ZZZ

    Truncate the time value to seconds, minutes, hours or days as
    indicated. If level = "day", don't print the time zone string.

    Pass an empty string ("", the default) for zone if you want to do
    your own zone formatting in the calling module.

    Parameters:
        jd    : Julian Day number
        zone  : Time zone string (default = "")
        level : "day" or "hour" or "minute" or "second" (default = "second")

    Return:
        formatted date/time string

    """
    yr, mon, day = jd_to_cal(jd)
    fday, iday = modf(day)
    iday = int(iday)
    hr, mn, sec = fday_to_hms(fday)
    sec = int(sec)

    month = astronomia.globals.month_names[mon-1]

    if level == "second":
        return "%d-%s-%02d %02d:%02d:%02d %s" % (yr, month, iday, hr, mn, sec, zone)
    if level == "minute":
        return "%d-%s-%02d %02d:%02d %s" % (yr, month, iday, hr, mn, zone)
    if level == "hour":
        return "%d-%s-%02d %02d %s" % (yr, month, iday, hr, zone)
    if level == "day":
        return "%d-%s-%02d" % (yr, astronomia.globals.month_names[mon-1], iday)
    raise Error, "unknown time level = " + level


def sidereal_time_greenwich(jd):
    """Return the mean sidereal time at Greenwich.

    The Julian Day number must represent Universal Time.

    Parameters:
        jd : Julian Day number

    Return:
        sidereal time in radians (2pi radians = 24 hrs)

    """
    T = jd_to_jcent(jd)
    T2 = T * T
    T3 = T2 * T
    theta0 = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T2 - T3/38710000
    result = d_to_r(theta0)
    return modpi2(result)


def ut_to_lt(jd):
    """Convert universal time in Julian Days to a local time.

    Include Daylight Savings Time offset, if any.

    Parameters:
        jd : Julian Day number, universal time

    Return:
        Julian Day number, local time
        zone string of the zone used for the conversion

    """
    if is_dst(jd):
        zone = astronomia.globals.daylight_timezone_name
        offset = astronomia.globals.daylight_timezone_offset
    else:
        zone = astronomia.globals.standard_timezone_name
        offset = astronomia.globals.standard_timezone_offset

    jd = jd - offset
    return jd, zone
