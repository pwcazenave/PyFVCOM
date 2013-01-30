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

"""Useful constants.

Don't change these unless you are moving to a new universe.

"""
from math import pi

#
# The number of radians in a circle.
#
pi2 = 2 * pi

#
# Ratio of Earth's polar to equitorial radius.
#
#flattening = 0.99664719
flattening = 0.99664713901477464

#
# Equitorial radius of the Earth in km.
#
earth_equ_radius = 6378.135

#
# How many minutes in a day?
#
minutes_per_day = 24.0 * 60.0

#
# How many days in minute?
#
days_per_minute = 1.0 / minutes_per_day

#
# How many seconds (time) in a day?
#
seconds_per_day = 24.0 * 60.0 * 60.0

#
# How many days in a second?
#
days_per_second = 1.0 / seconds_per_day

#
# How many kilometers in an astronomical unit?
#
# km_per_au = 149597870
# More accurate? Does it matter?
km_per_au = 149597870.691

#
# For rise-set-transit: altitude deflection caused by refraction
#
standard_rst_altitude = -0.00989078087105 # -0.5667 degrees
sun_rst_altitude = -0.0145438286569       # -0.8333 degrees
