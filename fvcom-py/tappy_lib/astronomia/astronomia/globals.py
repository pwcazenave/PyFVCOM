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

"""Global values.

These can be set directly, or there is a routine astronomia.util.load_params()
which will assign them based on values in a parameter text file.

"""
import os

#
# Abbreviation for standard timezone (e.g., "CST" for North American 
# Central Standard Time)
#
standard_timezone_name = "UT"

#
# Time in fractional days to be subtracted from UT to calculate the standard
# time zone offset. Locations east of Greenwich should use negative values.
#       
standard_timezone_offset = 0.0        

#
# Abbreviation for daylight savings timezone (e.g., "CDT" for North American 
# Central Daylight Time)
#
# This is optional. If set to None, no daylight savings conversions 
# will be performed.
#
daylight_timezone_name = None         

#
# Time in fractional days to be subtracted from UT to calculate the daylight savings
# time zone offset. Locations east of Greenwich should use negative values.
#
# This value is not used unless "daylight_timezone_name" has an value other
# than None.
#       
daylight_timezone_offset = None

#
# Observer's longitude in radians, measured positive west of Greenwich, 
# negative to the east. Should be between -pi...pi.
# 
longitude = 0.0

#
# Observer's latitude in radians, measured positive north of the equator, 
# negative to the south. Should be between -pi/2...pi/2.
#
latitude = 0.0

#
# Month names. There must be twelve. The default is three-character
# abbreviations so that listings line up.
#
month_names = ("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec")

#
# Season names. There must be four. These are used to characterize the
# equinoxes and solstices.
#
season_names = ("spring", "summer", "autumn", "winter")
