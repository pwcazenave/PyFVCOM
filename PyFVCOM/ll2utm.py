#!/usr/bin/env python

# Lat Long - UTM, UTM - Lat Long conversions

# Adjusted for use with NumPy by Pierre Cazenave <pica {at} pml.ac.uk>

import inspect

import numpy as np

from warnings import warn

# LatLong- UTM conversion..h
# definitions for lat/long to UTM and UTM to lat/lng conversions
# include <string.h>

_EquatorialRadius = 2
_eccentricitySquared = 3

_ellipsoid = [
    # id, Ellipsoid name, Equatorial Radius, square of eccentricity first once
    # is a placeholder only, To allow array indices to match id numbers
    [-1, "Placeholder", 0, 0],
    [1, "Airy", 6377563, 0.00667054],
    [2, "Australian National", 6378160, 0.006694542],
    [3, "Bessel 1841", 6377397, 0.006674372],
    [4, "Bessel 1841 (Nambia] ", 6377484, 0.006674372],
    [5, "Clarke 1866", 6378206, 0.006768658],
    [6, "Clarke 1880", 6378249, 0.006803511],
    [7, "Everest", 6377276, 0.006637847],
    [8, "Fischer 1960 (Mercury] ", 6378166, 0.006693422],
    [9, "Fischer 1968", 6378150, 0.006693422],
    [10, "GRS 1967", 6378160, 0.006694605],
    [11, "GRS 1980", 6378137, 0.00669438],
    [12, "Helmert 1906", 6378200, 0.006693422],
    [13, "Hough", 6378270, 0.00672267],
    [14, "International", 6378388, 0.00672267],
    [15, "Krassovsky", 6378245, 0.006693422],
    [16, "Modified Airy", 6377340, 0.00667054],
    [17, "Modified Everest", 6377304, 0.006637847],
    [18, "Modified Fischer 1960", 6378155, 0.006693422],
    [19, "South American 1969", 6378160, 0.006694542],
    [20, "WGS 60", 6378165, 0.006693422],
    [21, "WGS 66", 6378145, 0.006694542],
    [22, "WGS-72", 6378135, 0.006694318],
    [23, "WGS-84", 6378137, 0.00669438]
]

# Reference ellipsoids derived from Peter H. Dana's website:
# http://www.utexas.edu/depts/grg/gcraft/notes/datum/elist.html
# Department of Geography,
# University of Texas at Austin
# Email: pdana@mail.utexas.edu
# 3/22/95

# Source:
# Defense Mapping Agency. 1987b. DMA Technical Report: Supplement to Department
# of Defense World Geodetic System 1984 Technical Report. Part I and II.
# Washington, DC: Defense Mapping Agency


def _test(inLat, inLong, inZone=False):
    """ Simple test of the functions with a given lat/long pair.

    Parameters
    ----------
    inLat, inLong : float
        Input latitude and longitude pair.

    """

    z, e, n = LL_to_UTM(23, inLat, inLong, ZoneNumber=inZone)
    outLat, outLong = UTM_to_LL(23, n, e, z)

    return z, e, n, outLat, outLong


def _UTM_letter_designator(Lat):
    # This routine determines the correct UTM letter designator for the given
    # latitude. Returns 'Z' if latitude is outside the UTM limits of 84N to 80S
    # Written by Chuck Gantz: chuck.gantz@globalstar.com

    if 84 >= Lat >= 72:
        return 'X'
    elif 72 > Lat >= 64:
        return 'W'
    elif 64 > Lat >= 56:
        return 'V'
    elif 56 > Lat >= 48:
        return 'U'
    elif 48 > Lat >= 40:
        return 'T'
    elif 40 > Lat >= 32:
        return 'S'
    elif 32 > Lat >= 24:
        return 'R'
    elif 24 > Lat >= 16:
        return 'Q'
    elif 16 > Lat >= 8:
        return 'P'
    elif 8 > Lat >= 0:
        return 'N'
    elif 0 > Lat >= -8:
        return 'M'
    elif -8 > Lat >= -16:
        return 'L'
    elif -16 > Lat >= -24:
        return 'K'
    elif -24 > Lat >= -32:
        return 'J'
    elif -32 > Lat >= -40:
        return 'H'
    elif -40 > Lat >= -48:
        return 'G'
    elif -48 > Lat >= -56:
        return 'F'
    elif -56 > Lat >= -64:
        return 'E'
    elif -64 > Lat >= -72:
        return 'D'
    elif -72 > Lat >= -80:
        return 'C'
    else:
        return 'Z'    # if the Latitude is outside the UTM limits

    # void UTM_to_LL(int ReferenceEllipsoid, const double UTMNorthing,
    #             const double UTMEasting, const char* UTMZone,
    #             double& Lat,  double& Long )


def LL_to_UTM(ReferenceEllipsoid, Lat, Long, ZoneNumber=False):
    """ Converts lat/long to UTM coords. Equations from USGS Bulletin 1532.

    East Longitudes are positive, west longitudes are negative. North latitudes
    are positive, south latitudes are negative. Lat and Long are in decimal
    degrees.

    Parameters
    ----------
    ReferenceEllipsoid : int
        Select from 23 reference ellipsoids:
            1. Airy
            2. Australian National
            3. Bessel 1841
            4. Bessel 1841 (Nambia)
            5. Clarke 1866
            6. Clarke 1880
            7. Everest
            8. Fischer 1960 (Mercury)
            9. Fischer 1968
            10. GRS 1967
            11. GRS 1980
            12. Helmert 1906
            13. Hough
            14. International
            15. Krassovsky
            16. Modified Airy
            17. Modified Everest
            18. Modified Fischer 1960
            19. South American 1969
            20. WGS 60
            21. WGS 66
            22. WGS-72
            23. WGS-84

    Lat, Long : float, ndarray
        Latitude and longitude values as floating point values. Negative
        coordinates are west and south for longitude and latitude respectively.

    ZoneNumber : str, optional
        UTM zone number in which the coordinates should be forced. This is
        useful if the spherical coordinates supplied in Lat and Long exceed
        a single UTM zone. Speficy both number and letter e.g. '30N'. If
        omitted, it is calculated automatically.


    Returns
    -------
    Zone : list
        UTM Zone for the coordinates supplied in Lat and Long.

    UTMEasting, UTMNorthing : ndarray
        Cartesian coordinates for the positions in Lat and Long.

    Notes
    -----
    Written by Chuck Gantz - chuck.gantz@globalstar.com
    Modified by Pierre Cazenave - pica {at} pml <dot> ac (dot) uk

    """

    if ~isinstance(Long, np.ndarray):
        Long = np.asarray(Long)

    if ~isinstance(Lat, np.ndarray):
        Lat = np.asarray(Lat)

    a = _ellipsoid[ReferenceEllipsoid][_EquatorialRadius]
    eccSquared = _ellipsoid[ReferenceEllipsoid][_eccentricitySquared]
    k0 = 0.9996

    # Make sure the longitude is between -180.00 .. 179.9
    LongTemp = (Long + 180) - np.floor((Long + 180) / 360).astype(int) \
            * 360 - 180

    LatRad = np.deg2rad(Lat)
    LongRad = np.deg2rad(LongTemp)

    # Add loop around the Latitudes to get the correct zone information. Build
    # a list of the zones for all coordinates.
    Zone = []
    if np.ndim(Lat) > 0:
        for c, ll in enumerate(zip(np.asarray(LongTemp), Lat)):
            # Generate the ZoneNumber if we haven't been given one.
            if not ZoneNumber:
                ZoneNumberTemp = int((ll[0] + 180) / 6) + 1
            else:
                ZoneNumberTemp = ZoneNumber

            if ll[1] >= 56. and ll[1] < 64. and ll[0] >= 3. and ll[0] < 12.:
                ZoneNumberTemp = 32

            # Special zones for Svalbard
            if ll[1] >= 72.0 and ll[1] < 84.0:
                if ll[0] >= 0.0 and ll[0] < 9.0:
                    ZoneNumberTemp = 31
                elif ll[0] >= 9.0 and ll[0] < 21.0:
                    ZoneNumberTemp = 33
                elif ll[0] >= 21.0 and ll[0] < 33.0:
                    ZoneNumberTemp = 35
                elif ll[0] >= 33.0 and ll[0] < 42.0:
                    ZoneNumberTemp = 37

            # + 3 puts origin in middle of zone
            LongOrigin = (ZoneNumberTemp - 1) * 6 - 180 + 3
            LongOriginRad = np.deg2rad(LongOrigin)

            # Compute the UTM Zone from the latitude and longitude
            UTMZone = "%d%c" % (ZoneNumberTemp, _UTM_letter_designator(ll[1]))

            Zone.append(UTMZone)
    else:
        # Not an array, so assume single values in Long and Lat.

        ll = [Long, Lat]
        # Set the Zone Number if we've not been given one.
        if not ZoneNumber:
            ZoneNumber = int((ll[0] + 180) / 6) + 1

        if ll[1] >= 56.0 and ll[1] < 64.0 and ll[0] >= 3.0 and ll[0] < 12.0:
            ZoneNumber = 32

        # Special zones for Svalbard
        if ll[1] >= 72.0 and ll[1] < 84.0:
            if ll[0] >= 0.0 and ll[0] < 9.0:
                ZoneNumber = 31
            elif ll[0] >= 9.0 and ll[0] < 21.0:
                ZoneNumber = 33
            elif ll[0] >= 21.0 and ll[0] < 33.0:
                ZoneNumber = 35
            elif ll[0] >= 33.0 and ll[0] < 42.0:
                ZoneNumber = 37

        # + 3 puts origin in middle of zone
        LongOrigin = (ZoneNumber - 1) * 6 - 180 + 3
        LongOriginRad = np.deg2rad(LongOrigin)

        # Compute the UTM Zone from the latitude and longitude
        UTMZone = "%d%c" % (ZoneNumber, _UTM_letter_designator(ll[1]))

        Zone = UTMZone

    eccPrimeSquared = (eccSquared) / (1 - eccSquared)
    N = a / np.sqrt(1 - eccSquared * np.sin(LatRad) * np.sin(LatRad))
    T = np.tan(LatRad) * np.tan(LatRad)
    C = eccPrimeSquared * np.cos(LatRad) * np.cos(LatRad)
    A = np.cos(LatRad) * (LongRad - LongOriginRad)

    M = a * ((1 - eccSquared / 4 - 3 * eccSquared * eccSquared / 64 - 5
        * eccSquared * eccSquared * eccSquared / 256) * LatRad
        - (3 * eccSquared / 8 + 3 * eccSquared * eccSquared / 32
            + 45 * eccSquared * eccSquared * eccSquared / 1024)
        * np.sin(2 * LatRad) + (15 * eccSquared * eccSquared / 256
            + 45 * eccSquared * eccSquared * eccSquared / 1024)
        * np.sin(4 * LatRad) - (35 * eccSquared * eccSquared
            * eccSquared / 3072) * np.sin(6 * LatRad))

    UTMEasting = (k0 * N * (A + (1 - T + C) * A * A * A / 6
        + (5 - 18 * T + T * T + 72 * C - 58 * eccPrimeSquared)
        * A * A * A * A * A / 120)
        + 500000.0)

    UTMNorthing = (k0 * (M + N * np.tan(LatRad)
        * (A * A / 2 + (5 - T + 9 * C + 4 * C * C)
            * A * A * A * A / 24 + (61 - 58 * T + T * T + 600 * C - 330
                * eccPrimeSquared) * A * A * A * A * A * A / 720)))

    Northing = []
    if np.ndim(Lat) > 0:
        for c, l in enumerate(Lat):
            if l < 0:
                # 10000000 meter offset for southern hemisphere
                Northing.append(UTMNorthing[c] + 10000000.0)
            else:
                Northing.append(UTMNorthing[c])

        UTMNorthing = Northing
    else:
        if Lat < 0:
            # 10000000 meter offset for southern hemisphere
            UTMNorthing = UTMNorthing + 10000000.0

    return Zone, UTMEasting, UTMNorthing


def UTM_to_LL(ReferenceEllipsoid, northing, easting, zone):
    """ Converts UTM coords to lat/long. Equations from USGS Bulletin 1532.

    East Longitudes are positive, west longitudes are negative. North latitudes
    are positive, south latitudes are negative. Lat and Long are in decimal
    degrees.

    Parameters
    ----------
    ReferenceEllipsoid : int
        Select from 23 reference ellipsoids:
            1. Airy
            2. Australian National
            3. Bessel 1841
            4. Bessel 1841 (Nambia)
            5. Clarke 1866
            6. Clarke 1880
            7. Everest
            8. Fischer 1960 (Mercury)
            9. Fischer 1968
            10. GRS 1967
            11. GRS 1980
            12. Helmert 1906
            13. Hough
            14. International
            15. Krassovsky
            16. Modified Airy
            17. Modified Everest
            18. Modified Fischer 1960
            19. South American 1969
            20. WGS 60
            21. WGS 66
            22. WGS-72
            23. WGS-84

    northing, easting : float, ndarray
        Latitude and longitude values as floating point values. Negative
        coordinates are west and south for longitude and latitude respectively.

    zone : str, list, optional
        UTM zone number in which the coordinates are referenced.

    Reutrns
    -------
    Lat, Long : ndarray
        Latitude and longitudes for the coordinates in easting and northing.

    Notes
    -----
    Written by Chuck Gantz: chuck.gantz@globalstar.com.
    Converted to Python by Russ Nelson <nelson@crynwr.com>
    Modified by Pierre Cazenave <pica {at} pml.ac.uk>

    """

    k0 = 0.9996
    a = _ellipsoid[ReferenceEllipsoid][_EquatorialRadius]
    eccSquared = _ellipsoid[ReferenceEllipsoid][_eccentricitySquared]
    e1 = (1 - np.sqrt(1 - eccSquared)) / (1 + np.sqrt(1 - eccSquared))

    if ~isinstance(easting, np.ndarray):
        easting = np.asarray(easting)

    if ~isinstance(northing, np.ndarray):
        northing = np.asarray(northing)

    # remove 500,000 meter offset for longitude
    x = easting - 500000.0
    y = northing

    # If our zone array is a string, we assume it's the same for all positions,
    # which is easy peasy. Otherwise, if it's a list or a numpy array, we need
    # to iterate through each zone to return the correct latitude and
    # longitudes.
    if isinstance(zone, str):
        ZoneLetter = zone[-1]
        ZoneNumber = int(zone[:-1])
        if ZoneLetter >= 'N':
            pass
        else:
            # remove 10,000,000 meter offset used for southern hemisphere
            y -= 10000000.0

        # + 3 puts origin in middle of zone
        LongOrigin = (ZoneNumber - 1) * 6 - 180 + 3

    elif isinstance(zone, np.ndarray) or isinstance(zone, list):
        # Make the necessary output array depending on whether we got a list or
        # an array.
        if isinstance(zone, np.ndarray):
            LongOrigin = np.empty(zone.shape)
        if isinstance(zone, list):
            LongOrigin = np.empty(np.asarray(zone).shape)

        for c, i in enumerate(zone):
            ZoneLetter = zone[c][-1]
            ZoneNumber = int(zone[c][:-1])
            if ZoneLetter >= 'N':
                pass
            else:
                # remove 10,000,000 meter offset used for southern hemisphere
                y -= 10000000.0

            # + 3 puts origin in middle of zone
            LongOrigin[c] = (ZoneNumber - 1) * 6 - 180 + 3

    eccPrimeSquared = (eccSquared) / (1 - eccSquared)

    M = y / k0
    mu = M / (a * (1 - eccSquared / 4 - 3 * eccSquared * eccSquared
        / 64 - 5 * eccSquared * eccSquared * eccSquared / 256))

    phi1Rad = (mu + (3 * e1 / 2 - 27 * e1 * e1 * e1 / 32) * np.sin(2 * mu)
               + (21 * e1 * e1 / 16 - 55 * e1 * e1 * e1 * e1 / 32)
               * np.sin(4 * mu)
               + (151 * e1 * e1 * e1 / 96) * np.sin(6 * mu))

    N1 = a / np.sqrt(1 - eccSquared * np.sin(phi1Rad) * np.sin(phi1Rad))
    T1 = np.tan(phi1Rad) * np.tan(phi1Rad)
    C1 = eccPrimeSquared * np.cos(phi1Rad) * np.cos(phi1Rad)
    R1 = a * (1 - eccSquared) / np.power(1 - eccSquared * np.sin(phi1Rad)
            * np.sin(phi1Rad), 1.5)
    D = x / (N1 * k0)

    Lat = phi1Rad - (N1 * np.tan(phi1Rad) / R1) * (D * D / 2 - (5 + 3 * T1
        + 10 * C1 - 4 * C1 * C1 - 9 * eccPrimeSquared) * D * D * D * D
        / 24 + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252
            * eccPrimeSquared - 3 * C1 * C1) * D * D * D * D * D * D / 720)
    Lat = np.rad2deg(Lat)

    Long = (D - (1 + 2 * T1 + C1) * D * D * D / 6
            + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8
                * eccPrimeSquared + 24 * T1 * T1)
            * D * D * D * D * D / 120) / np.cos(phi1Rad)
    Long = LongOrigin + np.rad2deg(Long)

    return Lat, Long

# For backwards-compatibility.
def LLtoUTM(*args, **kwargs):
    warn('{} is deprecated. Use LL_to_UTM instead.'.format(inspect.stack()[0][3]))
    return LL_to_UTM(*args, **kwargs)


def UTMtoLL(*args, **kwargs):
    warn('{} is deprecated. Use UTM_to_LL instead.'.format(inspect.stack()[0][3]))
    return UTM_to_LL(*args, **kwargs)


if __name__ == '__main__':

    print('\nTest with NumPy single values')
    latTest, lonTest = 50, -5
    z, e, n, outLat, outLong = _test(latTest, lonTest)
    print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
        latTest, lonTest, outLat, outLong))
    print("Intermediate UTM: {}, {}, {}".format(e, n, z))

    print('\nTest with lists')
    latTest, lonTest = [50, 55], [-5, -20]
    z, e, n, outLat, outLong = _test(latTest, lonTest)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}, {}".format(e[c], n[c], z[c]))

    print('\nTest with NumPy arrays')
    latTest, lonTest = np.asarray([50, 55]), np.asarray([-5, -20])
    z, e, n, outLat, outLong = _test(latTest, lonTest)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}, {}".format(e[c], n[c], z[c]))

    print('\nTest with NumPy arrays but a single zone')
    latTest, lonTest = np.asarray([50, 55]), np.asarray([-5, -20])
    z, e, n, outLat, outLong = _test(latTest, lonTest, inZone=30)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}, {}".format(e[c], n[c], z[c]))
