"""
Utilities to convert from cartesian UTM coordinates to spherical latitude and
longitudes.

This function uses functions lifted from the utm library. For those that are
interested, the reason we don't just use the utm utility is because it
refuses to do coordinate conversions outside a single zone.

"""

from __future__ import division

import re
import inspect
import multiprocessing
import numpy as np

from pyproj import Proj
from warnings import warn

# Convert a string, tuple, float or int to a list.
to_list = lambda x: [x] if isinstance(x, str) or isinstance(x, float) or isinstance(x, int) else x


def __test(inLat, inLong, inZone=False):
    """ Simple back and forth test of the functions with a given lat/long pair.

    Parameters
    ----------
    inLat, inLong : float
        Input latitude and longitude pair.

    """
    e, n, z = utm_from_lonlat(inLong, inLat, inZone)
    lon, lat = lonlat_from_utm(e, n, z)

    return z, e, n, lon, lat


def __test_compatibility_functions(inLat, inLong, inZone=False):
    """ Simple test of the functions with a given lat/long pair.

    Parameters
    ----------
    inLat, inLong : float
        Input latitude and longitude pair.

    """

    z, e, n = LL_to_UTM(23, inLat, inLong, ZoneNumber=inZone)
    outLat, outLong = UTM_to_LL(23, n, e, z)

    return z, e, n, outLat, outLong


def __convert(args):
    """
    Child function to create a projection object and convert coordinates (optinally inverse).

    This function can therefore to UTM -> lat/lon as well as lat/lon -> UTM

    Parameters
    ----------
    args : tuple
        Arguments:
            in_x, in_y : float
            zone, ellipsoid, datum : str
            inverse : bool

    Returns
    -------
    out_x, out_y : float
        Coordinates following conversion.

    """
    a, b, zone, ellipsoid, datum, inverse = args
    projection = Proj("+proj=utm +zone={}, +ellps={} +datum={} +units=m +no_defs".format(zone, ellipsoid, datum))
    c, d = projection(a, b, inverse=inverse)

    return c, d


def _get_zone_number(longitude, latitude):
    """
    Calculate the UTM zone number from the given coordinate. Shamelessly lifted
    from the utm.lonlat_to_zone_number function.

    Parameters
    ----------
    lon, lat : float
        Longitude and latitude

    Returns
    -------
    zone_number : int
        Zone number

    """

    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37

    return int((longitude + 180) / 6) + 1


def _get_zone_letter(latitude):
    """
    Calculate the UTM zone letter from the given latitude. Shamelessly lifted
    from the utm.latitude_to_zone_letter function.

    Parameters
    ----------
    lat : float
        Latitude

    Returns
    -------
    zone_letter : str
        Zone letter

    """

    ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"

    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    else:
        return None


def utm_from_lonlat(lon, lat, zone=None, ellipsoid='WGS84', datum='WGS84', parallel=False):
    """
    Converts lat/long to UTM for the specified zone.

    East Longitudes are positive, west longitudes are negative. North latitudes
    are positive, south latitudes are negative. Lat and Long are in decimal
    degrees.

    Parameters
    ----------
    lon, lat : float, tuple, list, np.ndarray
        Longitudes and latitudes.
    zone : str, optional
        Zone number and letter (e.g. 30N). If omitted, calculate it based on the
        position.
    ellipsoid : str, optional
        Give an ellipsoid for the conversion. Defaults to WGS84.
    datum : str, optional
        Give a datum for the conversion. Defaults to WGS84.
    parallel : bool, optional
        Optionally enable the parallel processing (sometimes this is faster
        for a very large number of positions). Defaults to False.

    Returns
    -------
    eastings, northings : np.ndarray
        Eastings and northings for the given longitude, latitudes and zone.
    zone : np.ndarray
        List of UTM zones for the eastings and northings.

    """
    lon = to_list(lon)
    lat = to_list(lat)

    # Fix zone arrays/lists/tuple/strings.
    if isinstance(zone, str):
        zone = to_list(zone)
    elif isinstance(zone, np.ndarray):
        zone = zone.tolist()

    # For the spherical case, if we haven't been given zones, find them.
    if not zone:
        zone = []
        try:
            for lonlat in zip(lon, lat):
                zone_number = _get_zone_number(*lonlat)
                zone_letter = _get_zone_letter(lonlat[-1])
                if zone_number and zone_letter:
                    zone.append('{:d}{}'.format(zone_number, zone_letter))
                else:
                    raise ValueError('Invalid zone letter: are your latitudes north or south of 84 and -80 respectively?')
        except TypeError:
            zone_number = _get_zone_number(lon, lat)
            zone_letter = _get_zone_letter(lat)
            if zone_number and zone_letter:
                zone.append('{:d}{}'.format(zone_number, zone_letter))
            else:
                raise ValueError('Invalid zone letter: are your latitudes north or south of 84 and -80 respectively?')

    zone = to_list(zone)
    inverse = False

    try:
        npos = len(lon)
        if npos != len(lon):
            raise ValueError('Supplied latitudes and longitudes are not the same size.')
    except TypeError:
        npos = 1

    # If we've been given a single zone and multiple coordinates, make a
    # list of zones so we can do things easily in parallel.
    try:
        if len(zone) != npos:
            zone = zone * npos
    except TypeError:
        # Leave zone as is.
        pass

    # Do this in parallel unless we only have a single position or we've been
    # told not to.
    if npos > 1 and parallel:
        pool = multiprocessing.Pool()
        arguments = zip(lon, lat, zone,
                        [ellipsoid] * npos,
                        [datum] * npos,
                        [inverse] * npos)
        results = pool.map(__convert, arguments)
        results = np.asarray(results)
        eastings, northings = results[:, 0], results[:, 1]
        pool.close()
    elif npos > 1 and not parallel:
        eastings, northings = [], []
        for pos in zip(lon, lat, zone, [ellipsoid] * npos, [datum] * npos, [inverse] * npos):
            result = __convert(pos)
            eastings.append(result[0])
            northings.append(result[1])
    else:
        # The lon, lat and zone will all be lists here, For
        # cross-python2/python3 support, we can't just * them, so assume
        # the first value in the list is what we want.
        try:
            eastings, northings = __convert((lon[0], lat[0], zone[0], ellipsoid, datum, inverse))
        except IndexError:
            eastings, northings = __convert((lon, lat, zone, ellipsoid, datum, inverse))

        eastings = to_list(eastings)
        northings = to_list(northings)

    return np.asarray(eastings), np.asarray(northings), np.asarray(zone)


def lonlat_from_utm(eastings, northings, zone, ellipsoid='WGS84', datum='WGS84', parallel=False):
    """
    Converts UTM coordinates to lat/long.

    East Longitudes are positive, west longitudes are negative. North latitudes
    are positive, south latitudes are negative. Lat and Long are in decimal
    degrees.

    Parameters
    ----------
    eastings, northings : float, tuple, list, np.ndarray
        Eastings and northings.
    zone : str, tuple, list, np.ndarray
        Zone number and letter (e.g. 30N).
    ellipsoid : str, optional
        Give an ellipsoid for the conversion. Defaults to WGS84.
    datum : str, optional
        Give a datum for the conversion. Defaults to WGS84.
    parallel : bool, optional
        Optionally enable parallel processing (sometimes this is faster
        for a large number of positions). Defaults to False.

    Returns
    -------
    lon, lat : float, np.ndarray
        Longitude and latitudes for the given eastings and northings.

    """

    eastings = to_list(eastings)
    northings = to_list(northings)
    zone = to_list(zone)

    inverse = True

    npos = len(eastings)
    if npos != len(northings):
        raise ValueError('Supplied eastings and northings are not the same size.')

    # If we've been given a single zone and multiple coordinates, make a
    # list of zones so we can do things easily in parallel.
    if len(zone) != npos:
        zone = zone * npos

    # Do this in parallel unless we only have a small number of positions or
    # we've been told not to.
    if npos > 1 and parallel:
        pool = multiprocessing.Pool()
        arguments = zip(eastings, northings, zone,
                        [ellipsoid] * npos,
                        [datum] * npos,
                        [inverse] * npos)
        results = pool.map(__convert, arguments)
        results = np.asarray(results)
        lon, lat = results[:, 0], results[:, 1]
        pool.close()
    elif npos > 1 and not parallel:
        lon, lat = [], []
        for pos in zip(eastings, northings, zone, [ellipsoid] * npos, [datum] * npos, [inverse] * npos):
            result = __convert(pos)
            lon.append(result[0])
            lat.append(result[1])
    else:
        # The eastings, northings and zone will all be lists here, For
        # cross-python2/python3 support, we can't just * them, so assume
        # the first value in the list is what we want.
        lon, lat = __convert((eastings[0], northings[0], zone[0], ellipsoid, datum, inverse))

    return np.asarray(lon), np.asarray(lat)


def british_national_grid_to_lonlat(eastings, northings):
    """
    Converts British National Grid coordinates to latitude and longitude on the
    WGS84 spheroid.

    Shamelessly taken from:
        http://hannahfry.co.uk/2012/02/01/converting-british-national-grid-to-latitude-and-longitude-ii/

    Modified to read NumPy arrays for eastings and northings. The conversion is
    pretty crude (just a loop through all the input coordinates rather than
    a proper array-based conversion). Also flipped order of the returned arrays
    to be the same as the input (x then y).

    Parameters
    ----------
    eastings : ndarray
        Array of eastings (in metres)
    northings : ndarray
        Array of northings (in metres)

    Returns
    -------
    Lon : ndarray
        Array of converted longitudes.
    Lat : ndarray
        Array of converted latitudes.

    """

    # E, N are the British national grid coordinates - eastings and northings

    # The Airy 180 semi-major and semi-minor axes used for OSGB36 (m)
    a, b = 6377563.396, 6356256.909
    # Scale factor on the central meridian
    F0 = 0.9996012717
    # Latitude of true origin (radians)
    lat0 = np.deg2rad(49)
    # Longtitude of true origin and central meridian (radians)
    lon0 = np.deg2rad(-2)
    # Northing & easting of true origin (m)
    N0, E0 = -100000, 400000
    # eccentricity squared
    e2 = 1 - (b*b)/(a*a)
    n = (a-b)/(a+b)

    # Iterate through the pairs of values in eastings and northings.
    lonlist, latlist = [], []
    for xy in zip(eastings, northings):

        E = xy[0]
        N = xy[1]

        # Initialise the iterative variables
        lat, M = lat0, 0

        while N - N0 - M >= 0.00001:  # Accurate to 0.01mm
            lat = (N - N0 - M)/(a * F0) + lat
            M1 = (1 + n + (5./4) * n**2 + (5./4) * n**3) * (lat-lat0)
            M2 = (3*n + 3 * n**2 + (21./8)*n**3) * np.sin(lat-lat0) * \
                np.cos(lat+lat0)
            M3 = ((15./8) * n**2 + (15./8)*n**3) * np.sin(2*(lat-lat0)) * \
                np.cos(2 * (lat+lat0))
            M4 = (35./24)*n**3 * np.sin(3*(lat-lat0)) * np.cos(3*(lat+lat0))
            # meridional arc
            M = b * F0 * (M1 - M2 + M3 - M4)

        # transverse radius of curvature
        nu = a * F0 / np.sqrt(1-e2 * np.sin(lat)**2)

        # meridional radius of curvature
        rho = a * F0 * (1-e2) * (1-e2 * np.sin(lat)**2)**(-1.5)
        eta2 = nu / rho-1

        secLat = 1./np.cos(lat)
        VII = np.tan(lat) / (2 * rho * nu)
        VIII = np.tan(lat) / (24 * rho * nu**3) * (5 + 3 * np.tan(lat)**2 +
            eta2 - 9 * np.tan(lat)**2 * eta2)
        IX = np.tan(lat) / (720 * rho * nu**5) * (61 + 90 * np.tan(lat)**2 +
            45 * np.tan(lat)**4)
        X = secLat / nu
        XI = secLat / (6 * nu**3) * (nu / rho + 2 * np.tan(lat)**2)
        XII = secLat / (120 * nu**5) * (5 + 28 * np.tan(lat)**2 + 24 *
            np.tan(lat)**4)
        XIIA = secLat / (5040 * nu**7) * (61 + 662 * np.tan(lat)**2 + 1320 *
            np.tan(lat)**4 + 720 * np.tan(lat)**6)
        dE = E-E0

        # These are on the wrong ellipsoid currently: Airy1830. (Denoted by _1)
        lat_1 = lat - VII * dE**2 + VIII * dE**4 - IX * dE**6
        lon_1 = lon0 + X * dE - XI * dE**3 + XII * dE**5 - XIIA * dE**7

        # Want to convert to the GRS80 ellipsoid.
        # First convert to cartesian from spherical polar coordinates
        H = 0  # Third spherical coord.
        x_1 = (nu / F0 + H) * np.cos(lat_1) * np.cos(lon_1)
        y_1 = (nu / F0 + H) * np.cos(lat_1) * np.sin(lon_1)
        z_1 = ((1-e2) * nu / F0 + H) * np.sin(lat_1)

        # Perform Helmut transform (to go between Airy 1830 (_1) and GRS80 (_2))
        s = -20.4894 * 10**-6  # The scale factor -1
        # The translations along x,y,z axes respectively
        tx, ty, tz = 446.448, -125.157, + 542.060
        # The rotations along x,y,z respectively, in seconds
        rxs, rys, rzs = 0.1502, 0.2470, 0.8421
        # And in radians
        rx = rxs * np.pi / (180 * 3600.)
        ry = rys * np.pi / (180 * 3600.)
        rz = rzs * np.pi / (180 * 3600.)

        x_2 = tx + (1 + s) * x_1 + (-rz) * y_1 + (ry) * z_1
        y_2 = ty + (rz) * x_1 + (1 + s) * y_1 + (-rx) * z_1
        z_2 = tz + (-ry) * x_1 + (rx) * y_1 + (1 + s) * z_1

        # Back to spherical polar coordinates from cartesian
        # Need some of the characteristics of the new ellipsoid

        # The GSR80 semi-major and semi-minor axes used for WGS84(m)
        a_2, b_2 = 6378137.000, 6356752.3141
        # The eccentricity of the GRS80 ellipsoid
        e2_2 = 1 - (b_2 * b_2) / (a_2 * a_2)
        p = np.sqrt(x_2**2 + y_2**2)

        # Lat is obtained by an iterative proceedure:
        lat = np.arctan2(z_2, (p * (1-e2_2)))  # Initial value
        latold = 2 * np.pi
        while abs(lat - latold) > 10**-16:
            lat, latold = latold, lat
            nu_2 = a_2 / np.sqrt(1-e2_2 * np.sin(latold)**2)
            lat = np.arctan2(z_2 + e2_2 * nu_2 * np.sin(latold), p)

        # Lon and height are then pretty easy
        lon = np.arctan2(y_2, x_2)
        H = p / np.cos(lat) - nu_2

        # Convert to degrees
        latlist.append(np.rad2deg(lat))
        lonlist.append(np.rad2deg(lon))

    # Convert to NumPy arrays.
    lon = np.asarray(lonlist)
    lat = np.asarray(latlist)

    # Job's a good'n.
    return lon, lat


# For backwards-compatibility. At some point, these will have to be removed!
def LL_to_UTM(ReferenceEllipsoid, Lat, Long, ZoneNumber=None):
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

    ZoneNumber : int, optional
        UTM zone number in which the coordinates should be forced. This is
        useful if the spherical coordinates supplied in Lat and Long exceed
        a single UTM zone. If omitted, it is calculated automatically.

    Returns
    -------
    Zone : list
        UTM Zone for the coordinates supplied in Lat and Long.

    UTMEasting, UTMNorthing : ndarray
        Cartesian coordinates for the positions in Lat and Long.

    """
    if ReferenceEllipsoid < 20:
        raise ValueError(
            'This compatibilty function wil only work with relatively modern ellipsoids (WGS60, WGS66, WGS72 and WGS84)')

    _ellipsoid = [None] * 24
    for ee, ellipsoid in enumerate(['WGS60', 'WGS66', 'WGS72', 'WGS84'][::-1]):
        _ellipsoid[23 - ee] = ellipsoid

    Long = to_list(Long)
    Lat = to_list(Lat)

    if ZoneNumber:
        zone = re.sub('\D', '', str(ZoneNumber))
    else:
        if len(Lat) > 1:
            zone = []
            for longlat in zip(Long, Lat):
                zone.append(_get_zone_number(*longlat))
        else:
            zone = _get_zone_number(Long[0], Lat[0])

    UTMEasting, UTMNorthing, Zone = utm_from_lonlat(Long, Lat, zone, ellipsoid=_ellipsoid[ReferenceEllipsoid])

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

    """

    if ReferenceEllipsoid < 20:
        raise ValueError(
            'This compatibilty function wil only work with relatively modern ellipsoids (WGS60, WGS66, WGS72 and WGS84)')

    _ellipsoid = [None] * 24
    for ee, ellipsoid in enumerate(['WGS60', 'WGS66', 'WGS72', 'WGS84'][::-1]):
        _ellipsoid[23 - ee] = ellipsoid

    Long, Lat = lonlat_from_utm(easting, northing, zone, ellipsoid=_ellipsoid[ReferenceEllipsoid])

    return Lat, Long


def LLtoUTM(*args, **kwargs):
    warn('{} is deprecated. Use LL_to_UTM instead.'.format(inspect.stack()[0][3]))
    return LL_to_UTM(*args, **kwargs)


def UTMtoLL(*args, **kwargs):
    warn('{} is deprecated. Use UTM_to_LL instead.'.format(inspect.stack()[0][3]))
    return UTM_to_LL(*args, **kwargs)


if __name__ == '__main__':

    print('\nTest with NumPy single values')
    latTest, lonTest = 50, -5
    z, e, n, outLat, outLong = __test(latTest, lonTest)
    print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
        latTest, lonTest, outLat, outLong))
    print("Intermediate UTM: {}, {}".format(e, n))

    print('\nTest with lists')
    latTest, lonTest = [50, 55], [-5, -20]
    z, e, n, outLat, outLong = __test(latTest, lonTest)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}".format(e[c], n[c]))

    print('\nTest with NumPy arrays')
    latTest, lonTest = np.asarray([50, 55]), np.asarray([-5, -20])
    z, e, n, outLat, outLong = __test(latTest, lonTest)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}".format(e[c], n[c]))

    print('\nTest with NumPy arrays but a single zone')
    latTest, lonTest = np.asarray([50, 55]), np.asarray([-5, -20])
    z, e, n, outLat, outLong = __test(latTest, lonTest, inZone=30)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}".format(e[c], n[c]))

    # Now test the compatibilty functions too.
    print('\nTest with NumPy single values')
    latTest, lonTest = 50, -5
    z, e, n, outLat, outLong = __test_compatibility_functions(latTest, lonTest)
    print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
        latTest, lonTest, outLat, outLong))
    print("Intermediate UTM: {}, {}".format(e, n))

    print('\nTest with lists')
    latTest, lonTest = [50, 55], [-5, -20]
    z, e, n, outLat, outLong = __test_compatibility_functions(latTest, lonTest)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}".format(e[c], n[c]))

    print('\nTest with NumPy arrays')
    latTest, lonTest = np.asarray([50, 55]), np.asarray([-5, -20])
    z, e, n, outLat, outLong = __test_compatibility_functions(latTest, lonTest)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}".format(e[c], n[c]))

    print('\nTest with NumPy arrays but a single zone')
    latTest, lonTest = np.asarray([50, 55]), np.asarray([-5, -20])
    z, e, n, outLat, outLong = __test_compatibility_functions(latTest, lonTest, inZone=30)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}".format(e[c], n[c]))
