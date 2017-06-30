"""
Utilities to convert from cartesian UTM coordinates to spherical latitude and
longitudes.

This function uses functions lifted from the utm library. For those that are
interested, the reason we don't just use the utm utility is because it
refuses to do coordinate conversions outside a single zone.

"""

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
    e, n, z = utm_from_latlon(inLong, inLat, inZone)
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
    from the utm.latlon_to_zone_number function.

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


def utm_from_latlon(lon, lat, zone=None, ellipsoid='WGS84', datum='WGS84'):
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

    Returns
    -------
    eastings, northings : np.ndarray
        Eastings and northings for the given longitude, latitudes and zone.

    """
    lon = to_list(lon)
    lat = to_list(lat)

    # For the spherical case, if we haven't been given zones, find them.
    if not zone:
        zone = []
        for lonlat in zip(lon, lat):
            zone_number = _get_zone_number(*lonlat)
            zone_letter = _get_zone_letter(lonlat[-1])
            if zone_number and zone_letter:
                zone.append('{:d}{}'.format(zone_number, zone_letter))
            else:
                raise ValueError('Invalid zone letter: are your latitudes north or south of 84 and -80 respectively?')

    zone = to_list(zone)
    inverse = False

    npos = len(lon)
    if npos != len(lon):
        raise ValueError('Supplied latitudes and longitudes are not the same size.')

    # If we've been given a single zone and multiple coordinates, make a
    # list of zones so we can do things easily in parallel.
    if len(zone) != npos:
        zone = zone * npos

    # Do this in parallel unless we only have a single position.
    if npos > 1:
        pool = multiprocessing.Pool()
        arguments = zip(lon, lat, zone,
                        [ellipsoid] * npos,
                        [datum] * npos,
                        [inverse] * npos)
        results = pool.map(__convert, arguments)
        results = np.asarray(results)
        lon, lat = results[:, 0], results[:, 1]
        pool.close()
    else:
        # The lon, lat and zone will all be lists here, For
        # cross-python2/python3 support, we can't just * them, so assume
        # the first value in the list is what we want.
        lon, lat = __convert((lon[0], lat[0], zone[0], ellipsoid, datum, inverse))
        lon = to_list(lon)
        lat = to_list(lat)

    return np.asarray(lon), np.asarray(lat), np.asarray(zone)


def lonlat_from_utm(eastings, northings, zone, ellipsoid='WGS84', datum='WGS84'):
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

    # Do this in parallel unless we only have a single position.
    if npos > 1:
        pool = multiprocessing.Pool()
        arguments = zip(eastings, northings, zone,
                        [ellipsoid] * npos,
                        [datum] * npos,
                        [inverse] * npos)
        results = pool.map(__convert, arguments)
        results = np.asarray(results)
        lon, lat = results[:, 0], results[:, 1]
        pool.close()
    else:
        # The eastings, northings and zone will all be lists here, For
        # cross-python2/python3 support, we can't just * them, so assume
        # the first value in the list is what we want.
        lon, lat = __convert((eastings[0], northings[0], zone[0], ellipsoid, datum, inverse))

    return np.asarray(lon), np.asarray(lat)


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

    UTMEasting, UTMNorthing, Zone = utm_from_latlon(Long, Lat, zone, ellipsoid=_ellipsoid[ReferenceEllipsoid])

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
