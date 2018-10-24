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

import pyproj
from warnings import warn

# Convert a string, tuple, float or int to a list.
to_list = lambda x: [x] if isinstance(x, str) or isinstance(x, (float, int, np.float32)) else x


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
    projection = pyproj.Proj("+proj=utm +zone={}, +ellps={} +datum={} +units=m +no_defs".format(zone, ellipsoid, datum))
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
    Converts British National Grid coordinates to latitude and longitude on the WGS84 spheroid.

    Parameters
    ----------
    eastings : ndarray
        Array of eastings (in metres)
    northings : ndarray
        Array of northings (in metres)

    Returns
    -------
    lon : ndarray
        Array of converted longitudes (decimal degrees)
    lat : ndarray
        Array of converted latitudes (decimal degrees)

    """

    crs_british = pyproj.Proj(init='EPSG:27700')
    crs_wgs84 = pyproj.Proj(init='EPSG:4326')
    lon, lat = pyproj.transform(crs_british, crs_wgs84, eastings, northings)

    return lon, lat


def lonlat_decimal_from_degminsec(lon_degminsec, lat_degminsec):
    """
    Converts from lon lat in degrees minutes and seconds to decimal degrees

    Parameters
    ----------
    lon_degminsec : Mx3 array
        Array of longitude degrees, minutes and seconds in 3 seperate columns (for M positions). East and West is
        determined by the sign of the leading number (e.g. [-4,20,16] or [0,-3,10])
    lat_degminsec : Mx3 array
        Array of latitude degrees, minutes and seconds in 3 seperate columns (for M positions). North and South are
        determined by the sign of the leading number.

    Returns
    -------
    lon : N array
        Array of converted decimal longitudes.
    lat : N array
        Array of converted decimal latitudes.

    """
    if len(lon_degminsec.shape) != 2 or len(lat_degminsec.shape) != 2 or lon_degminsec.shape[1] != 3 or lat_degminsec.shape[1] != 3:
        raise ValueError('Input wrong shape')
    
    lon_sign = np.ones(lon_degminsec.shape)
    lon_sign[:, 1] = np.sign(lon_degminsec[:, 0])
    lon_sign[:, 2] = np.sign(lon_degminsec[:, 0])

    if np.any(lon_sign == 0):
        zero_rows = np.where(lon_sign == 0)[1]
        lon_sign[zero_rows, 1] = 1    
        lon_sign[zero_rows, 2] = np.sign(lon_degminsec[zero_rows, 1])
        if np.any(lon_sign == 0):
            zero_rows = np.where(lon_sign == 0)[1]
            lon_sign[zero_rows, 2] = 1

    lat_sign = np.ones(lat_degminsec.shape)
    lat_sign[:, 1] = np.sign(lat_degminsec[:, 0])
    lat_sign[:, 2] = np.sign(lat_degminsec[:, 0])

    if np.any(lat_sign == 0):
        zero_rows = np.where(lat_sign == 0)[1]
        lat_sign[zero_rows, 1] = 1 
        lat_sign[zero_rows, 2] = np.sign(lat_degminsec[zero_rows,1])
        if np.any(lat_sign == 0):
            zero_rows = np.where(lat_sign == 0)[1]
            lat_sign[zero_rows, 2] = 1
    
    lon_adj = lon_degminsec * lon_sign
    lat_adj = lat_degminsec * lat_sign

    lon = lon_adj[:, 0] + lon_adj[:, 1] / 60 + lon_adj[:, 2] / 6000
    lat = lat_adj[:, 0] + lat_adj[:, 1] / 60 + lat_adj[:, 2] / 6000

    return lon, lat


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
