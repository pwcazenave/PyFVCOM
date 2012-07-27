#!/usr/bin/env python

import math

""" 
    Two functions to convert from UTM to Latitude and longitude and back again. 

    utm2latlong lifted from http://stackoverflow.com/questions/343865.
    latlong2utm follows method outlined by Steve Dutch at his homepage:
        http://www.uwgb.edu/dutchs/UsefulData/UTMFormulas.HTM

    latlong2utm assumes the WGS84 spheroid and should be good to within a metre.

    Pierre Cazenave 2012/03/29

"""

def utm2latlong(zone, easting, northing, northernHemisphere=True):
    if not northernHemisphere:
        northing = 10000000 - northing

    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996

    arc = northing / k0
    mu = arc / (a * (1 - math.pow(e, 2) / 4.0 - 3 * math.pow(e, 4) / 64.0 - 5 * math.pow(e, 6) / 256.0))

    ei = (1 - math.pow((1 - e * e), (1 / 2.0))) / (1 + math.pow((1 - e * e), (1 / 2.0)))

    ca = 3 * ei / 2 - 27 * math.pow(ei, 3) / 32.0

    cb = 21 * math.pow(ei, 2) / 16 - 55 * math.pow(ei, 4) / 32
    cc = 151 * math.pow(ei, 3) / 96
    cd = 1097 * math.pow(ei, 4) / 512
    phi1 = mu + ca * math.sin(2 * mu) + cb * math.sin(4 * mu) + cc * math.sin(6 * mu) + cd * math.sin(8 * mu)

    n0 = a / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (1 / 2.0))

    r0 = a * (1 - e * e) / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * math.tan(phi1) / r0

    _a1 = 500000 - easting
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2

    t0 = math.pow(math.tan(phi1), 2)
    Q0 = e1sq * math.pow(math.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * math.pow(dd0, 4) / 24

    fact4 = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * math.pow(dd0, 6) / 720

    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * math.pow(dd0, 3) / 6.0
    lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * math.pow(Q0, 2) + 8 * e1sq + 24 * math.pow(t0, 2)) * math.pow(dd0, 5) / 120
    _a2 = (lof1 - lof2 + lof3) / math.cos(phi1)
    _a3 = _a2 * 180 / math.pi

    latitude = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / math.pi

    if not northernHemisphere:
        latitude = -latitude

    longitude = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3

    return (latitude, longitude)

def latlong2utm(latitude, longitude):

    # Convert input to radians
    lat = math.radians(latitude)
    lon = math.radians(longitude)

    # Some constants
    a = 6378137
    b = 6356752.3142
    e = 0.081819191
    k0 = 0.9996

    # Some constants derived from those above.
    e = math.sqrt(1 - math.pow(b, 2.0) / math.pow(a, 2.0))
    e_prime2 = math.pow((e * a / b), 2.0)
    e_prime4 = math.pow((e * a / b), 4.0)
    n = (a - b) / (a + b)
    rho = a * (1 - math.pow(e, 2.0)) / math.pow((1 - math.pow(e, 2.0) * math.pow(math.sin(lat), 2.0)),(3.0 / 2))
    nu = a / math.pow((1-(e * math.pow(math.sin(lat),2))),(1/2))

    # Get UTM zone central meridian
    if longitude == 0:
    	long0 = 3.0
    elif longitude < 0:
    	long0 = int((180+longitude)/6)+1
    else:
    	long0 = int(longitude/6)+31

    p = lon - math.radians(long0)

    # Meridional arc length (S)
    A0 = a * (1 - n + (5 * n * n / 4) * (1 - n) + (81 * math.pow(n,4)/64) * (1 - n))
    B0 = (3 * a * n / 2) * (1 - n - (7 * n * n / 8) * (1 - n) + 55 * math.pow(n, 4) / 64)
    C0 = (15 * a * n * n / 16) * (1 - n +(3 * n * n / 4) * (1 - n))
    D0 = (35 * a * math.pow(n, 3) / 48) * (1 - n + 11 * n * n / 16)
    E0 = (315 * a * math.pow(n, 4) / 51) * (1 - n)
    S = A0 * lat - B0 * math.sin(2 * lat) + C0 * math.sin(4 * lat) - D0 * math.sin(6 * lat) + E0 * math.sin(8 * lat)

    K1 = S * k0
    K2 = k0 * nu * math.sin(2 * lat) / 4
    K3 = (k0 * nu * math.sin(lat) * math.pow(math.cos(lat), 3.0) / 24) * ((5 - math.pow(math.tan(lat), 2.0) + (9 * e_prime2 * math.pow(math.cos(lat), 2.0)) + (4 * e_prime4 * math.pow(math.cos(lat), 4.0))))
    northing = K1 + (K2*math.pow(p, 2.0)) + (K3*math.pow(p, 4.0))
    # Adjust northing value depending on hemisphere
    if latitude<0:
    	northing = northing + 10000000

    K4 = k0 * nu * math.cos(lat)
    K5 = (k0 * nu * math.pow(math.cos(lat), 3.0) / 6) * (1 - (math.pow(math.tan(lat), 2.0)) + e_prime2 * (math.pow(math.cos(lat), 2.0)))
    # Add false easting (500000)
    easting = (K4 * p) + (K5 * math.pow(p, 3.0)) + 500000

#    print S, K1, K2, K3, K4, K5
    print 'A0: ' + str(A0) + '\nB0: ' + str(B0) + '\nC0: ' + str(C0) + '\nD0: ' + str(D0) + '\nE0: ' + str(E0) + '\nS: ' + str(S)

    return (easting, northing)

[easting, northing] = latlong2utm(51,0)
print easting, northing
[lat, long] = utm2latlong(31, easting, northing)
print lat, long
