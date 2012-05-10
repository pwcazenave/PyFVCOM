#!/usr/bin/env python

""" Calculate centroids of polygons """

import time
import numpy
from sys import argv
import csv


def calculate_polygon_area(polygon, signed=False):
    """Calculate the signed area of non-self-intersecting polygon

    Input
        polygon: Numeric array of points (longitude, latitude). It is assumed
                 to be closed, i.e. first and last points are identical
        signed: Optional flag deciding whether returned area retains its sign:
                If points are ordered counter clockwise, the signed area
                will be positive.
                If points are ordered clockwise, it will be negative
                Default is False which means that the area is always positive.
    Output
        area: Area of polygon (subject to the value of argument signed)
    """

    # Make sure it is numeric
    P = numpy.array(polygon)

    # Check input
    msg = ('Polygon is assumed to consist of coordinate pairs. '
           'I got second dimension %i instead of 2' % P.shape[1])
    assert P.shape[1] == 2, msg

    msg = ('Polygon is assumed to be closed. '
           'However first and last coordinates are different: '
           '(%f, %f) and (%f, %f)' % (P[0, 0], P[0, 1], P[-1, 0], P[-1, 1]))
    assert numpy.allclose(P[0, :], P[-1, :]), msg

    # Extract x and y coordinates
    x = P[:, 0]
    y = P[:, 1]

    # Area calculation
    a = x[:-1] * y[1:]
    b = y[:-1] * x[1:]
    A = numpy.sum(a - b) / 2.

    # Return signed or unsigned area
    if signed:
        return A
    else:
        return abs(A)


def calculate_polygon_centroid(polygon):
    """Calculate the centroid of non-self-intersecting polygon

    Input
        polygon: Numeric array of points (longitude, latitude). It is assumed
                 to be closed, i.e. first and last points are identical
    Output
        Numeric (1 x 2) array of points representing the centroid
    """

    # Make sure it is numeric
    P = numpy.array(polygon)

    # Get area - needed to compute centroid
    A = calculate_polygon_area(P, signed=True)

    # Extract x and y coordinates
    x = P[:, 0]
    y = P[:, 1]

    # Exercise: Compute C as shown in http://paulbourke.net/geometry/polyarea
    a = x[:-1] * y[1:]
    b = y[:-1] * x[1:]

    cx = x[:-1] + x[1:]
    cy = y[:-1] + y[1:]

    Cx = numpy.sum(cx * (a - b)) / (6. * A)
    Cy = numpy.sum(cy * (a - b)) / (6. * A)

    # Create Nx2 array and return
    C = numpy.array([Cx, Cy])
    return C

def test_routines():
    # Test the centroid function with a couple of examples

    #----------------------------------------------
    # Test 1 - a super simple test using a "square"
    #----------------------------------------------
    P = numpy.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    C = calculate_polygon_centroid(P)

    msg = ('Calculated centroid was (%f, %f), expected '
           '(0.5, 0.5)' % tuple(C))
    assert numpy.allclose(C, [0.5, 0.5]), msg

    #---------------------------------------
    # Test 2 - Polygon not starting at origo
    #---------------------------------------
    P = numpy.array([[168, -2], [169, -2], [169, -1],
                     [168, -1], [168, -2]])
    C = calculate_polygon_centroid(P)

    msg = ('Calculated centroid was (%f, %f), expected '
           '(168.5, -1.5)' % tuple(C))
    print msg
    assert numpy.allclose(C, [168.5, -1.5]), msg

    #---------------------------
    # Test 2 - Realistic polygon
    #---------------------------
    P = numpy.array([[106.7922547, -6.2297884],
                     [106.7924589, -6.2298087],
                     [106.7924538, -6.2299127],
                     [106.7922547, -6.2298899],
                     [106.7922547, -6.2297884]])

    C = calculate_polygon_centroid(P)

    # Check against reference centroid from qgis
    reference_centroid = [106.79235602697445, -6.229849764722536]
    msg = 'Got %s but expected %s' % (str(C), str(reference_centroid))
    assert numpy.allclose(C, reference_centroid, rtol=1.0e-6), msg

def read_arc_csv(file):
    
    inFile = open(file, 'r')
    lines = inFile.readlines()
    inFile.close()

    P = []

    for line in lines:
        S = line.strip().split(',')
        S2 = [float(i) for i in S[-2:]]
        P.append(S2)

    # Add the first line to the end to close the polygon
    S3 = [float(i) for i in lines[0].strip().split(',')]
    P.append(S3[-2:])

    P = numpy.asarray(P)

    return P

if __name__ == '__main__':

    # Analyse the files specified on the command line and output all results
    # to a single file
    outFile = csv.writer(open('centroids_python.csv','w'))

    outFile.writerow(['year', 'eastings', 'northings'])
    for year in argv[1:]:
        P = read_arc_csv(year)
        C = calculate_polygon_centroid(P)
        outYear = numpy.array(float(year))
        outData = numpy.concatenate((outYear, C), axis=None)
        outFile.writerow(outData)

