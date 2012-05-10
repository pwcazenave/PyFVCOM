#!/usr/bin/env python
""" Generate an unstructred grid from a model boundary """

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import math


def generateUnstructuredGrid(x, y, minAngle):
    """ 

    Create an unstructured grid from a boundary series of points. The 
    boundary points must be in x y format (space or tab delimited). 
    Specify an angle which limits the internal angles in the generated
    elements.

    """

    triang = tri.Triangulation(x, y)

    return triang

    
def generateCircleBoundary(angleStep, xOrigin, yOrigin, radiusCircle):
    """ Generate a simple circlular boundary. """

    x = [xOrigin + radiusCircle/40]
    y = [yOrigin - radiusCircle/4]
    x.append(xOrigin + radiusCircle/4)
    y.append(yOrigin - radiusCircle/5)

    steps = 360 / angleStep
    sampleStep = 2 * math.pi / steps

    for a in range(0, steps):
        x.append(math.sin(a * sampleStep) * radiusCircle + xOrigin)
        y.append(math.cos(a * sampleStep) * radiusCircle + yOrigin)

    # Make numpy arrays of x and y
    X = np.asarray(x)
    Y = np.asarray(y)

    return (X, Y)

def readBoundary(filename):
    """ Read an xy file of boundary positions. """

    fileRead = open(boundary, 'r')
    lines = fileRead.readlines()
    fileRead.close()

    for line in lines:
        [x, y] = line.strip().split()

    # Make numpy arrays of x and y
    X = np.asarray(x)
    Y = np.asarray(y)

    return (X, Y)

[x, y] = generateCircleBoundary(10, 10, 400, 50)
triang = generateUnstructuredGrid(x, y, 10)

plt.figure()
plt.plot(x, y, '.-')
plt.triplot(triang,)
plt.gca().set_aspect('equal')
plt.show()
