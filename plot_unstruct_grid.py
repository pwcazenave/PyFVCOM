#!/usr/bin/env python

# Plot an unstructured grid.

import matplotlib.pyplot as plt
import numpy as np
import math

def parseUnstructuredGridSMS(mesh):
    """ Reads in the output of SMS Mesh generation. """

    fileRead = open(mesh, 'r')
    lines = fileRead.readlines()
    fileRead.close()

    triangles = []
    nodes = []
    x = []
    y = []
    z = []

    for line in lines:
        if 'E3T' in line:
            ttt = line.split()
            t1 = int(ttt[2])-1
            t2 = int(ttt[3])-1
            t3 = int(ttt[4])-1
            triangles.append([t1, t2, t3])
        elif 'ND ' in line:
            xy = line.split()
            x.append(float(xy[2]))
            y.append(float(xy[3]))
            z.append(float(xy[4]))
            nodes.append(int(xy[1]))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)

    return(triangle, nodes, X, Y, Z)

def parseUnstructuredGridFVCOM(mesh):
    """ Reads in the unstructured grid input for FVCOM. """

    fileRead = open(mesh, 'r')
    # Skip the file header (two lines)
    lines = fileRead.readlines()[2:]
    fileRead.close()

    triangles = []
    nodes = []
    x = []
    y = []
    z = []

    for line in lines:
        ttt = line.strip().split()
        if len(ttt) == 5:
            t1 = int(ttt[1])-1
            t2 = int(ttt[2])-1
            t3 = int(ttt[3])-1
            triangles.append([t1, t2, t3])
        elif len(ttt) == 4:
            x.append(float(ttt[1]))
            y.append(float(ttt[2]))
            z.append(float(ttt[3]))
            nodes.append(int(ttt[0]))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)

    return(triangle, nodes, X, Y, Z)

def plotUnstructuredGrid(triangles, nodes, x, y, z, colourLabel, addText=False):
    """ 
    Takes the output of parseUnstructuredGridFVCOM() or 
    parseUnstructuredGridSMS() and readFVCOM() and plots it.

    Give triangles, nodes, x, y, z and a label for the colour scale. The first
    five arguments are the output of parseUnstructuredGridFVCOM() or 
    parseUnstructuredGridSMS().
    """

    plt.figure()
    if z.max()-z.min() != 0:
        plt.tripcolor(x, y, triangles, z, shading='interp')
        cb = plt.colorbar()
        cb.set_label(colourLabel)

    #plt.triplot(x, y, triangles, '-', color=[0.6, 0.6, 0.6])
    # Add the node numbers (this is slow)
    if addText:
        for node in nodes:
            plt.text(x[node-1], y[node-1], str(nodes[node-1]),
                horizontalalignment='center', verticalalignment='center', size=8)
    plt.axes().set_aspect('equal', 'datalim')
    plt.clim(-1.5, 1.5)
    #plt.title('Triplot of user-specified triangulation')
    #plt.xlabel('x')
    #plt.ylabel('y')

    plt.show()
    #plt.close() # for 'looping' (slowly)

if __name__ == '__main__':
    
    from sys import argv

    # An SMS grid
    #[triangles, nodes, x, y, z] = parseUnstructuredGridSMS('tamar_co2V4.2dm')
    # The model input grid
    for grid in argv[1:]:
        [triangles, nodes, x, y, z] = parseUnstructuredGridFVCOM(grid)

        # Let's have a look-see
        plotUnstructuredGrid(triangles, nodes, x, y, z, 'Depth (m)')

