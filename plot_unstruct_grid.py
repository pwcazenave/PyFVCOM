#!/usr/bin/env python

# Plot an unstructured grid.

import matplotlib.pyplot as plt
import numpy as np
import math

def parseUnstructuredGridSMS(mesh):
    """ Reads in the SMS unstructured grid format. """

    fileRead = open(mesh, 'r')
    lines = fileRead.readlines()
    fileRead.close()

    triangles = []
    nodes = []
    x = []
    y = []
    z = []

    for line in lines:
        if line.startswith('E3T'):
            ttt = line.split()
            t1 = int(ttt[2])-1
            t2 = int(ttt[3])-1
            t3 = int(ttt[4])-1
            triangles.append([t1, t2, t3])
        elif line.startswith('ND '):
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
    """ Reads in the FVCOM unstructured grid format. """

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

def parseUnstructuredGridMIKE(mesh):
    """ Reads in the MIKE unstructured grid format. """

    fileRead = open(mesh, 'r')
    # Skip the file header (one line)
    lines = fileRead.readlines()[1:]
    fileRead.close()

    triangles = []
    nodes = []
    x = []
    y = []
    z = []

    for line in lines:
        ttt = line.strip().split()
        if len(ttt) == 4:
            t1 = int(ttt[1])-1
            t2 = int(ttt[2])-1
            t3 = int(ttt[3])-1
            triangles.append([t1, t2, t3])
        elif len(ttt) == 5:
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


def writeUnstructuredGridSMS(triangles, nodes, x, y, z, mesh):
    """ 
    Takes appropriate triangle, node and coordinate data and writes out an SMS
    formatted grid file. A lot of this is guessed from an existing file I 
    have, so it may be incorrect for all uses. 

    Input data is probably best obtained from one of:

        parseUnstructuredGridSMS()
        parseUnstructuredGridFVCOM()
        parseUnstructuredGridMIKE()

    which read in the relevant grids and output the required information for
    this function.
    """
    
    # Get some information needed for the metadata side of things
    nodeNumber = max(nodes)+1
    elementNumber = max(triangles[:,0])

    fileWrite = open(mesh, 'w')
    # Add a header
    fileWrite.write('MESH2D\n')

    # Write out the connectivity table (triangles)
    currentNode = 0
    for line in triangles:
        
        # Bump the numbers by one to correct for Python indexing from zero
        line = line + 1
        strLine = []
        # Convert the numpy array to a string array
        for value in line:
            strLine.append(str(value))

        currentNode+=1
        # Build the output string for the connectivity table
        output = ['E3T'] + [str(currentNode)] + strLine + ['1']
        output = ' '.join(output)
        #print output

        fileWrite.write(output + '\n')

    # Add the node information (nodes)
    for count, line in enumerate(nodes):

        strLine = []
        # Convert the numpy array to a string array
        strLine = str(line)

        output = ['ND'] + [strLine] + [str(x[count])] + [str(y[count])] + [str(z[count])]
        output = ' '.join(output)

        fileWrite.write(output + '\n')

    # Add all the blurb at th end of the file. This is where I'm guessing at
    # what it does...
    footer = 'NS  1 2 3 4 5 6 7 8 9 10\n\
NS  11 12 13 14 15 16 17 18 19 20\n\
NS  21 22 23 24 25 26 27 28 29 30\n\
NS  31 32 33 34 35 36 37 38 39 40\n\
NS  41 42 43 44 45 46 47 48 49 50\n\
NS  51 52 53 54 55 56 57 58 59 60\n\
NS  61 62 63 64 65 66 67 68 69 70\n\
NS  71 72 73 74 75 76 77 78 79 80\n\
NS  81 82 83 84 85 86 -87\n\
BEGPARAMDEF\n\
GM  "Mesh"\n\
SI  0\n\
DY  0\n\
TU  ""\n\
TD  0  0\n\
NUME  3\n\
BCPGC  0\n\
BEDISP  0 0 0 0 1 0 1 0 0 0 0 1\n\
BEFONT  0 2\n\
BEDISP  1 0 0 0 1 0 1 0 0 0 0 1\n\
BEFONT  1 2\n\
BEDISP  2 0 0 0 1 0 1 0 0 0 0 1\n\
BEFONT  2 2\n\
ENDPARAMDEF\n\
BEG2DMBC\n\
MAT  1 "material 01"\n\
END2DMBC\n'

    fileWrite.write(footer)

    fileWrite.close()



def plotUnstructuredGrid(triangles, nodes, x, y, z, colourLabel, addText=False, addMesh=False):
    """ 
    Takes the output of parseUnstructuredGridFVCOM() or 
    parseUnstructuredGridSMS() and readFVCOM() and plots it.

    Give triangles, nodes, x, y, z and a label for the colour scale. The first
    five arguments are the output of parseUnstructuredGridFVCOM() or 
    parseUnstructuredGridSMS(). Optionally append addText=True|False and 
    addMesh=True|False to enable/disable node numbers and grid overlays, 
    respectively.
    """

    plt.figure()
    if z.max()-z.min() != 0:
        plt.tripcolor(x, y, triangles, z, shading='interp')
        cb = plt.colorbar()
        cb.set_label(colourLabel)

    if addMesh:
        plt.triplot(x, y, triangles, '-', color=[0.6, 0.6, 0.6])

    # Add the node numbers (this is slow)
    if addText:
        for node in nodes:
            plt.text(x[node-1], y[node-1], str(nodes[node-1]),
                horizontalalignment='center', verticalalignment='top', size=8)
    plt.axes().set_aspect('equal')
    plt.axes().autoscale(tight=True)
    #plt.axis('tight')
    plt.clim(-50, 0)
    #plt.title('Triplot of user-specified triangulation')
    plt.xlabel('Metres')
    plt.ylabel('Metres')

    plt.show()
    #plt.close() # for 'looping' (slowly)

if __name__ == '__main__':
    
    from sys import argv

    # A MIKE grid
    #[triangles, nodes, x, y, z] = parseUnstructuredGridMIKE('../data/csm_culver_v7.mesh')
    # An SMS grid
    [triangles, nodes, x, y, z] = parseUnstructuredGridSMS('../data/tamar_co2V4.2dm')
    # An FVCOM grid
    #[triangles, nodes, x, y, z] = parseUnstructuredGridFVCOM('../data/co2_grd.dat')

    writeUnstructuredGridSMS(triangles, nodes, x, y, z, '../data/test.dat')

    # Let's have a look-see
    #plotUnstructuredGrid(triangles, nodes, x, y, z, 'Depth (m)')

    # Multiple grids
    #for grid in argv[1:]:
    #    [triangles, nodes, x, y, z] = parseUnstructuredGridFVCOM(grid)
    #    plotUnstructuredGrid(triangles, nodes, x, y, z, 'Depth (m)', True)

