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
    types = []
    x = []
    y = []
    z = []

    for line in lines:
        line = line.strip()
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
        elif line.startswith('NS '):
            allTypes = line.split(' ')
            for nodeID in allTypes[2:]:
                types.append(int(nodeID))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    types = np.asarray(types)
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)

    return(triangle, nodes, X, Y, Z, types)

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
    types = []
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
            types.append(int(ttt[4]))
            nodes.append(int(ttt[0]))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    types = np.asarray(types)
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)

    return(triangle, nodes, X, Y, Z, types)


def writeUnstructuredGridSMS(triangles, nodes, x, y, z, types, mesh):
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

    The footer contains meta data and additional information. See page 18 in
    http://smstutorials-11.0.aquaveo.com/SMS_Gen2DM.pdf.

    In essence, three bits are critical:
        1. The header/footer MESH2D/BEGPARAMDEF
        2. E3T prefix for the connectivity:
            (elementID, node1, node2, node3, material_type)
        3. ND prefix for the node information:
            (nodeID, x, y, z)

    The only potentially important bit is the nodestring section (prefix NS),
    which seems to be about defining boundaries. As far as I can tell, the
    footer is largely irrelevant for my purposes.

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

        # Convert the numpy array to a string array
        strLine = str(line)

        # Format output correctly
        output = ['ND'] + \
                [strLine] + \
                ['{:.8e}'.format(x[count])] + \
                ['{:.8e}'.format(y[count])] + \
                ['{:.8e}'.format(z[count])]
        output = ' '.join(output)

        fileWrite.write(output + '\n')

    # Convert MIKE boundary types to nodestrings. The format requires a prefix
    # NS, and then a maximum of 10 node IDs per line. The nodestring tail is
    # indicated by a negative node ID.

    # Iterate through the unique boundary types to get a new nodestring for
    # each boundary type.
    for boundaryType in np.unique(types[types>1]):

        # Find the nodes for the boundary type which are greater than 1 (i.e.
        # not 0 or 1).
        nodeBoundaries = nodes[types==boundaryType]

        nodestrings = 0
        oldNode = types[0]
        for counter, node in enumerate(nodeBoundaries):
            if counter+1 == len(nodeBoundaries) and node > 0:
                node = -node

            nodestrings += 1
            if nodestrings == 1:
                output = 'NS  {:d} '.format(int(node))
                fileWrite.write(output)
            elif nodestrings != 0 and nodestrings < 10:
                output = '{:d} '.format(int(node))
                fileWrite.write(output)
            elif nodestrings == 10:
                output = '{:d} '.format(int(node))
                fileWrite.write(output + '\n')
                nodestrings = 0

        # Add a new line at the end of each block. Not sure why the new line
        # above doesn't work...
        fileWrite.write('\n')



    # Add all the blurb at the end of the file.
    #
    # BEGPARAMDEF = Marks end of mesh data/beginning of mesh model definition
    # GM = Mesh name (enclosed in "")
    # SI = use SI units y/n = 1/0
    # DY = Dynamic model y/n = 1/0
    # TU = Time units
    # TD = Dynamic time data (?)
    # NUME = Number of entities available (nodes, nodestrings, elements)
    # BGPGC = Boundary group parameter group correlation y/n = 1/0
    # BEDISP/BEFONT = Format controls on display of boundary labels.
    # ENDPARAMDEF = End of the mesh model definition
    # BEG2DMBC = Beginning of the model assignments
    # MAT = Material assignment
    # END2DMBC = End of the model assignments
    footer = 'BEGPARAMDEF\n\
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
    #plt.clim(-500, 0)
    #plt.title('Triplot of user-specified triangulation')
    plt.xlabel('Metres')
    plt.ylabel('Metres')

    plt.show()
    #plt.close() # for 'looping' (slowly)

def plotUnstructuredGridProjected(triangles, nodes, x, y, z, colourLabel, addText=False, addMesh=False, extents=False):
    """
    Takes the output of parseUnstructuredGridFVCOM() or
    parseUnstructuredGridSMS() and readFVCOM() and plots it on a projected
    map. Best used for lat-long data sets.

    Give triangles, nodes, x, y, z and a label for the colour scale. The first
    five arguments are the output of parseUnstructuredGridFVCOM() or
    parseUnstructuredGridSMS(). Optionally append addText=True|False and
    addMesh=True|False to enable/disable node numbers and grid overlays,
    respectively. Finally, provide optional extents (W/E/S/N format).

    WARNING: THIS DOESN'T WORK ON FEDORA 14. REQUIRES FEDORA 16 AT LEAST
    (I THINK -- DIFFICULT TO VERIFY WITHOUT ACCESS TO A NEWER VERSION OF
    FEDORA).

    """

    from mpl_toolkits.basemap import Basemap
    from matplotlib import tri

    if extents is False:
        # We don't have a specific region defined, so use minmax of x and y.
        extents = [ min(x), max(x), min(y), max(y) ]

    # Create a triangulation object from the triagulated info read in from the
    # grid files.
    triang = tri.Triangulation(x, y, triangles)

    # Create the basemap
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(
            llcrnrlat=extents[2],
            urcrnrlat=extents[3],
            llcrnrlon=extents[0],
            urcrnrlon=extents[1],
            projection='merc',
            resolution='h',
            lat_1=extents[2],
            lat_2=extents[3],
            lat_0=(extents[3]-extents[2])/2,
            lon_0=extents[1],
            ax=ax)
    # Add the data
    #m.tripcolor(triang,z) # version of matplotlib is too old on Fedora 14
    # Add a coastline
    #m.drawlsmask(land_color='grey', lakes=True, resolution='h')
    # Can't add resolution here for some reason
    #m.drawlsmask(land_color='grey')
    m.drawcoastlines()
    plt.show()

if __name__ == '__main__':

    from sys import argv

    # A MIKE grid
    [triangles, nodes, x, y, z, types] = parseUnstructuredGridMIKE('../data/csm_culver_v7.mesh')
    #[triangles, nodes, x, y, z, types] = parseUnstructuredGridMIKE('../data/Low res.mesh')
    # An SMS grid
    #[triangles, nodes, x, y, z, types] = parseUnstructuredGridSMS('../data/tamar_co2V4.2dm')
    # An FVCOM grid
    #[triangles, nodes, x, y, z] = parseUnstructuredGridFVCOM('../data/co2_grd.dat')
    # types = [] # FVCOM doesn't record this information, I think.

    # Spit out an SMS version fo whatever's been loaded above.
    writeUnstructuredGridSMS(triangles, nodes, x, y, z, types, '../data/test.dat')

    # Let's have a look-see
    #plotUnstructuredGrid(triangles, nodes, x, y, z, 'Depth (m)', addMesh=True)
    #plotUnstructuredGridProjected(triangles, nodes, x, y, z, 'Depth (m)')

    # Multiple grids
    #for grid in argv[1:]:
    #    [triangles, nodes, x, y, z] = parseUnstructuredGridFVCOM(grid)
    #    plotUnstructuredGrid(triangles, nodes, x, y, z, 'Depth (m)', True)

