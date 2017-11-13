"""
Tools for manipulating and converting unstructured grids in a range of formats.

"""

from __future__ import print_function, division

import sys
import inspect
import multiprocessing
import numpy as np
import math
from matplotlib.tri.triangulation import Triangulation
from matplotlib.tri import CubicTriInterpolator
import matplotlib.path as mplPath
from warnings import warn
from functools import partial

from PyFVCOM.coordinate import UTM_to_LL


def read_sms_mesh(mesh, nodestrings=False):
    """
    Reads in the SMS unstructured grid format. Also creates IDs for output to
    MIKE unstructured grid format.

    Parameters
    ----------
    mesh : str
        Full path to an SMS unstructured grid (.2dm) file.
    nodestrings : bool, optional
        Set to True to return the IDs of the node strings as a dictionary.

    Returns
    -------
    triangle : ndarray
        Integer array of shape (nele, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in X and Y (see below). Values
        are python-indexed.
    nodes : ndarray
        Integer number assigned to each node.
    X, Y, Z : ndarray
        Coordinates of each grid node and any associated Z value.
    types : ndarray
        Classification for each node string based on the number of node
        strings + 2. This is mainly for use if converting from SMS .2dm
        grid format to DHI MIKE21 .mesh format since the latter requires
        unique IDs for each boundary (with 0 and 1 reserved for land and
        sea nodes).
    nodestrings : list, optional (see nodestrings above)
        Optional list of lists containing the node IDs (python-indexed) of the
        node strings in the SMS grid.

    """

    fileRead = open(mesh, 'r')
    lines = fileRead.readlines()
    fileRead.close()

    triangles = []
    nodes = []
    types = []
    nodeStrings = []
    nstring = []
    x = []
    y = []
    z = []

    # MIKE unstructured grids allocate their boundaries with a type ID flag.
    # Although this function is not necessarily always the precursor to writing
    # a MIKE unstructured grid, we can create IDs based on the number of node
    # strings in the SMS grid. MIKE starts counting open boundaries from 2 (1
    # and 0 are land and sea nodes, respectively).
    typeCount = 2

    for line in lines:
        line = line.strip()
        if line.startswith('E3T'):
            ttt = line.split()
            t1 = int(ttt[2]) - 1
            t2 = int(ttt[3]) - 1
            t3 = int(ttt[4]) - 1
            triangles.append([t1, t2, t3])
        elif line.startswith('ND '):
            xy = line.split()
            x.append(float(xy[2]))
            y.append(float(xy[3]))
            z.append(float(xy[4]))
            nodes.append(int(xy[1]))
            # Although MIKE keeps zero and one reserved for normal nodes and
            # land nodes, SMS doesn't. This means it's not straightforward
            # to determine this information from the SMS file alone. It would
            # require finding nodes which are edge nodes and assigning their
            # ID to one. All other nodes would be zero until they were
            # overwritten when examining the node strings below.
            types.append(0)
        elif line.startswith('NS '):
            allTypes = line.split(' ')

            for nodeID in allTypes[2:]:
                types[np.abs(int(nodeID)) - 1] = typeCount
                if int(nodeID) > 0:
                    nstring.append(int(nodeID) - 1)
                else:
                    nstring.append(np.abs(int(nodeID)) - 1)
                    nodeStrings.append(nstring)
                    nstring = []

                # Count the number of node strings, and output that to types.
                # Nodes in the node strings are stored in nodeStrings.
                if int(nodeID) < 0:
                    typeCount += 1

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    types = np.asarray(types)
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)
    if nodestrings:
        return triangle, nodes, X, Y, Z, types, nodeStrings
    else:
        return triangle, nodes, X, Y, Z, types


def read_fvcom_mesh(mesh):
    """
    Reads in the FVCOM unstructured grid format.

    Parameters
    ----------
    mesh : str
        Full path to the FVCOM unstructured grid file (.dat usually).

    Returns
    -------
    triangle : ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in X and Y (see below).
    nodes : ndarray
        Integer number assigned to each node.
    X, Y, Z : ndarray
        Coordinates of each grid node and any associated Z value.

    """

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
            t1 = int(ttt[1]) - 1
            t2 = int(ttt[2]) - 1
            t3 = int(ttt[3]) - 1
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

    return triangle, nodes, X, Y, Z


def read_mike_mesh(mesh, flipZ=True):
    """
    Reads in the MIKE unstructured grid format.

    Depth sign is typically reversed (i.e. z*-1) but can be disabled by
    passing flipZ=False.

    Parameters
    ----------
    mesh : str
        Full path to the DHI MIKE21 unstructured grid file (.mesh usually).
    flipZ : bool, optional
        DHI MIKE21 unstructured grids store the z value as positive down
        whereas FVCOM wants negative down. The conversion is
        automatically applied unless flipZ is set to False.

    Returns
    -------
    triangle : ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in X and Y (see below).
    nodes : ndarray
        Integer number assigned to each node.
    X, Y, Z : ndarray
        Coordinates of each grid node and any associated Z value.
    types : ndarray
        Classification for each open boundary. DHI MIKE21 .mesh format
        requires unique IDs for each open boundary (with 0 and 1
        reserved for land and sea nodes).

    """

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
            t1 = int(ttt[1]) - 1
            t2 = int(ttt[2]) - 1
            t3 = int(ttt[3]) - 1
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
    # N.B. Depths should be negative for FVCOM
    if flipZ:
        Z = -np.asarray(z)
    else:
        Z = np.asarray(z)

    return triangle, nodes, X, Y, Z, types


def read_gmsh_mesh(mesh):
    """
    Reads in the GMSH unstructured grid format (version 2.2).

    Parameters
    ----------
    mesh : str
        Full path to the DHI MIKE21 unstructured grid file (.mesh usually).

    Returns
    -------
    triangle : ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of three
        points and this contains the three node numbers (stored in nodes) which
        refer to the coordinates in X and Y (see below).
    nodes : ndarray
        Integer number assigned to each node.
    X, Y, Z : ndarray
        Coordinates of each grid node and any associated Z value.

    """

    fileRead = open(mesh, 'r')
    lines = fileRead.readlines()
    fileRead.close()

    _header = False
    _nodes = False
    _elements = False

    # Counters for the nodes and elements.
    n = 0
    e = 0

    for line in lines:
        line = line.strip()

        # If we've been told we've got to the header, read in the mesh version
        # here.
        if _header:
            _header = False
            continue

        # Grab the number of nodes.
        if _nodes:
            nn = int(line.strip())
            x, y, z, nodes = np.zeros((nn,)) - 1, \
                             np.zeros((nn,)) - 1, \
                             np.zeros((nn,)) - 1, \
                             np.zeros((nn,)).astype(int) - 1
            _nodes = False
            continue

        # Grab the number of elements.
        if _elements:
            ne = int(line.strip())
            triangles = np.zeros((ne, 3)).astype(int) - 1
            _elements = False
            continue

        if line == r'$MeshFormat':
            # Header information on the next line
            _header = True
            continue

        elif line == r'$EndMeshFormat':
            continue

        elif line == r'$Nodes':
            _nodes = True
            continue

        elif line == r'$EndNodes':
            continue

        elif line == r'$Elements':
            _elements = True
            continue

        elif line == r'$EndElements':
            continue

        else:
            # Some non-info line, so either nodes or elements. Discern that
            # based on the number of fields.
            s = line.split(' ')
            if len(s) == 4:
                # Nodes
                nodes[n] = int(s[0])
                x[n] = float(s[1])
                y[n] = float(s[2])
                z[n] = float(s[3])
                n += 1

            # Only keep the triangulation for the 2D mesh (ditch the 1D stuff).
            elif len(s) > 4 and int(s[1]) == 2:
                # Offset indices by one for Python indexing.
                triangles[e, :] = [int(i) - 1 for i in s[-3:]]
                e += 1

            else:
                continue

    # Tidy up the triangles array to remove the empty rows due to the number
    # of elements specified in the mesh file including the 1D triangulation.
    # triangles = triangles[triangles[:, 0] != -1, :]
    triangles = triangles[:e, :]

    return triangles, nodes, x, y, z


def read_fvcom_obc(obc):
    """
    Read in an FVCOM open boundary file.

    Parameters
    ----------
    obc : str
        Path to the casename_obc.dat file from FVCOM.

    Returns
    -------
    nodes : ndarray
        Node IDs (zero-indexed) for the open boundary.
    types : ndarray
        Open boundary node types (see the FVCOM manual for more information on
        what these values mean).
    count : ndarray
        Open boundary node number.


    """

    obcs = np.genfromtxt(obc, skip_header=1).astype(int)
    count = obcs[:, 0]
    nodes = obcs[:, 1] - 1
    types = obcs[:, 2]

    return nodes, types, count


def write_sms_mesh(triangles, nodes, x, y, z, types, mesh):
    """
    Takes appropriate triangle, node, boundary type and coordinate data and
    writes out an SMS formatted grid file (mesh). The footer is largely static,
    but the elements, nodes and node strings are parsed from the input data.

    Input data is probably best obtained from one of:

        grid.read_sms_mesh()
        grid.read_fvcom_mesh()
        grid.read_mike_mesh()

    which read in the relevant grids and output the required information for
    this function.

    The footer contains meta data and additional information. See page 18 in
    http://smstutorials-11.0.aquaveo.com/SMS_Gen2DM.pdf.

    In essence, four bits are critical:
        1. The header/footer MESH2D/BEGPARAMDEF
        2. E3T prefix for the connectivity:
            (elementID, node1, node2, node3, material_type)
        3. ND prefix for the node information:
            (nodeID, x, y, z)
        4. NS prefix for the node strings which indicate the open boundaries.

    As far as I can tell, the footer is largely irrelevant for FVCOM purposes.

    Parameters
    ----------
    triangles : ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in X and Y (see below).
    nodes : ndarray
        Integer number assigned to each node.
    x, y, z : ndarray
        Coordinates of each grid node and any associated Z value.
    types : ndarray
        Classification for each open boundary. DHI MIKE21 .mesh format
        requires unique IDs for each open boundary (with 0 and 1
        reserved for land and sea nodes). Similar values can be used in
        SMS grid files too.
    mesh : str
        Full path to the output file name.

    """

    fileWrite = open(mesh, 'w')
    # Add a header
    fileWrite.write('MESH2D\n')

    # Write out the connectivity table (triangles)
    currentNode = 0
    for line in triangles:

        # Bump the numbers by one to correct for Python indexing from zero
        line += 1
        strLine = []
        # Convert the numpy array to a string array
        for value in line:
            strLine.append(str(value))

        currentNode += 1
        # Build the output string for the connectivity table
        output = ['E3T'] + [str(currentNode)] + strLine + ['1']
        output = ' '.join(output)

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

    # Convert MIKE boundary types to node strings. The format requires a prefix
    # NS, and then a maximum of 10 node IDs per line. The node string tail is
    # indicated by a negative node ID.

    # Iterate through the unique boundary types to get a new node string for
    # each boundary type (ignore types of less than 2 which are not open
    # boundaries in MIKE).
    for boundaryType in np.unique(types[types > 1]):

        # Find the nodes for the boundary type which are greater than 1 (i.e.
        # not 0 or 1).
        nodeBoundaries = nodes[types == boundaryType]

        nodeStrings = 0
        for counter, node in enumerate(nodeBoundaries):
            if counter + 1 == len(nodeBoundaries) and node > 0:
                node = -node

            nodeStrings += 1
            if nodeStrings == 1:
                output = 'NS  {:d} '.format(int(node))
                fileWrite.write(output)
            elif nodeStrings != 0 and nodeStrings < 10:
                output = '{:d} '.format(int(node))
                fileWrite.write(output)
            elif nodeStrings == 10:
                output = '{:d} '.format(int(node))
                fileWrite.write(output + '\n')
                nodeStrings = 0

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
    # NUME = Number of entities available (nodes, node strings, elements)
    # BGPGC = Boundary group parameter group correlation y/n = 1/0
    # BEDISP/BEFONT = Format controls on display of boundary labels.
    # ENDPARAMDEF = End of the mesh model definition
    # BEG2DMBC = Beginning of the model assignments
    # MAT = Material assignment
    # END2DMBC = End of the model assignments
    footer = 'BEGPARAMDEF\nGM  "Mesh"\nSI  0\nDY  0\nTU  ""\nTD  0  0\nNUME  3\nBCPGC  0\nBEDISP  0 0 0 0 1 0 1 0 0 0 0 1\nBEFONT  0 2\nBEDISP  1 0 0 0 1 0 1 0 0 0 0 1\nBEFONT  1 2\nBEDISP  2 0 0 0 1 0 1 0 0 0 0 1\nBEFONT  2 2\nENDPARAMDEF\nBEG2DMBC\nMAT  1 "material 01"\nEND2DMBC\n'

    fileWrite.write(footer)

    fileWrite.close()


def write_sms_bathy(triangles, nodes, z, PTS):
    """
    Writes out the additional bathymetry file sometimes output by SMS. Not sure
    why this is necessary as it's possible to put the depths in the other file,
    but hey ho, it is obviously sometimes necessary.

    Parameters
    ----------
    triangles : ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in X and Y (see below).
    nodes : ndarray
        Integer number assigned to each node.
    z : ndarray
        Z values at each node location.
    PTS : str
        Full path of the output file name.

    """

    filePTS = open(PTS, 'w')

    # Get some information needed for the metadata side of things
    nodeNumber = len(nodes)
    elementNumber = len(triangles[:, 0])

    # Header format (see:
    #     http://wikis.aquaveo.com/xms/index.php?title=GMS:Data_Set_Files)
    # DATASET = indicates data
    # OBJTYPE = type of object (i.e. mesh 3d, mesh 2d) data is associated with
    # BEGSCL = Start of the scalar data set
    # ND = Number of data values
    # NC = Number of elements
    # NAME = Freeform data set name
    # TS = Time step of the data
    header = 'DATASET\nOBJTYEP = "mesh2d"\nBEGSCL\nND  {:<6d}\nNC  {:<6d}\nNAME "Z_interp"\nTS 0 0\n'.format(int(nodeNumber), int(elementNumber))
    filePTS.write(header)

    # Now just iterate through all the z values. This process assumes the z
    # values are in the same order as the nodes. If they're not, this will
    # make a mess of your data.
    for depth in z:
        filePTS.write('{:.5f}\n'.format(float(depth)))

    # Close the file with the footer
    filePTS.write('ENDDS\n')
    filePTS.close()


def write_mike_mesh(triangles, nodes, x, y, z, types, mesh):
    """
    Write out a DHI MIKE unstructured grid (mesh) format file. This
    assumes the input coordinates are in longitude and latitude. If they
    are not, the header will need to be modified with the appropriate
    string (which is complicated and of which I don't have a full list).

    If types is empty, then zeros will be written out for all nodes.

    Parameters
    ----------
    triangles : ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in X and Y (see below).
    nodes : ndarray
        Integer number assigned to each node.
    x, y, z : ndarray
        Coordinates of each grid node and any associated Z value.
    types : ndarray
        Classification for each open boundary. DHI MIKE21 .mesh format
        requires unique IDs for each open boundary (with 0 and 1
        reserved for land and sea nodes).
    mesh : str
        Full path to the output mesh file.

    """
    fileWrite = open(mesh, 'w')
    # Add a header
    output = '{}  LONG/LAT'.format(int(len(nodes)))
    fileWrite.write(output + '\n')

    if len(types) == 0:
        types = np.zeros(shape=(len(nodes), 1))

    # Write out the node information
    for count, line in enumerate(nodes):

        # Convert the numpy array to a string array
        strLine = str(line)

        output = \
            [strLine] + \
            ['{}'.format(x[count])] + \
            ['{}'.format(y[count])] + \
            ['{}'.format(z[count])] + \
            ['{}'.format(int(types[count]))]
        output = ' '.join(output)

        fileWrite.write(output + '\n')

    # Now for the connectivity

    # Little header. No idea what the 3 and 21 are all about (version perhaps?)
    output = '{} {} {}'.format(int(len(triangles)), '3', '21')
    fileWrite.write(output + '\n')

    for count, line in enumerate(triangles):

        # Bump the numbers by one to correct for Python indexing from zero
        line = line + 1
        strLine = []
        # Convert the numpy array to a string array
        for value in line:
            strLine.append(str(value))

        # Build the output string for the connectivity table
        output = [str(count + 1)] + strLine
        output = ' '.join(output)

        fileWrite.write(output + '\n')

    fileWrite.close()


def write_fvcom_mesh(triangles, nodes, x, y, z, mesh, extra_depth=None):
    """
    Write out an FVCOM unstructured grid (mesh) format file.

    Parameters
    ----------
    triangles : ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in X and Y (see below). Give as
        a zero-indexed array.
    nodes : ndarray
        Integer number assigned to each node.
    x, y, z : ndarray
        Coordinates of each grid node and any associated Z value.
    mesh : str
        Full path to the output mesh file.
    extra_depth : str, optional
        If given, output depths to a separate FVCOM compatible file.

    """

    with open(mesh, 'w') as f:
        f.write('Node Number = {:d}\n'.format(len(x)))
        f.write('Cell Number = {:d}\n'.format(np.shape(triangles)[0]))
        for i, triangle in enumerate(triangles, 1):
            f.write('{node:d} {:d} {:d} {:d} {node:d}\n'.format(node=i, *triangle + 1))
        for node in zip(nodes, x, y, z):
            f.write('{:d} {:.6f} {:.6f} {:.6f}\n'.format(*node))

    if extra_depth:
        with open(extra_depth, 'w') as f:
            f.write('Node Number = {:d}\n'.format(len(x)))
            for node in zip(x, y, z):
                f.write('{:.6f} {:.6f} {:.6f}\n'.format(*node))


def find_nearest_point(FX, FY, x, y, maxDistance=np.inf, noisy=False):
    """
    Given some point(s) x and y, find the nearest grid node in FX and
    FY.

    Returns the nearest coordinate(s), distance(s) from the point(s) and
    the index in the respective array(s).

    Optionally specify a maximum distance (in the same units as the
    input) to only return grid positions which are within that distance.
    This means if your point lies outside the grid, for example, you can
    use maxDistance to filter it out. Positions and indices which cannot
    be found within maxDistance are returned as NaN; distance is always
    returned, even if the maxDistance threshold has been exceeded.

    Parameters
    ----------
    FX, FY : ndarray
        Coordinates within which to search for the nearest point given
        in x and y.
    x, y : ndarray
        List of coordinates to find the closest value in FX and FY.
        Upper threshold of distance is given by maxDistance (see below).
    maxDistance : float, optional
        Unless given, there is no upper limit on the distance away from
        the source for which a result is deemed valid. Any other value
        specified here limits the upper threshold.
    noisy : bool or int, optional
        Set to True to enable verbose output. If int, outputs every nth
        iteration.

    Returns
    -------
    nearestX, nearestY : ndarray
        Coordinates from FX and FY which are within maxDistance (if
        given) and closest to the corresponding point in x and y.
    distance : ndarray
        Distance between each point in x and y and the closest value in
        FX and FY. Even if maxDistance is given (and exceeded), the
        distance is reported here.
    index : ndarray
        List of indices of FX and FY for the closest positions to those
        given in x, y.

    """

    if np.ndim(x) != np.ndim(y):
        raise Exception('Number of points in X and Y do not match')

    nearestX = np.empty(np.shape(x))
    nearestY = np.empty(np.shape(x))
    index = np.empty(np.shape(x))
    distance = np.empty(np.shape(x))

    # Make all values NaN
    nearestX = nearestX.ravel() * np.NaN
    nearestY = nearestY.ravel() * np.NaN
    index = index.ravel() * np.NaN
    distance = distance.ravel() * np.NaN

    if np.ndim(x) == 0:
        todo = np.column_stack([x, y])
        n = 1
    else:
        todo = list(zip(x, y))
        n = np.shape(x)[0]

    for c, pointXY in enumerate(todo):
        if type(noisy) == int:
            if c == 0 or (c + 1) % noisy == 0:
                print('Point {} of {}'.format(c + 1, n))
        elif noisy:
            print('Point {} of {}'.format(c + 1, n))

        findX, findY = FX - pointXY[0], FY - pointXY[1]
        vectorDistances = np.sqrt(findX**2 + findY**2)
        if vectorDistances.min() > maxDistance:
            distance[c] = np.min(vectorDistances)
            # Should be NaN already, but no harm in being thorough
            index[c], nearestX[c], nearestY[c] = np.NaN, np.NaN, np.NaN
        else:
            distance[c] = vectorDistances.min()
            index[c] = vectorDistances.argmin()
            nearestX[c] = FX[int(index[c])]
            nearestY[c] = FY[int(index[c])]

    # Convert the indices to ints if we don't have any NaNs.
    if not np.any(np.isnan(index)):
        index = index.astype(int)

    return nearestX, nearestY, distance, index


def element_side_lengths(triangles, x, y):
    """
    Given a list of triangle nodes, calculate the length of each side of each
    triangle and return as an array of lengths. Units are in the original input
    units (no conversion from lat/long to metres, for example).

    The arrays triangles, x and y can be created by running read_sms_mesh(),
    read_fvcom_mesh() or read_mike_mesh() on a given SMS, FVCOM or MIKE grid
    file.

    Parameters
    ----------
    triangles : ndarray
        Integer array of shape (ntri, 3). Each triangle is composed of
        three points and this contains the three node numbers (stored in
        nodes) which refer to the coordinates in X and Y (see below).
    x, y : ndarray
        Coordinates of each grid node.

    Returns
    -------
    elemSides : ndarray
        Length of each element described by triangles and x, y.

    """

    elemSides = np.zeros([np.shape(triangles)[0], 3])
    for it, tri in enumerate(triangles):
        pos1x, pos2x, pos3x = x[tri]
        pos1y, pos2y, pos3y = y[tri]

        elemSides[it, 0] = np.sqrt((pos1x - pos2x)**2 + (pos1y - pos2y)**2)
        elemSides[it, 1] = np.sqrt((pos2x - pos3x)**2 + (pos2y - pos3y)**2)
        elemSides[it, 2] = np.sqrt((pos3x - pos1x)**2 + (pos3y - pos1y)**2)

    return elemSides


def fix_coordinates(FVCOM, UTMZone, inVars=['x', 'y']):
    """
    Use the UTM_to_LL function to convert the grid from UTM to Lat/Long. Returns
    longitude and latitude in the range -180 to 180.

    By default, the variables which will be converted from UTM to Lat/Long are
    'x' and 'y'. To specify a different pair, give inVars=['xc', 'yc'], for
    example, to convert the 'xc' and 'yc' variables instead. Their order should
    be x-direction followed by y-direction.

    Parameters
    ----------
    FVCOM : dict
        Dict of the FVCOM model results (see read_FVCOM_results.readFVCOM).
    UTMZone : str
        UTM Zone (e.g. '30N').
    inVars : list, optional
        List of strings specifying the keys for FVCOM to be used as input.
        Defaults to ['x', 'y'] but if you wanted to convert element centres,
        change to ['xc', 'yc'] instead.

    Returns
    -------
    X, Y : ndarray
        Converted coordinates in longitude and latitude.

    """

    try:
        Y = np.zeros(np.shape(FVCOM[inVars[1]])) * np.nan
        X = np.zeros(np.shape(FVCOM[inVars[0]])) * np.nan
    except IOError:
        print(
            "Couldn't find the {} or {} variables in the FVCOM dict.".format(
                inVars[0], inVars[1],
                end=''
            )
        )
        print('Check you loaded them and try again.')

    for count, posXY in enumerate(zip(FVCOM[inVars[0]], FVCOM[inVars[1]])):

        posX = posXY[0]
        posY = posXY[1]

        # 23 is the WGS84 ellipsoid
        tmpLat, tmpLon = UTM_to_LL(23, posY, posX, UTMZone)

        Y[count] = tmpLat
        X[count] = tmpLon

    # Make the range -180 to 180 rather than 0 to 360.
    if np.min(X) >= 0:
        X[X > 180] = X[X > 180] - 360

    return X, Y


def clip_triangulation(MODEL, sideLength, keys=['xc', 'yc']):
    """
    Make a new triangulation of the element centres and clip according
    to a maximum length.

    Parameters
    ----------
    MODEL : dict
        Contains the MODEL model results. Keys are assumed to be ['xc', 'yc']
        unless the optional argument `keys' is specified (see below).
    sideLength : float
        Maximum length of an element before it is clipped.
    keys : list, optional
        List of two keys to use as the x and y coordinates for the
        triangulation. Defaults to ['xc', 'yc'].

    Returns
    -------
    triClip : ndarray
        Triangulation (indices of the coordinates which make up an
        element) of the new clipped elements. This can be used with the
        input coordinates in MODEL to plot the new unstructured grid.

    """

    tri = Triangulation(MODEL[keys[0]], MODEL[keys[1]]).triangles

    # Get the length of all element edges
    xx, yy = MODEL[keys[0]][tri], MODEL[keys[1]][tri]
    dx = np.empty(np.shape(xx))
    dy = np.empty(np.shape(yy))
    sxy = np.empty(np.shape(xx))
    dx[:, 0] = xx[:, 0] - xx[:, 1]
    dx[:, 1] = xx[:, 1] - xx[:, 2]
    dx[:, 2] = xx[:, 2] - xx[:, 0]
    dy[:, 0] = yy[:, 0] - yy[:, 1]
    dy[:, 1] = yy[:, 1] - yy[:, 2]
    dy[:, 2] = yy[:, 2] - yy[:, 0]
    sxy[:, 0] = np.sqrt(dx[:, 0]**2 + dy[:, 1]**2)
    sxy[:, 1] = np.sqrt(dx[:, 1]**2 + dy[:, 2]**2)
    sxy[:, 2] = np.sqrt(dx[:, 2]**2 + dy[:, 0]**2)

    triClip = []
    for i, t in enumerate(sxy):
        if max(t) <= sideLength:
            # Keep this element
            triClip.append(tri[i])

    triClip = np.asarray(triClip)

    return triClip


def get_river_config(fileName, noisy=False, zeroindex=False):
    """
    Parse the rivers namelist to extract the parameters and their values.
    Returns a dict of the parameters with the associated values for all the
    rivers defined in the namelist.

    Parameters
    ----------
    fileName : str
        Full path to an FVCOM Rivers name list.
    noisy : bool, optional
        Set to True to enable verbose output. Defaults to False.
    zeroindex : bool, optional
        Set to True to convert indices from 1-based to 0-based.

    Returns
    -------
    rivers : dict
        Dict of the parameters for each river defind in the name list.
        Dictionary keys are the name list parameter names (e.g. RIVER_NAME).

    Notes
    -----

    The indices returned in RIVER_GRID_LOCATION are 1-based (i.e. read in raw
    from the nml file). For use in Python, you can either subtract 1 yourself,
    or pass zeroindex=True to this function.

    """

    f = open(fileName)
    lines = f.readlines()
    rivers = {}
    for line in lines:
        line = line.strip()

        if line and not line.startswith('&') and not line.startswith('/'):
            param, value = [i.strip(",' ") for i in line.split('=')]
            if param in rivers:
                rivers[param].append(value)
            else:
                rivers[param] = [value]

    if noisy:
        print('Found {} rivers.'.format(len(rivers['RIVER_NAME'])))

    f.close()

    if zeroindex and 'RIVER_GRID_LOCATION' in rivers:
        rivers['RIVER_GRID_LOCATION'] = [int(i) - 1 for i in rivers['RIVER_GRID_LOCATION']]

    return rivers


def get_rivers(discharge, positions, noisy=False):
    """
    Extract the modified POLCOMS positions and the discharge data.

    Parameters
    ----------
    discharge : list
        Full path to the POLCOMS flw discharge ASCII file(s) for a given year.
        Number of rows is time, number of columns is number of rivers. The
        order of the locations in the positions file must match the order of
        the
    positions : str
        Full path to an ASCII file of the (modified) positions of the POLCOMS
        rivers as lon, lat, name.
    noisy : bool
        Set to True to enable verbose output (defaults to False)

    Returns
    -------
    rivers : dict
        Dictionary of the time series for each location in the positions file.
        For multiple discharge files, the data are appended in time. Dictionary
        keys are the river names in the positions file. N.B. The concatenation
        assumes the files are given in chronological order.
    locations : dict
        Dictionary of longitudes and latitudes for each of the rivers in the
        positions file. Keys are the river names.

    """

    f = open(positions, 'r')
    lines = f.readlines()
    locations = {}
    order = 0
    for c, line in enumerate(lines):
        if c > 0:
            line = line.strip()
            lon, lat, name = line.split(',')
            if name.strip() in locations:
                # Key already exists... just append a 1 to the key name.
                if noisy:
                    print('Duplicate key {}. Renaming to {}_1'.format(
                        name.strip(), name.strip()))

                locations[name.strip() + '_1'] = [float(lon),
                    float(lat),
                    order]
            else:
                locations[name.strip()] = [float(lon), float(lat), order]

            # Keep a track of the order we're putting the data into the dict
            # for the extraction to the flux array.
            order += 1

    f.close()

    rivers = {}

    for c, dfile in enumerate(discharge):
        if noisy:
            print('Reading in river discharge from file {}... '.format(dfile),
                  end=' ')

        # Just dump the file with np.genfromtxt.
        if c == 0:
            flux = np.genfromtxt(dfile)
        else:
            flux = np.vstack((flux, np.genfromtxt(dfile)))

        if noisy:
            print('done.')

    if flux.shape[-1] != len(list(locations.keys())):
        raise Exception('Inconsistent number of rivers and discharge profiles')

    # Now we need to iterate through the names and create the dict with the
    # relevant data.
    for station in locations:
        # Get the order index from the locations dict.
        n = locations[station][-1]
        # Extract the data from the flux array.
        rivers[station] = flux[:, n]

    return rivers, locations


def mesh2grid(meshX, meshY, meshZ, nx, ny, thresh=None, noisy=False):
    """
    Resample the unstructured grid in meshX and meshY onto a regular grid whose
    size is nx by ny or which is specified by the arrays nx, ny. Optionally
    specify dist to control the proximity of a value considered valid.

    Parameters
    ----------
    meshX, meshY : ndarray
        Arrays of the unstructured grid (mesh) node positions.
    meshZ : ndarray
        Array of values to be resampled onto the regular grid. The shape of the
        array should have the nodes as the first dimension. All subsequent
        dimensions will be propagated automatically.
    nx, ny : int, ndarray
        Number of samples in x and y onto which to sample the unstructured
        grid. If given as a list or array, the values within the arrays are
        assumed to be positions in x and y.
    thresh : float, optional
        Distance beyond which a sample is considered too far from the current
        node to be included.
    noisy : bool, optional
        Set to True to enable verbose messages.

    Returns
    -------
    xx, yy : ndarray
        New position arrays (1D). Can be used with numpy.meshgrid to plot the
        resampled variables with matplotlib.pyplot.pcolor.
    zz : ndarray
        Array of the resampled data from meshZ. The first dimension from the
        input is now replaced with two dimensions (x, y). All other input
        dimensions follow.

    """

    if not thresh:
        thresh = np.inf

    # Get the extents of the input data.
    xmin, xmax, ymin, ymax = meshX.min(), meshX.max(), meshY.min(), meshY.max()

    if isinstance(nx, int) and isinstance(ny, int):
        xx = np.linspace(xmin, xmax, nx)
        yy = np.linspace(ymin, ymax, ny)
    else:
        xx = nx
        yy = ny

    # We need to check the input we're resampling for its number of dimensions
    # so we can create an output array of the right shape. We can just take the
    # shape of the input, omitting the first value (the nodes). That should
    # leave us with the right shape. This even works for 1D inputs (i.e. a
    # single value at each unstructured grid location).
    if isinstance(nx, int) and isinstance(ny, int):
        zz = np.empty((nx, ny) + meshZ.shape[1:]) * np.nan
    else:
        zz = np.empty((nx.shape) + meshZ.shape[1:]) * np.nan

    if noisy:
        if isinstance(nx, int) and isinstance(ny, int):
            print('Resampling unstructured to regular ({} by {}). '.format(nx, ny), end='')
            print('Be patient...')
        else:
            _nx, _ny = len(nx[:, 1]), len(ny[0, :])
            print('Resampling unstructured to regular ({} by {}). '.format(_nx, _ny), end='')
            print('Be patient...')

        sys.stdout.flush()

    if isinstance(nx, int) and isinstance(ny, int):
        for xi, xpos in enumerate(xx):
            # Do all the y-positions with findNearestPoint
            for yi, ypos in enumerate(yy):
                # Find the nearest node in the unstructured grid data and grab
                # its u and v values. If it's beyond some threshold distance,
                # leave the z value as NaN.
                dist = np.sqrt((meshX - xpos)**2 + (meshY - ypos)**2)

                # Get the index of the minimum and extract the values only if
                # the nearest point is within the threshold distance (thresh).
                if dist.min() < thresh:
                    idx = dist.argmin()

                    # The ... means "and all the other dimensions". Since we've
                    # asked for our input array to have the nodes as the first
                    # dimension, this means we can just get all the others when
                    # using the node index.
                    zz[xi, yi, ...] = meshZ[idx, ...]
    else:
        # We've been given positions, so run through those instead of our
        # regularly sampled grid.
        c = 0
        for ci, _ in enumerate(xx[0, :]):
            for ri, _ in enumerate(yy[:, 0]):
                if noisy:
                    if np.mod(c, 1000) == 0 or c == 0:
                        print('{} of {}'.format(
                            c,
                            len(xx[0, :]) * len(yy[:, 0])
                            ))
                c += 1

                dist = np.sqrt(
                    (meshX - xx[ri, ci])**2 + (meshY - yy[ri, ci])**2
                )
                if dist.min() < thresh:
                    idx = dist.argmin()
                    zz[ri, ci, ...] = meshZ[idx, ...]

    if noisy:
        print('done.')

    return xx, yy, zz


def line_sample(x, y, positions, num=0, return_distance=False, noisy=False):
    """
    Function to take an unstructured grid of positions x and y and find the
    points which fall closest to a line defined by the coordinate pairs start
    and end.

    If num=0 (default), then the line will be sampled at each nearest node; if
    num is greater than 1, then the line will be subdivided into num segments
    (at num + 1 points), and the closest point to that line used as the sample.

    Returns a list of array indices.

    N.B. Most of the calculations assume we're using cartesian coordinates.
    Things might get a bit funky if you're using spherical coordinates. Might
    be easier to convert before using this.

    Parameters
    ----------
    x, y : ndarray
        Position arrays for the unstructured grid.
    positions : ndarray
        Coordinate pairs of the sample line coordinates [[xpos, ypos], ...,
        [xpos, ypos]].  Units must match those in (x, y).
    num : int, optional
        Optionally specify a number of points to sample along the line
        described in `positions'. If no number is given, then the sampling of
        the line is based on the closest nodes to that line.
    return_distance : bool, optional
        Set to True to return the distance along the sampling line. Defaults
        to False.
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    idx : list
        List of indices for the nodes used in the line sample.
    line : ndarray
        List of positions which fall along the line described by (start, end).
        These are the projected positions of the nodes which fall closest to
        the line (not the positions of the nodes themselves).
    distance : ndarray, optional
        If `return_distance' is True, return the distance along the line
        described by the nodes in idx.

    """

    if not isinstance(num, int):
        raise TypeError('num must be an int')

    def __nodes_on_line__(xs, ys, start, end, pdist, noisy=False):
        """
        Child function to find all the points within the coordinates in xs and
        ys which fall along the line described by the coordinate pairs start
        and end. Uses pdist (distance of all coordinates from the line [start,
        end]) to select nodes.

        Parameters
        ----------
        xs, ys : ndarray
            Node position arrays.
        start, end : ndarray
            Coordinate pairs for the start and end of the sample line.
        pdist : ndarray
            Distance of the nodes in xs and ys from the line defined by
            `start' and `end'.

        Returns
        -------
        idx : list
            List of indices for the nodes used in the line sample.
        line : ndarray
            List of positions which fall along the line described by (start,
            end). These are the projected positions of the nodes which fall
            closest to the line (not the positions of the nodes themselves).

        """

        # Create empty lists for the indices and positions.
        sidx = []
        line = []

        beg = start  # seed the position with the start of the line.

        while True:

            # Find the nearest point to the start which hasn't already been
            # used (if this is the first iteration, we need to use all the
            # values).
            sx = np.ma.array(xs, mask=False)
            sy = np.ma.array(ys, mask=False)
            sx.mask[sidx] = True
            sy.mask[sidx] = True
            ndist = np.sqrt((sx - beg[0])**2 + (sy - beg[1])**2)

            # Create an array summing the distance from the current node with
            # the distance of all nodes to the line.
            sdist = ndist + pdist

            # Add the closest index to the list of indices. In some unusual
            # circumstances, the algorithm ends up going back along the line.
            # As such, add a check for the remaining distance to the end, and
            # if we're going backwards, find the next best point and use that.

            # Closest node index.
            tidx = sdist.argmin().astype(int)
            # Distance from the start point.
            fdist = np.sqrt((start[0] - xx[tidx])**2 +
                            (start[1] - yy[tidx])**2
                            ).min()
            # Distance to the end point.
            tdist = np.sqrt((end[0] - xx[tidx])**2 +
                            (end[1] - yy[tidx])**2
                            ).min()
            # Last node's distance to the end point.
            if len(sidx) >= 1:
                oldtdist = np.sqrt((end[0] - xx[sidx[-1]])**2 +
                                   (end[1] - yy[sidx[-1]])**2
                                   ).min()
            else:
                # Haven't found any points yet.
                oldtdist = tdist

            if fdist >= length:
                # We've gone beyond the end of the line, so don't bother trying
                # to find another node.  Leave the if block so we actually add
                # the current index and position to sidx and line. We'll break
                # out of the main while loop a bit later.
                break

            elif tdist > oldtdist:
                # We're moving away from the end point. Find the closest point
                # in the direction of the end point.
                c = 0
                sdistidx = np.argsort(sdist)

                while True:
                    try:
                        tidx = sdistidx[c]
                        tdist = np.sqrt((end[0] - xx[tidx])**2 +
                                        (end[1] - yy[tidx])**2
                                        ).min()
                        c += 1
                    except IndexError:
                        # Eh, we've run out of indices for some reason. Let's
                        # just go with whatever we had as the last set of
                        # values.
                        break

                    if tdist < oldtdist:
                        break

            sidx.append(tidx)

            line.append([xx[tidx], yy[tidx]])

            if noisy:
                done = 100 - ((tdist / length) * 100)
                if len(sidx) == 1:
                    print('Found {} node ({:.2f}%)'.format(len(sidx), done))
                else:
                    print('Found {} nodes ({:.2f}%)'.format(len(sidx), done))

            # Check if we've gone beyond the end of the line (by checking the
            # length of the sampled line), and if so, break out of the loop.
            # Otherwise, carry on.
            if beg.tolist() == start.tolist() or fdist <= length:
                # Reset the beginning point for the next iteration if we're at
                # the start or within the line extent.
                beg = np.array(([xx[tidx], yy[tidx]]))
            else:
                if noisy:
                    print('Reached the end of the line segment')

                break

        # Convert the list to an array before we leave.
        line = np.asarray(line)

        return sidx, line

    # To do multi-segment lines, we'll break each one down into a separate
    # line and do those sequentially. This means I don't have to rewrite
    # masses of the existing code and it's still pretty easy to understand (for
    # me at least!).
    nlocations = len(positions)

    idx = []
    line = []
    if return_distance:
        dist = []

    for xy in range(1, nlocations):
        # Make the first segment.
        start = positions[xy - 1]
        end = positions[xy]

        # Get the lower left and upper right coordinates of this section of the
        # line.
        lowerx = min(start[0], end[0])
        lowery = min(start[1], end[1])
        upperx = max(start[0], end[0])
        uppery = max(start[1], end[1])
        ll = [lowerx, lowery]
        ur = [upperx, uppery]

        lx = float(end[0] - start[0])
        ly = float(end[1] - start[1])
        length = np.sqrt(lx**2 + ly**2)
        dcn = np.degrees(np.arctan2(lx, ly))

        if num > 1:
            # This is easy: decimate the line between the start and end and
            # find the grid nodes which fall closest to each point in the line.

            # Create the line segments
            inc = length / num
            xx = start[0] + (np.cumsum(np.hstack((0, np.repeat(inc, num)))) *
                             np.sin(np.radians(dcn)))
            yy = start[1] + (np.cumsum(np.hstack((0, np.repeat(inc, num)))) *
                             np.cos(np.radians(dcn)))
            [line.append(xy) for xy in zip([xx, yy])]

            # For each position in the line array, find the nearest indices in
            # the supplied unstructured grid. We'll use our existing function
            # findNearestPoint for this.
            _, _, _, tidx = find_nearest_point(x, y, xx, yy, noisy=noisy)
            [idx.append(i) for i in tidx.tolist()]

        else:
            # So really, this shouldn't be that difficult, all we're doing is
            # finding the intersection of two lines which are orthogonal to one
            # another. We basically need to find the equations of both lines
            # and then solve for the intersection.

            # First things first, clip the coordinates to a rectangle defined
            # by the start and end coordinates. We'll use a buffer based on the
            # size of the elements which surround the first and last nodes.
            # This ensures we'll get relatively sensible results if the profile
            # is relatively flat or vertical. Use the six closest nodes as the
            # definition of surrounding elements.
            bstart = np.mean(np.sort(np.sqrt((x - start[0])**2 +
                                             (y - start[1])**2))[:6])
            bend = np.mean(np.sort(np.sqrt((x - end[0])**2 +
                                           (y - end[1])**2))[:6])
            # Use the larger of the two lengths to be on the safe side.
            bb = 2 * np.max((bstart, bend))
            ss = np.where((x >= (ll[0] - bb)) *
                          (x <= (ur[0] + bb)) *
                          (y >= (ll[1] - bb)) *
                          (y <= (ur[1] + bb))
                          )[0]
            xs = x[ss]
            ys = y[ss]

            # Sampling line equation.
            if lx == 0:
                # Vertical line.
                yy = ys
                xx = np.repeat(start[0], len(yy))

            elif ly == 0:
                # Horizontal line.
                xx = xs
                yy = np.repeat(start[1], len(xx))

            else:
                m1 = ly / lx  # sample line gradient
                c1 = start[1] - (m1 * start[0])  # sample line intercept

                # Find the equation of the line through all nodes in the domain
                # normal to the original line (gradient = -1 / m).
                m2 = -1 / m1
                c2 = ys - (m2 * xs)

                # Now find the intersection of the sample line and then all the
                # lines which go through the nodes.
                #   1a. y1 = (m1 * x1) + c1  # sample line
                #   2a. y2 = (m2 * x2) + c2  # line normal to it
                # Rearrange 1a for x.
                #   1b. x1 = (y1 - c1) / m1

                # Substitute equation 1a (y1) into 2a and solve for x.
                xx = (c2 - c1) / (m1 - m2)
                # Substitute xx into 2a to solve for y.
                yy = (m2 * xx) + c2

            # Find the distance from the original nodes to their corresponding
            # projected node.
            pdist = np.sqrt((xx - xs)**2 + (yy - ys)**2)

            # Now we need to start our loop until we get beyond the end of the
            # line.
            tidx, tline = __nodes_on_line__(xs, ys,
                                            start, end,
                                            pdist,
                                            noisy=noisy)

            # Now, if we're being asked to return the distance along the
            # profile line (rather than the distance along the line
            # connecting the positions in xs and ys together), generate that
            # for this segment here.
            if return_distance:
                # Make the distances relative to the first node we've found.
                # Doing this, instead of using the coordinates given in start
                # means we don't end up with negative distances, which means
                # we don't have to worry about signed distance functions and
                # other fun things to get proper distance along the transect.
                xdist = np.diff(xx[tidx])
                ydist = np.diff(xx[tidx])
                tdist = np.hstack((0, np.cumsum(np.sqrt(xdist**2 + ydist**2))))
                # Make distances relative to the end of the last segment,
                # if we have one.
                if not dist:
                    distmax = 0
                else:
                    distmax = np.max(dist)
                [dist.append(i + distmax) for i in tdist.tolist()]

            [line.append(i) for i in tline.tolist()]
            # Return the indices in the context of the original input arrays so
            # we can more easily extract them from the main data arrays.
            [idx.append(i) for i in ss[tidx]]

    # Return the distance as a numpy array rather than a list.
    if return_distance:
        dist = np.asarray(dist)

    # Make the line list an array for easier plotting.
    line = np.asarray(line)

    if return_distance:
        return idx, line, dist
    else:
        return idx, line


def connectivity(p, t):
    """
    Assemble connectivity data for a triangular mesh.

    The edge based connectivity is built for a triangular mesh and the boundary
    nodes identified. This data should be useful when implementing FE/FV
    methods using triangular meshes.

    Parameters
    ----------
    p : ndarray
        Nx2 array of nodes coordinates, [[x1, y1], [x2, y2], etc.]
    t : ndarray
        Mx3 array of triangles as indices, [[n11, n12, n13], [n21, n22, n23],
        etc.]

    Returns
    -------
    e : ndarray
        Kx2 array of unique mesh edges - [[n11, n12], [n21, n22], etc.]
    te : ndarray
        Mx3 array of triangles as indices into e, [[e11, e12, e13], [e21, e22,
        e23], etc.]
    e2t : ndarray
        Kx2 array of triangle neighbours for unique mesh edges - [[t11, t12],
        [t21, t22], etc]. Each row has two entries corresponding to the
        triangle numbers associated with each edge in e. Boundary edges have
        e2t[i, 1] = -1.
    bnd : ndarray, bool
        Nx1 logical array identifying boundary nodes. p[i, :] is a boundary
        node if bnd[i] = True.

    Notes
    -----
    Python translation of the MATLAB MESH2D connectivity function by Darren
    Engwirda.

    """

    def _unique_rows(A, return_index=False, return_inverse=False):
        """
        Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
        where B is the unique rows of A and I and J satisfy
        A = B[J,:] and B = A[I,:]

        Returns I if return_index is True
        Returns J if return_inverse is True

        Taken from https://github.com/numpy/numpy/issues/2871

        """
        A = np.require(A, requirements='C')
        assert A.ndim == 2, "array must be 2-dim'l"

        B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
                      return_index=return_index,
                      return_inverse=return_inverse)

        if return_index or return_inverse:
            return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
                + B[1:]
        else:
            return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

    if p.shape[-1] != 2:
        raise Exception('p must be an Nx2 array')
    if t.shape[-1] != 3:
        raise Exception('t must be an Mx3 array')
    if np.any(t.ravel() < 0) or t.max() > p.shape[0] - 1:
        raise Exception('Invalid t')

    # Unique mesh edges as indices into p
    numt = t.shape[0]
    # Triangle indices
    vect = np.arange(numt)
    # Edges - not unique
    e = np.vstack(([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]))
    # Unique edges
    e, j = _unique_rows(np.sort(e, axis=1), return_inverse=True)
    # Unique edges in each triangle
    te = np.column_stack((j[vect], j[vect + numt], j[vect + (2 * numt)]))

    # Edge-to-triangle connectivity
    # Each row has two entries corresponding to the triangle numbers
    # associated with each edge. Boundary edges have e2t[i, 1] = -1.
    nume = e.shape[0]
    e2t = np.zeros((nume, 2)).astype(int) - 1
    for k in range(numt):
        for j in range(3):
            ce = te[k, j]
            if e2t[ce, 0] == -1:
                e2t[ce, 0] = k
            else:
                e2t[ce, 1] = k

    # Flag boundary nodes
    bnd = np.zeros((p.shape[0],)).astype(bool)
    # True for bnd nodes
    bnd[e[e2t[:, 1] == -1, :]] = True

    return e, te, e2t, bnd


def clip_domain(x, y, extents, noisy=False):
    """
    Function to find the indices for the positions in `x' and `y' which fall within the
    bounding box or polygon defined in extents. For a bounding box supply an array or list
    of size (4,) with (xmin, xmax, ymin, ymax). For a polygon provide a list of x,y points
    or an Nx2 array of x,y points.

    Parameters
    ----------
    x, y : ndarray
        x and y coordinate arrays.
    extents : ndarray or list
        minimum and maximum of the extents of the x and y coordinates for the
        bounding box (xmin, xmax, ymin, ymax) or for a polygon a list or array
        of x,y coordinates

    Returns
    -------
    mask : ndarrary
        Mask (True = within the bounding box, False = not) array of the indices
        of the positions in pos which fall within the bounding box.

    """
    if np.asarray(extents).shape != (4,):
        poly_path = mplPath.Path(extents)
        mask = np.where(np.asarray(poly_path.contains_points(np.asarray([x,y]).T)))[0]
    else:
        mask = np.where((x >= extents[0]) *
            (x <= extents[1]) *
            (y >= extents[2]) *
            (y <= extents[3]))[0]

    if noisy:
        print('Subset contains {} points of {} total.'.format(len(mask),
                                                              len(x)))

    return mask


def find_connected_nodes(n, triangles):
    """
    Return the IDs of the nodes surrounding node number `n'.

    Parameters
    ----------
    n : int
        Node ID around which to find the connected nodes.
    triangles : ndarray
        Triangulation matrix to find the connected nodes. Shape is [nele,
        3].

    Returns
    -------
    surroundingidx : ndarray
        Indices of the surrounding nodes.

    See Also
    --------
    PyFVCOM.grid.find_connected_elements().

    Notes
    -----

    Check it works with:
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy.spatial import Delaunay
    >>> x, y = np.meshgrid(np.arange(25), np.arange(100, 125))
    >>> x = x.flatten() + np.random.randn(x.size) * 0.1
    >>> y = y.flatten() + np.random.randn(y.size) * 0.1
    >>> tri = Delaunay(np.array((x, y)).transpose())
    >>> for n in np.linspace(1, len(x) - 1, 5).astype(int):
    ...     aa = surrounders(n, tri.vertices)
    ...     plt.figure()
    ...     plt.triplot(x, y, tri.vertices, zorder=20, alpha=0.5)
    ...     plt.plot(x[n], y[n], 'ro', label='central node')
    ...     plt.plot(x[aa], y[aa], 'ko', label='connected nodes')
    ...     plt.xlim(x[aa].min() - 1, x[aa].max() + 1)
    ...     plt.ylim(y[aa].min() - 1, y[aa].max() + 1)
    ...     plt.legend(numpoints=1)

    """

    eidx = np.max((np.abs(triangles - n) == 0), axis=1)
    surroundingidx = np.unique(triangles[eidx][triangles[eidx] != n])

    return surroundingidx


def find_connected_elements(n, triangles):
    """
    Return the IDs of the elements connected to node number `n'.

    Parameters
    ----------
    n : int or iterable
        Node ID(s) around which to find the connected elements. If more than
        one node is given, the unique elements for all nodes are returned.
        Order of results is not maintained.
    triangles : ndarray
        Triangulation matrix to find the connected elements. Shape is [nele,
        3].

    Returns
    -------
    surroundingidx : ndarray
        Indices of the surrounding elements.

    See Also
    --------
    PyFVCOM.grid.surrounders().

    """

    try:
        surroundingidx = []
        for ni in n:
            idx = np.argwhere(triangles == ni)[:, 0]
            surroundingidx.append(idx)
        surroundingidx = np.asarray([item for sublist in surroundingidx for item in sublist])
        surroundingidx = np.unique(surroundingidx)
    except TypeError:
        surroundingidx = np.argwhere(triangles == n)[:, 0]

    return surroundingidx


def get_area(v1, v2, v3):
    """ Calculate the area of a triangle/set of triangles.

    Parameters
    ----------
    v1, v2, v3 : tuple, list (float, float)
        Coordinate pairs (x, y) of the three vertices of a triangle. Can be 1D
        arrays of positions or lists of positions.

    Returns
    -------
    area : tuple, ndarray
        Area of the triangle(s). Units of v0, v1 and v2.

    Examples
    --------
    >>> v1 = ((4, 0), (0, 0))
    >>> v2 = ((10, -3), (2, 6))
    >>> v3 = ((7, 9), (10, -5))
    >>> a = get_area(v1, v2, v3)
    >>> print(a)
    [ 31.5  35. ]

    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v3 = np.asarray(v3)

    if np.size(v1) == 2:
        # Single position
        area = 0.5 * (v1[0] * (v2[1] - v3[1]) + v2[0] * (v3[1] - v1[1]) + v3[0] * (v1[1] - v2[1]))
    else:
        # Array of positions
        area = 0.5 * (v1[:, 0] * (v2[:, 1] - v3[:, 1]) + v2[:, 0] * (v3[:, 1] - v1[:, 1]) + v3[:, 0] * (v1[:, 1] - v2[:, 1]))

    return abs(area)


def find_bad_node(nv, node_id):
    """
    Check nodes on the boundary of a grid for nodes connected to a single
    element only. These elements will always have zero velocities,
    which means rivers input here will cause the model crash as the water
    never advects away. It is also computationally wasteful to include these
    elements.

    Return True if it was bad, False otherwise.

    Parameters
    ----------
    nv : ndarray
        Connectivity table for the grid.
    node_id : int
        Node ID to check.

    Returns
    -------
    bad : bool
        Status of the supplied node ID: True is connected to a single element
        only (bad), False is connected to multiple elements (good).

    Examples
    --------
    >>> from PyFVCOM.grid import read_sms_mesh, connectivity
    >>> nv, nodes, x, y, z, _ = read_sms_mesh('test_mesh.2dm')
    >>> _, _, _, bnd = connectivity(np.asarray((x, y)).transpose(), nv)
    >>> coast_nodes = nodes[bnd]
    >>> bad_nodes = np.empty(coast_nodes.shape).astype(bool)
    >>> for i, node_id in enumerate(coast_nodes):
    >>>     bad_nodes[i] = find_bad_node(node_id, nv)

    """

    was = False
    if len(np.argwhere(nv == node_id)) == 1:
        was = True

    return was


def trigradient(x, y, z, t=None):
    """
    Returns the gradient of `z' defined on the irregular mesh with Delaunay
    triangulation `t'. `dx' corresponds to the partial derivative dZ/dX,
    and `dy' corresponds to the partial derivative dZ/dY.

    Parameters
    ----------
    x, y, z : array_like
        Horizontal (`x' and `y') positions and vertical position (`z').
    t : array_like, optional
        Connectivity table for the grid. If omitted, one will be calculated
        automatically.

    Returns
    -------
    dx, dy : ndarray
        `dx' corresponds to the partial derivative dZ/dX, and `dy'
        corresponds to the partial derivative dZ/dY.

    Example
    -------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from PyFVCOM.grid import trigradient
    >>> from matplotlib.tri.triangulation import Triangulation
    >>> x, y = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
    >>> x[1:2:-1, :] = x[1:2:-1, :] + 0.1 / 2
    >>> tt = Triangulation(x.ravel(), y.ravel())
    >>> z = x * np.exp(-x**2 - y**2)
    >>> dx, dy = trigradient(x.ravel(), y.ravel(), z.ravel())
    >>> dzdx = (1 / x - 2 * x) * z
    >>> dzdy = -2 * y * z
    >>> plt.figure(1)
    >>> plt.quiver(x.ravel(), y.ravel(), dzdx.ravel(), dzdy.ravel(),
    >>>            color='r', label='Exact')
    >>> plt.quiver(x.ravel(), y.ravel(), dx, dy,
    >>>            color='k', label='trigradient', alpha=0.5)
    >>> tp = plt.tripcolor(x.ravel(), y.ravel(), tt.triangles, z.ravel(),
    >>>                    zorder=0)
    >>> plt.colorbar(tp)
    >>> plt.legend()

    Notes
    -----
    Lifted from:
        http://matplotlib.org/examples/pylab_examples/trigradient_demo.html

    """

    if np.any(t):
        tt = Triangulation(x.ravel(), y.ravel())
    else:
        tt = Triangulation(x.ravel(), y.ravel(), t)

    tci = CubicTriInterpolator(tt, z.ravel())
    # Gradient requested here at the mesh nodes but could be anywhere else:
    dx, dy = tci.gradient(tt.x, tt.y)

    return dx, dy


def rotate_points(x, y, origin, angle):
    """
    Rotate the points in `x' and `y' around the point `origin' by `angle'
    degrees.

    Parameters
    ----------
    x, y : ndarray
        Coordinates to rotate.
    origin : list, ndarray
        Point about which to rotate the grid (x, y).
    angle : float
        Angle (in degrees) by which to rotate the grid. Positive clockwise.

    Returns
    -------
    xr, yr : ndarray
        Rotated coordinates.

    """

    # Make the x and y values relative to the origin.
    x -= origin[0]
    y -= origin[1]

    # Rotate clockwise by some angle `rot'. See
    # http://stackoverflow.com/questions/29708840 for a faster version on 2D
    # arrays with np.einsum if necessary in future.
    xr = np.cos(np.deg2rad(angle)) * x + np.sin(np.deg2rad(angle)) * y
    yr = -np.sin(np.deg2rad(angle)) * x + np.cos(np.deg2rad(angle)) * y

    # Add the origin back to restore the right coordinates.
    xr += origin[0]
    yr += origin[1]

    return xr, yr


def get_boundary_polygons(triangle, noisy=False):
    """
    Gets a list of the grid boundary nodes ordered correctly.

    ASSUMPTIONS: This function assumes a 'clean' FVCOM grid, i.e. no
    elements with 3 boundary nodes and no single element width channels.

    Parameters
    ----------
    triangle : ndarray
        The triangle connectivity matrix as produced by the read_fvcom_mesh
        function.

    Returns
    -------
    boundary_polygon_list : list
        List of integer arrays. Each array is one closed boundary polygon with
        the integers referring to node number.

    """

    u, c = np.unique(triangle, return_counts=True)
    uc = np.asarray([u, c]).T

    nodes_lt_4 = np.asarray(uc[uc[:,1] < 4, 0], dtype=int)
    boundary_polygon_list = []

    while len(nodes_lt_4) > 0:

        start_node = nodes_lt_4[0]

        boundary_node_list = [start_node, get_attached_unique_nodes(start_node, triangle)[-1]]

        full_loop = True
        while full_loop:
            next_nodes = get_attached_unique_nodes(boundary_node_list[-1], triangle)
            node_ind = 0
            len_bl = len(boundary_node_list)
            if noisy:
                print(len_bl)
            while len(boundary_node_list) == len_bl:
                try:
                    if next_nodes[node_ind] not in boundary_node_list:
                        boundary_node_list.append(next_nodes[node_ind])
                    else:
                        node_ind += 1
                except:
                    full_loop = False
                    len_bl += 1

        boundary_polygon_list.append(np.asarray(boundary_node_list))
        nodes_lt_4 = np.asarray(list(set(nodes_lt_4) - set(boundary_node_list)), dtype=int)

    return boundary_polygon_list


def get_attached_unique_nodes(this_node, trinodes):
    """
    Find the nodes on the boundary connected to `this_node'.

    Parameters
    ----------
    this_node : int
        Node ID.
    trinodes : ndarray
        Triangulation table for an unstructured grid.

    Returns
    -------
    connected_nodes : ndarray
        IDs of the nodes connected to `this_node' on the boundary. If `this_node' is not on the boundary,
        `connected_nodes' is empty.

    """

    all_trinodes = trinodes[(trinodes[:, 0] == this_node) | (trinodes[:, 1] == this_node) | (trinodes[:, 2] == this_node), :]
    u, c = np.unique(all_trinodes, return_counts=True)

    return u[c == 1]


def grid_metrics(tri, noisy=False):
    """
    Calculate unstructured grid metrics (most of FVCOM's tge.F).

    Parameters
    ----------
    tri : ndarray
        Triangulation table for the grid.
    noisy : bool
        Set to True to enable verbose output (default = False)

    Returns
    -------
    ntve : ndarray
        The number of neighboring elements of each grid node
    nbve : ndarray
        nbve(i,1->ntve(i)) = ntve elements containing node i
    nbe : ndarray
        Indices of tri for the elements connected to each element in the domain. To visualise:
            plt.plot(x[tri[1000, :], y[tri[1000, :], 'ro')
            plt.plot(x[tri[nbe[1000], :]] and y[tri[nbe[1000], :]], 'k.')
        plots the 999th element nodes with the nodes of the surrounding elements too.
    isbce : ndarray
        Flag if element is on the boundary (True = yes, False = no)
    isonb : ndarray
        Flag if node is on the boundary (True = yes, False = no)

    Notes
    -----
    This is more or less a direct translation from FORTRAN (FVCOM's tge.F).

    """

    m = len(np.unique(tri.ravel()))

    # Allocate all our arrays. Use masked by default arrays so we only use valid indices.
    isonb = np.zeros(m).astype(bool)
    ntve = np.zeros(m, dtype=int)
    nbe = np.ma.array(np.zeros(tri.shape, dtype=int), mask=True)
    nbve = np.ma.array(np.zeros((m, 10), dtype=int), mask=True)
    # Number of elements connected to each node (ntve) and the IDs of the elements connected to each node (nbve).
    if noisy:
        print('Counting neighbouring nodes and elements')
    for i, (n1, n2, n3) in enumerate(tri):
        nbve[tri[i, 0], ntve[n1]] = i
        nbve[tri[i, 1], ntve[n2]] = i
        nbve[tri[i, 2], ntve[n3]] = i
        # Only increment the counters afterwards as Python indexes from 0.
        ntve[n1] += 1
        ntve[n2] += 1
        ntve[n3] += 1

    if noisy:
        print('Getting neighbouring elements for each element')
    # Get the element IDs connected to each element.
    for i, (n1, n2, n3) in enumerate(tri):
        for j1 in range(ntve[n1]):
            for j2 in range(ntve[n2]):
                if nbve[n1, j1] == nbve[n2, j2] and nbve[n1, j1] != i:
                    nbe[i, 2] = nbve[n1, j1]
        for j2 in range(ntve[n2]):
            for j3 in range(ntve[n3]):
                if nbve[n2, j2] == nbve[n3, j3] and nbve[n2, j2] != i:
                    nbe[i, 0] = nbve[n2, j2]
        for j1 in range(ntve[n1]):
            for j3 in range(ntve[n3]):
                if nbve[n1, j1] == nbve[n3, j3] and nbve[n1, j1] != i:
                    nbe[i, 1] = nbve[n3, j3]

    if noisy:
        print('Getting boundary element IDs')
    isbce = np.max(nbe.mask, axis=1)

    if noisy:
        print('Getting boundary node IDs')

    # Get the boundary node IDs. Holy nested list comprehensions, Batman!
    boundary_element_node_ids = np.unique(tri[isbce, :]).ravel()
    boundary_nodes = []
    for i in boundary_element_node_ids:
        current_nodes = get_attached_unique_nodes(i, tri)
        if np.any(current_nodes):
            boundary_nodes += current_nodes.tolist()
    boundary_nodes = np.unique(boundary_nodes)
    # Make a boolean of that.
    isonb[boundary_nodes] = True

    return ntve, nbve, nbe, isbce, isonb


def control_volumes(x, y, tri, node_control=True, element_control=True, noisy=False):
    """
    This calculates the surface area of individual control volumes consisted of triangles with a common node point.

    Parameters
    ----------
    x, y : ndarray
        Node positions
    tri : ndarray
        Triangulation table for the unstructured grid.
    node_control : bool
        Set to False to disable calculation of node control volumes. Defaults to True.
    element_control : bool
        Set to False to disable calculation of element control volumes. Defaults to True.
    noisy : bool
        Set to True to enable verbose output.

    Returns
    -------
    art1 : ndarray
        Area of interior control volume (for node value integration)
    art2 : ndarray
        Sum area of all cells around each node.

    Notes
    -----
    This is a python reimplementation of the FVCOM function CELL_AREA in cell_area.F. Whilst the reimplementation is
    coded with efficiency in mind (the calculations occur in parallel), this is still slow for large grids. Please be
    patient!

    """

    if not node_control and not element_control:
        raise ValueError("Set either `node_control' or `element_control' to `True'")

    pool = multiprocessing.Pool()

    m = len(x)  # number of nodes

    # Calculate art1 (control volume for fluxes of node-based values). I do this differently from how it's done in
    # FVCOM as I can't wrap my head around the seemingly needlessly complicated approach they've taken. Here,
    # my approach is:
    #   1. For each node, find all the elements connected to it (find_connected_elements).
    #   2. Identify the nodes in all those elements.
    #   3. Find the position of the halfway point along each vertex between our current node and all the other nodes
    #   connected to it.
    #   4. Using those positions and the positions of the centre of each element, create a polygon (ordered clockwise).
    #   5. Calculate the area of that polygon with the shapely tools.
    if node_control:
        if noisy:
            print('Compute control volume for fluxes at nodes (art1)')
        xc = nodes2elems(x, tri)
        yc = nodes2elems(y, tri)
        art1 = pool.map(partial(node_control_area, x=x, y=y, xc=xc, yc=yc, tri=tri), range(m))

    # Compute area of control volume art2(i) = sum(all tris surrounding node i)
    if element_control:
        if noisy:
            print('Compute control volume for fluxes over elements (art2)')
        art = get_area(np.asarray((x[tri[:, 0]], y[tri[:, 0]])).T, np.asarray((x[tri[:, 1]], y[tri[:, 1]])).T, np.asarray((x[tri[:, 2]], y[tri[:, 2]])).T)
        art2 = pool.map(partial(element_control_area, triangles=tri, art=art), range(m))

    pool.close()

    if node_control and element_control:
        return art1, art2
    elif node_control and not element_control:
        return art1
    elif not node_control and element_control:
        return art2


def node_control_area(n, x, y, xc, yc, tri):
    """
    Worker function to calculate the control volume for fluxes of node-based values for a given node.

    Parameters
    ----------
    n : int
        Current node ID.
    x, y : list-like
        Node positions
    xc, yc : list-like
        Element centre positions
    tri : list-like
        Unstructured grid triangulation table.

    Returns
    -------
    node_area : float
        Node control volume area in x or y length units squared.

    """

    connected_elements = find_connected_elements(n, tri)
    area = 0
    # Create two triangles which are from the mid-point of each vertex with the current node and the element centre.
    # Sum the areas as we go.
    for element in connected_elements:
        # Find the nodes in this element.
        connected_nodes = np.unique(tri[element, :])
        other_nodes = connected_nodes[connected_nodes != n]
        centre_x = xc[element]
        centre_y = yc[element]
        mid_x, mid_y = [], []
        for node in other_nodes:
            mid_x.append(x[n] - ((x[n] - x[node]) / 2))
            mid_y.append(y[n] - ((y[n] - y[node]) / 2))
        # Now calculate the area of the triangles formed by [(x[n], y[n]), (centre_x, centre_y), (mid_x, mid_y)].
        for mid_xy in zip(mid_x, mid_y):
            area += get_area((x[n], y[n]), mid_xy, (centre_x, centre_y))

    return area


def element_control_area(node, triangles, art):
    """
    Worker function to calculate the control volume for fluxes of element-based values for a given node.

    Parameters
    ----------
    node : int
        Node ID.
    triangles : list-like
        Unstructured grid triangulation table.
    art : list-like
        Element areas.

    Returns
    -------
    element_area : float
        Element control volume area in x or y length units squared.

    Notes
    -----

    """

    connected_elements = find_connected_elements(node, triangles)

    return np.sum(art[connected_elements])


def unstructured_grid_volume(area, depth, surface_elevation, thickness, depth_integrated=False):
    """
    Calculate the volume for every cell in the unstructured grid.

    Parameters
    ----------
    area : np.ndarray
        Element area
    depth : np.ndarray
        Static water depth
    surface_elevation : np.ndarray
        Time-varying surface elevation
    thickness : np.ndarray
        Level (i.e. between layer) position (range 0-1). In FVCOM, this is siglev.
    depth_intergrated : bool, optional
        Set to True to return the depth-integrated volume in addition to the depth-resolved volume. Defaults to False.

    Returns
    -------
    depth_volume : np.ndarray
        Depth-resolved volume of all the elements with time.
    volume : np.ndarray, optional
        Depth-integrated volume of all the elements with time.

    """

    # Convert thickness to actual thickness rather than position in water column of the layer.
    dz = np.abs(np.diff(thickness, axis=0))
    volume = (area * (surface_elevation + depth))
    depth_volume = volume[:, np.newaxis, :] * dz[np.newaxis, ...]

    if depth_integrated:
        return depth_volume, volume
    else:
        return depth_volume


def unstructured_grid_depths(h, zeta, sigma, nan_invalid=False):
    """
    Calculate the depth time series for cells in an unstructured grid.

    Parameters
    ----------
    h : np.ndarray
        Water depth
    zeta : np.ndarray
        Surface elevation time series
    sigma : np.ndarray
        Sigma level layer thickness, range 0-1 (siglev or siglay)

    Returns
    -------
    allDepths : np.ndarray
        Time series of model depths.
    """

    if nan_invalid:
        invalid = -zeta > h
        zeta[invalid] = np.NAN

    abs_water_depth = zeta + h
    allDepths = abs_water_depth[:, np.newaxis,:] * sigma[np.newaxis, :,:] + zeta[:, np.newaxis, :]

    return allDepths


def elems2nodes(elems, tri, nvert=None):
    """
    Calculate a nodal value based on the average value for the elements
    of which it a part. This necessarily involves an average, so the
    conversion from nodes2elems and elems2nodes is not reversible.

    Parameters
    ----------
    elems : ndarray
        Array of unstructured grid element values to move to the element
        nodes.
    tri : ndarray
        Array of shape (nelem, 3) comprising the list of connectivity
        for each element.
    nvert : int, optional
        Number of nodes (vertices) in the unstructured grid.

    Returns
    -------
    nodes : ndarray
        Array of values at the grid nodes.

    """

    if not nvert:
        nvert = np.max(tri) + 1
    count = np.zeros(nvert, dtype=int)

    # Deal with 1D and 2D element arrays separately
    if np.ndim(elems) == 1:
        nodes = np.zeros(nvert)
        for i, indices in enumerate(tri):
            n0, n1, n2 = indices
            nodes[n0] = nodes[n0] + elems[i]
            nodes[n1] = nodes[n1] + elems[i]
            nodes[n2] = nodes[n2] + elems[i]
            count[n0] = count[n0] + 1
            count[n1] = count[n1] + 1
            count[n2] = count[n2] + 1

    elif np.ndim(elems) > 1:
        # Horrible hack alert to get the output array shape for multiple
        # dimensions.
        nodes = np.zeros((list(np.shape(elems)[:-1]) + [nvert]))
        for i, indices in enumerate(tri):
            n0, n1, n2 = indices
            nodes[..., n0] = nodes[..., n0] + elems[..., i]
            nodes[..., n1] = nodes[..., n1] + elems[..., i]
            nodes[..., n2] = nodes[..., n2] + elems[..., i]
            count[n0] = count[n0] + 1
            count[n1] = count[n1] + 1
            count[n2] = count[n2] + 1

    # Now calculate the average for each node based on the number of
    # elements of which it is a part.
    nodes /= count

    return nodes


def nodes2elems(nodes, tri):
    """
    Calculate a element centre value based on the average value for the
    nodes from which it is formed. This necessarily involves an average,
    so the conversion from nodes2elems and elems2nodes is not
    necessarily reversible.

    Parameters
    ----------
    nodes : ndarray
        Array of unstructured grid node values to move to the element
        centres.
    tri : ndarray
        Array of shape (nelem, 3) comprising the list of connectivity
        for each element.

    Returns
    -------
    elems : ndarray
        Array of values at the grid nodes.

    """

    if np.ndim(nodes) == 1:
        elems = nodes[tri].mean(axis=-1)
    elif np.ndim(nodes) == 2:
        elems = nodes[..., tri].mean(axis=-1)
    else:
        raise Exception('Too many dimensions (maximum of two)')

    return elems


def vincenty_distance(point1, point2, miles=False):
    """
    Author Maurycy Pietrzak (https://github.com/maurycyp/vincenty)

    Vincenty's formula (inverse method) to calculate the distance (in
    kilometers or miles) between two points on the surface of a spheroid

    Parameters
    ----------
    point1 : list, tuple, np.ndarray
        Decimal degree longitude and latitude for the start.
    point2 : list, tuple, np.ndarray
        Decimal degree longitude and latitude for the end.
    miles : bool
        Set to True to return the distance in miles. Defaults to False (kilometres).

    Returns
    -------
    distance : float
        Distance between point1 and point2 in kilometres.

    """

    a = 6378137  # meters
    f = 1 / 298.257223563
    b = 6356752.314245  # meters; b = (1 - f)a

    MILES_PER_KILOMETER = 0.621371

    MAX_ITERATIONS = 200
    CONVERGENCE_THRESHOLD = 1e-12  # .000,000,000,001


    # short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0

    U1 = math.atan((1 - f) * math.tan(math.radians(point1[0])))
    U2 = math.atan((1 - f) * math.tan(math.radians(point2[0])))
    L = math.radians(point2[1] - point1[1])
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in range(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return None  # failure to converge

    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                 (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                 (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s = b * A * (sigma - deltaSigma)

    s /= 1000  # meters to kilometers
    if miles:
        s *= MILES_PER_KILOMETER  # kilometers to miles

    return round(s, 6)


def haversine_distance(point1, point2, miles=False):
    """
    Haversine function to calculate first order distance measurement. Assumes
    spherical Earth surface. Converted from MATLAB function:

    http://www.mathworks.com/matlabcentral/fileexchange/27785

    Parameters
    ----------
    point1 : list, tuple, np.ndarray
        Decimal degree longitude and latitude for the start.
    point2 : list, tuple, np.ndarray
        Decimal degree longitude and latitude for the end.
    miles : bool
        Set to True to return the distance in miles. Defaults to False (kilometres).

    Returns
    -------
    distance : np.ndarray
        Distance between point1 and point2 in kilometres.

    """

    # Convert all decimal degree inputs to radians.
    point1 = np.deg2rad(point1)
    point2 = np.deg2rad(point2)

    R = 6371                           # Earth's mean radius in kilometres
    delta_lat = point2[1] - point1[1]  # difference in latitude
    delta_lon = point2[0] - point1[0]  # difference in longitude
    print(delta_lon, delta_lat)
    # Magic follows
    a = np.sin(delta_lat / 2)**2 + np.cos(point1[1]) * np.cos(point2[1]) * np.sin(delta_lon / 2)**2
    print(a)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    print(c)
    distance = R * c                   # distance in kilometres

    if miles:
        distance *= 0.621371

    return distance


def shape_coefficients(xc, yc, nbe, isbce):
    """
    This function is used to calculate the coefficients for a linear function on the x-y plane:
        r(x, y; phai) = phai_c + cofa1 * x + cofa2 * y

    Unlike the FVCOM implementation, this returns NaNs for boundary elements (FVCOM returns zeros).

    Parameters
    ----------
    xc, yc : np.ndarray, list-like
        Element centre coordinates (cartesian coordinates)
    nbe : np.ndarray
        The three element IDs which surround each element.
    isbce : np.ndarray
        Flag for whether a given element is on the grid boundary.

    Returns
    -------
    a1u : np.ndarray
        Interpolant in the x direction.
    a2u : np.ndarray
        Interpolant in the y direction.

    Notes
    -----
    This is a more or less direct translation of the FVCOM subroutine SHAPE_COEFFICIENTS_GCN from shape_coef_gcn.F.
    There is probably a lot of optimisation to be done, but it seems to run in a not totally unreasonable amount of
    time as is, so I'm leaving it for now.

    """

    a1u = np.empty((len(xc), 4))
    a2u = np.empty((len(xc), 4))
    a1u[:] = np.nan
    a2u[:] = np.nan
    non_boundary_indices = np.arange(len(xc))[~isbce]
    for i in non_boundary_indices:
        y1 = (yc[nbe[i, 0]] - yc[i]) / 1000
        y2 = (yc[nbe[i, 1]] - yc[i]) / 1000
        y3 = (yc[nbe[i, 2]] - yc[i]) / 1000
        x1 = (xc[nbe[i, 0]] - xc[i]) / 1000
        x2 = (xc[nbe[i, 1]] - xc[i]) / 1000
        x3 = (xc[nbe[i, 2]] - xc[i]) / 1000

        delt = ((x1 * y2 - x2 * y1)**2 + (x1 * y3 - x3 * y1)**2 + (x2 * y3 - x3 * y2)**2) * 1000

        a1u[i, 0] = (y1 + y2 + y3) * (x1 * y1 + x2 * y2+x3 * y3)- (x1 + x2 + x3) * (y1**2 + y2**2 + y3**2)
        a1u[i, 0] = a1u[i, 0] / delt
        a1u[i, 1] = (y1**2 + y2**2 + y3**2) * x1 - (x1 * y1 + x2 * y2 + x3 * y3) * y1
        a1u[i, 1] = a1u[i, 1] / delt
        a1u[i, 2] = (y1**2 + y2**2 + y3**2) * x2 - (x1 * y1 + x2 * y2 + x3 * y3) * y2
        a1u[i, 2] = a1u[i, 2] / delt
        a1u[i, 3] = (y1**2 + y2**2 + y3**2) * x3 - (x1 * y1 + x2 * y2 + x3 * y3) * y3
        a1u[i, 3] = a1u[i, 3] / delt

        a2u[i, 0] = (x1 + x2 + x3) * (x1 * y1 + x2 * y2+x3 * y3)- (y1 + y2 + y3) * (x1**2 + x2**2 + x3**2)
        a2u[i, 0] = a2u[i, 0] / delt
        a2u[i, 1] = (x1**2 + x2**2 + x3**2) * y1 - (x1 * y1 + x2 * y2 + x3 * y3) * x1
        a2u[i, 1] = a2u[i, 1] / delt
        a2u[i, 2] = (x1**2 + x2**2 + x3**2) * y2 - (x1 * y1 + x2 * y2 + x3 * y3) * x2
        a2u[i, 2] = a2u[i, 2] / delt
        a2u[i, 3] = (x1**2 + x2**2 + x3**2) * y3 - (x1 * y1 + x2 * y2 + x3 * y3) * x3
        a2u[i, 3] = a2u[i, 3] / delt

    # Return transposed arrays to match what gets read in from a netCDF file.
    return a1u.T, a2u.T


def reduce_triangulation(tri, nodes):
    """
    Returns the triangulation for a subset of grid nodes.

    Parameters
    ----------
    tri : np.ndarray Nx3
        Grid triangulation (e.g. triangle as returned from read_fvcom_mesh)
    nodes : np.ndarray M
        Selected subset of nodes for re-triangulating

    Returns
    -------
    reduced_tri : np.ndarray Mx3
        Triangulation for just the nodes listed in nodes

    Notes
    -----
    Assumes the nodes selected are a contiguous part of the grid.

    """

    reduced_tri = tri[np.all(np.isin(tri, nodes), axis=1), :]

    # remap nodes to a new index
    new_index = np.arange(0, len(nodes))
    for this_old, this_new in zip(nodes, new_index):
        reduced_tri[reduced_tri == this_old] = this_new

    return reduced_tri


def getcrossectiontriangles(cross_section_pnts, trinodes, X, Y, dist_res):
    """
    Subsamples the line defined by cross_section_pnts at the resolution dist_res on the grid defined by
    the triangulation trinodes, X, Y. Returns the location of the sub sampled points (sub_samp), which
    triangle they are in (sample_cells) and their nearest nodes (sample_nodes).


    Parameters
    ----------
    cross_section_pnts : 2x2 list_like
        The two ends of the cross section line.

    trinodes : list-like
        Unstructured grid triangulation table

    X,Y : list-like
        Node positions

    dist_res : float
        Approximate distance at which to sample the line

    Returns
    -------
    sub_samp : 2xN list
        Positions of sample points

    sample_cells : N list
        The cells within which the subsample points fall. -1 indicates that the point is outside the grid.

    sample_nodes : N list
        The nodes nearest the subsample points. -1 indicates that the point is outside the grid.


    Example
    -------


    TO DO
    -----

    Messy code. There definitely should be a more elegant version of this...
    Set up example and tests.

    """
    cross_section_x = [cross_section_pnts[0][0], cross_section_pnts[1][0]]
    cross_section_y = [cross_section_pnts[0][1], cross_section_pnts[1][1]]

    cross_section_dist = np.sqrt((cross_section_x[1] - cross_section_x[0])**2 + (cross_section_y[1] - cross_section_y[0])**2)
    res = np.ceil(cross_section_dist/dist_res)

    # first reduce the number of points to consider by only including triangles which cross the line through the two points
    tri_X = X[trinodes]
    tri_Y = Y[trinodes]

    tri_cross_log_1_1 = np.logical_or(np.logical_and(tri_X.min(1) < min(cross_section_x), tri_X.max(1) > max(cross_section_x)),
                            np.logical_and(tri_Y.min(1) < min(cross_section_y), tri_Y.max(1) > max(cross_section_y)))

    tri_cross_log_1_2 = np.any(np.logical_and(np.logical_and(tri_X < max(cross_section_x), tri_X > min(cross_section_x)), np.logical_and(tri_Y < max(cross_section_y), tri_Y > min(cross_section_y))), axis = 1)
    tri_cross_log_1 = np.logical_or(tri_cross_log_1_1, tri_cross_log_1_2)

    tri_cross_log_1_2 = np.any(np.logical_and(np.logical_and(tri_X < max(cross_section_x), tri_X > min(cross_section_x)), np.logical_and(tri_Y < max(cross_section_y), tri_Y > min(cross_section_y))), axis = 1)
    tri_cross_log_1 = np.logical_or(tri_cross_log_1_1, tri_cross_log_1_2)

    # and add a buffer of one attached triangle
    tri_cross_log_1 = np.any(np.isin(trinodes, np.unique(trinodes[tri_cross_log_1,:])), axis=1)

    # and reduce further by requiring every node to be within 1 line length + 10%
    line_len = np.sqrt((cross_section_x[0] - cross_section_x[1])**2 + (cross_section_y[0] - cross_section_y[1])**2)
    line_len_plus = line_len * 1.1

    tri_dist_1 = np.sqrt((tri_X - cross_section_x[0])**2 + (tri_Y - cross_section_y[0])**2)
    tri_dist_2 = np.sqrt((tri_X - cross_section_x[1])**2 + (tri_Y - cross_section_y[1])**2)

    tri_cross_log_2 = np.logical_and(tri_dist_1.min(1) < line_len_plus, tri_dist_2.min(1) < line_len_plus)
    tri_cross_log = np.logical_and(tri_cross_log_1, tri_cross_log_2)

    # but as a fall back for short lines add back in triangles within a threshold of 100m
    tri_cross_log_3 = np.logical_or(tri_dist_1.min(1) < 100, tri_dist_2.min(1) < 100)
    tri_cross_log = np.logical_or(tri_cross_log, tri_cross_log_3)

    # and add a buffer of one attached triangle
    tri_cross_log_1 = np.any(np.isin(trinodes, np.unique(trinodes[tri_cross_log,:])), axis=1)
    tri_cross_log = np.logical_or(tri_cross_log, tri_cross_log_1)


    # then subsample the line at a given resolution and find which triangle each sample falls in (if at all)
    sub_samp = np.asarray([np.linspace(cross_section_x[0], cross_section_x[1], res), np.linspace(cross_section_y[0], cross_section_y[1], res)]).T
    red_tri_list_ind = np.arange(0, len(trinodes))[tri_cross_log]
    sample_cells = np.zeros(len(sub_samp))

    for this_ind, this_point in enumerate(sub_samp):
        in_this_tri = False
        this_tri_ind = 0
        while in_this_tri is False:
            this_tri = red_tri_list_ind[this_tri_ind]
            is_in = isintriangle(tri_X[this_tri,:], tri_Y[this_tri,:], this_point[0], this_point[1])

            if is_in:
                sample_cells[this_ind] = this_tri
                in_this_tri = True
            elif this_tri_ind == len(red_tri_list_ind)-1:
                sample_cells[this_ind] = -1
                in_this_tri = True
            else:
                this_tri_ind +=1

    # for node properties now need the weight the nearest nodes to the sample point
    sample_nodes = np.zeros(len(sub_samp))
    red_node_ind = np.unique(trinodes[red_tri_list_ind,:])

    for this_ind, this_point in enumerate(sub_samp):
        if sample_cells[this_ind] == -1:
            sample_nodes[this_ind] = -1
        else:
            all_dist = np.sqrt((X[red_node_ind] - this_point[0])**2 + (Y[red_node_ind] - this_point[1])**2)
            sample_nodes[this_ind] = red_node_ind[np.where(all_dist==all_dist.min())[0][0]]

    return sub_samp, sample_cells, sample_nodes


def isintriangle(tri_x, tri_y, point_x, point_y):
    # Returns a boolean as to whether the point (point_x, point_y) is within the triangle (tri_x, tri_y)
    # method from http://totologic.blogspot.co.uk/2014/01/accurate-point-in-triangle-test.html without edge test
    # used in getcrossectiontriangles
    x1 = tri_x[0]
    x2 = tri_x[1]
    x3 = tri_x[2]

    y1 = tri_y[0]
    y2 = tri_y[1]
    y3 = tri_y[2]

    a = ((y2 - y3)*(point_x - x3) + (x3 - x2)*(point_y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    b = ((y3 - y1)*(point_x - x3) + (x1 - x3)*(point_y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    c = 1 - a - b

    is_in =  0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1

    return is_in


# For backwards compatibility.
def parseUnstructuredGridSMS(*args, **kwargs):
    warn('{} is deprecated. Use read_sms_mesh instead.'.format(inspect.stack()[0][3]))
    return read_sms_mesh(*args, **kwargs)


def parseUnstructuredGridFVCOM(*args, **kwargs):
    warn('{} is deprecated. Use read_fvcom_mesh instead.'.format(inspect.stack()[0][3]))
    return read_fvcom_mesh(*args, **kwargs)


def parseUnstructuredGridMIKE(*args, **kwargs):
    warn('{} is deprecated. Use read_mike_mesh instead.'.format(inspect.stack()[0][3]))
    return read_mike_mesh(*args, **kwargs)


def parseUnstructuredGridGMSH(*args, **kwargs):
    warn('{} is deprecated. Use read_gmsh_mesh instead.'.format(inspect.stack()[0][3]))
    return read_gmsh_mesh(*args, **kwargs)


def writeUnstructuredGridSMS(*args, **kwargs):
    warn('{} is deprecated. Use write_sms_mesh instead.'.format(inspect.stack()[0][3]))
    return write_sms_mesh(*args, **kwargs)


def writeUnstructuredGridSMSBathy(*args, **kwargs):
    warn('{} is deprecated. Use write_sms_bathy instead.'.format(inspect.stack()[0][3]))
    return write_sms_bathy(*args, **kwargs)


def writeUnstructuredGridMIKE(*args, **kwargs):
    warn('{} is deprecated. Use write_mike_mesh instead.'.format(inspect.stack()[0][3]))
    return write_mike_mesh(*args, **kwargs)


def findNearestPoint(*args, **kwargs):
    warn('{} is deprecated. Use find_nearest_point instead.'.format(inspect.stack()[0][3]))
    return find_nearest_point(*args, **kwargs)


def elementSideLengths(*args, **kwargs):
    warn('{} is deprecated. Use element_side_lengths instead.'.format(inspect.stack()[0][3]))
    return element_side_lengths(*args, **kwargs)


def fixCoordinates(*args, **kwargs):
    warn('{} is deprecated. Use fix_coordinates instead.'.format(inspect.stack()[0][3]))
    return fix_coordinates(*args, **kwargs)


def clipTri(*args, **kwargs):
    warn('{} is deprecated. Use clip_triangulation instead.'.format(inspect.stack()[0][3]))
    return clip_triangulation(*args, **kwargs)


def getRiverConfig(*args, **kwargs):
    warn('{} is deprecated. Use get_river_config instead.'.format(inspect.stack()[0][3]))
    return get_river_config(*args, **kwargs)


def getRivers(*args, **kwargs):
    warn('{} is deprecated. Use get_rivers instead.'.format(inspect.stack()[0][3]))
    return get_rivers(*args, **kwargs)


def lineSample(*args, **kwargs):
    warn('{} is deprecated. Use line_sample instead.'.format(inspect.stack()[0][3]))
    return line_sample(*args, **kwargs)


def clipDomain(*args, **kwargs):
    warn('{} is deprecated. Use clip_domain instead.'.format(inspect.stack()[0][3]))
    return clip_domain(*args, **kwargs)


def OSGB36toWGS84(*args, **kwargs):
    warn('{} is deprecated. Use OSGB36_to_WGS84 instead.'.format(inspect.stack()[0][3]))
    return OSGB36_to_WGS84(*args, **kwargs)


def UTMtoLL(*args, **kwargs):
    warn('{} is deprecated. Use UTM_to_LL instead.'.format(inspect.stack()[0][3]))
    return UTM_to_LL(*args, **kwargs)


def heron(*args):
    warn('{} is deprecated. Use get_area instead.'.format(inspect.stack()[0][3]))
    return get_area(*args)


def surrounders(*args):
    warn('{} is deprecated. Use find_connected_nodes instead.'.format(inspect.stack()[0][3]))
    return find_connected_nodes(*args)
