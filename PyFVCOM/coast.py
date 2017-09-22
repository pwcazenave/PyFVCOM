"""
Convert from a range of input coastline file types to a CST file compatible
with SMS.

CST (FVCOM) format is a text file structured as:

  COAST
  nArc
  n z
  x1 y1 z0
  ...
  xn yn zn
  x1 y1 z0

where nArc is the total number of arcs, n is the number of nodes in the arc and
z is a z value (typically zero). The polygon must close with the first value at
the end.


"""


import inspect
import shapefile

import numpy as np

from warnings import warn


def read_ESRI_shapefile(file, fileOut):
    """
    Convert ESRI ShapeFiles to SMS-compatible CST files.

    Parameters
    ----------
    file : str
        Full path to the ESRI ShapeFile to convert.
    fileOut : str
        Full path to the output file.

    """

    sf = shapefile.Reader(file)
    shapes = sf.shapes()

    nArcs = sf.numRecords

    # Set up the output file
    fileWrite = open(fileOut, 'w')
    fileWrite.write('COAST\n')
    fileWrite.write('{}\n'.format(int(nArcs)))

    z = 0

    for arc in range(nArcs):
        # Write the current arc out to file. Start with number of nodes and z
        arcLength = len(shapes[arc].points)
        fileWrite.write('{}\t{}\n'.format(arcLength, float(z)))
        # Add the actual arc
        for arcPos in shapes[arc].points:
            fileWrite.write('\t{}\t{}\t{}\n'.format(
                float(arcPos[0]),
                float(arcPos[1]),
                float(z)))

    fileWrite.close()


def read_arc_MIKE(file, fileOut):
    """
    Read in a set of MIKE arc files and export to CST format compatible with
    FVCOM.

    MIKE format is:

        x, y, position, z(?), ID

    where position is 1 = along arc and 0 = end of arc.

    In the CST format, the depth is typically zero, but we'll read it from the
    MIKE z value and add it to the output file nevertheless. For the
    conversion, we don't need the ID, so we can ignore that.

    Parameters
    ----------
    file : str
        Full path to the DHI MIKE21 arc files.
    fileOut : str
        Full path to the output file.

    """
    fileRead = open(file, 'r')
    lines = fileRead.readlines()
    fileRead.close()

    fileWrite = open(fileOut, 'w')
    # Add the easy header
    fileWrite.write('COAST\n')
    # This isn't pretty, but assuming you're coastline isn't millions of
    # points, it should be ok...
    nArcs = 0
    for line in lines:
        x, y, pos, z, ID = line.strip().split(' ')
        if int(pos) == 0:
            nArcs += 1

    fileWrite.write('{}\n'.format(int(nArcs)))

    arc = []
    n = 1

    for line in lines:

        x, y, pos, z, ID = line.strip().split(' ')
        if int(pos) == 1:
            arc.append([x, y])
            n += 1
        elif int(pos) == 0:
            arc.append([x, y])
            # We're at the end of an arc, so write out to file. Start with
            # number of nodes and z
            fileWrite.write('{}\t{}\n'.format(int(n), float(z)))
            for arcPos in arc:
                fileWrite.write('\t{}\t{}\t{}\n'.format(
                    float(arcPos[0]),
                    float(arcPos[1]),
                    float(z)))
            # Reset n and arc for new arc
            n = 1
            arc = []

    fileWrite.close()


def read_CST(cst):
    """
    Read a CST file and store the vertices in a dict.

    Parameters
    ----------
    cst : str
        Path to the CST file to load in.

    Returns
    -------
    vert : dict
        Dictionary with the coordinates of the vertices of the arcs defined in
        the CST file.

    """

    f = open(cst, 'r')
    lines = f.readlines()
    f.close()

    vert = {}
    c = 0
    for line in lines:
        line = line.strip()
        if line.startswith('COAST'):
            pass
        else:
            # Split the line on tabs and work based on that output.
            line = line.split('\t')
            if len(line) == 1:
                # Number of arcs. We don't especially need to know this.
                pass

            elif len(line) == 2:
                # Number of nodes within a single arc. Store the current index
                # and use as the key for the dict.
                nv = int(line[0])
                id = str(c)    # dict key
                vert[id] = []  # initialise the vert list
                c += 1         # arc counter

            elif len(line) == 3:
                coords = [float(x) for x in line[:-1]]
                # Skip the last position if we've already got some data in the
                # dict for this arc.
                if vert[id]:
                    if len(vert[id]) != nv - 1:
                        vert[id].append(coords)
                    else:
                        # We're at the end of this arc, so convert the
                        # coordinates we've got to a numpy array for easier
                        # handling later on.
                        vert[id] = np.asarray(vert[id])
                else:
                    vert[id].append(coords)

    return vert


def write_CST(obc, file, sort=False):
    """
    Read a CST file and store the vertices in a dict.

    Parameters
    ----------
    obc : dict
        Dict with each entry as a NumPy array of coordinates (x, y).
    file : str
        Path to the CST file to which to write (overwrites existing files).
    sort : bool, optional
        Optionally sort the output coordinates (by x then y). This might break
        things with complicated open boundary geometries.

    """
    nb = len(obc)

    with open(file, 'w') as f:
        # Header
        f.write('COAST\n')
        f.write('{:d}\n'.format(nb))

        for _, bb in obc.iteritems():  # each boundary
            nn = len(bb)

            # The current arc's header
            f.write('{:d}\t0.0\n'.format(nn))

            if sort:
                idx = np.lexsort(bb.transpose())
                bb = bb[idx, :]

            for xy in bb:
                f.write('\t{:.6f}\t{:.6f}\t0.0\n'.format(xy[0], xy[1]))

        f.close


def readESRIShapeFile(*args, **kwargs):
    warn('{} is deprecated. Use read_ESRI_shapefile instead.'.format(inspect.stack()[0][3]))
    return read_ESRI_shapefile(*args, **kwargs)


def readArcMIKE(*args, **kwargs):
    warn('{} is deprecated. Use read_arc_MIKE instead.'.format(inspect.stack()[0][3]))
    return read_arc_MIKE(*args, **kwargs)


def readCST(*args, **kwargs):
    warn('{} is deprecated. Use read_CST instead.'.format(inspect.stack()[0][3]))
    return read_CST(*args, **kwargs)


def writeCST(*args, **kwargs):
    warn('{} is deprecated. Use write_CST instead.'.format(inspect.stack()[0][3]))
    return write_CST(*args, **kwargs)


