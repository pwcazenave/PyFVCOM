"""
Convert from a range of input coastline file types to a CST file compatible with SMS.

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


def readESRIShapeFile(file, fileOut):
    """ Convert ESRI ShapeFiles to SMS-compatible CST files. """

    try:
        import shapefile
    except ImportError:
        print('Module shapefile not found (and required). Try sudo yum install pyshape.')

    sf = shapefile.Reader(file)
    shapes = sf.shapes()

    nArcs = sf.numRecords

    # Set up the output file
    fileWrite = open(fileOut, 'w')
    fileWrite.write('COAST\n')
    fileWrite.write('{}\n'.format(int(nArcs)))

    z = 0

    for arc in xrange(nArcs):
        # Write the current arc out to file. Start with number of nodes and z
        arcLength = len(shapes[arc].points)
        fileWrite.write('{}\t{}\n'.format(arcLength, float(z)))
        # Add the actual arc
        for arcPos in shapes[arc].points:
            fileWrite.write('\t{}\t{}\t{}\n'.format(float(arcPos[0]), float(arcPos[1]), float(z)))

    fileWrite.close()



def readArcMIKE(file, fileOut):
    """
    Read in a set of MIKE arc files and export to CST format compatible with
    FVCOM.

    MIKE format is:

        x, y, position, z(?), ID

    where position is 1 = along arc and 0 = end of arc.

    In the CST format, the depth is typically zero, but we'll read it from the
    MIKE z value and add it to the output file nevertheless. For the
    conversion, we don't need the ID, so we can ignore that.

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
                fileWrite.write('\t{}\t{}\t{}\n'.format(float(arcPos[0]), float(arcPos[1]), float(z)))
            # Reset n and arc for new arc
            n = 1
            arc = []

    fileWrite.close()


if __name__ == '__main__':

    from os import path
    from sys import argv

    if len(argv[1:]) == 0:
        # We don't have a supplied file
        #infile = '../data/test.xyz'
        #infile = '../data/ukerc_shelf/ukerc/mike/shelf_coast.xyz'
        #infile = '../data/ukerc_shelf/ukerc/mike/shelf_coast_utm.xyz'
        infile = '../../data/GSHHS/modelling/gshhs_shelf_utm30n.xyz'
        #infile = '../../../Remote/Mike/desktop/mesh/data/coast/synthetic/kinked_boundary_0.001_utm30n.xyz'
        base, ext = path.splitext(infile)

        readArcMIKE(infile, base + '.cst')

    else:
        # Run through the files supplied on the command line
        for file in argv[1:]:
            infile = file
            base, ext = path.splitext(infile)

            readArcMIKE(infile, base + '.cst')


