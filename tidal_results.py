#!/usr/bin/env python


def extractTideElevation(data, elementIdx, noisy=False):
    """
    Extract the tidal elevation only (no water depth) from an element (or
    range of elements) and return as arrays for plotting or further analysis.

    Give input data as 2D array (location and time). Specify the indices to
    extract the tidal elevation. Optionally give noisy=True to turn on
    information during processing.

    Specify the elements as an array (even if there's only one).

    """

    try:
        import numpy as np
    except ImportError:
        print 'NumPy not found'


    numSteps, numElem = np.shape(data)
    numStations = np.shape(elementIdx)[0]
    output = np.zeros((numStations, numSteps)) * np.nan


if __name__ == '__main__':


    # Be verbose?
    noisy = True

    getVars = ['x', 'y', 'xc', 'yc', 'zeta', 'art1', 'h', 'time', 'TCO2', 'PH', 'DYE', 'siglev']


    base = '/data/medusa/pica/models/FVCOM/runCO2_leak'
    # Coarse
    in1 = base + '/output/rate_ranges/11days/co2_S5_high_run_0001.nc'
    # Coarse grid
    in2 = base + '/input/configs/inputV5/co2_grd.dat'

    # Read in the NetCDF file and the unstructured grid file
    FVCOM = readFVCOM(in1, getVars, noisy)
    [triangles, nodes, x, y, z] = gp.parseUnstructuredGridFVCOM(in2)

    # Set the leak index point (one less than MATLAB)
    if 'V5' in in1:
        leakIdx = 1315
    elif 'V7' in in1:
        leakIdx = 8186
    elif 'S5' in in1:
        leakIdx = 1315
    elif 'S7' in in1:
        leakIdx = 8186
    else:
        # Could be either...
        leakIdx = 1315
        #leakIdx = 8186

    # Sigma layer
    layerIdx = 0

    # Start index for each input file
    #startIdx = 770 # Riqui's
    startIdx = 120 # Mine

    # Skip value for animation (not zero)
    skipIdx = 5

    dt = (FVCOM['time'][1]-FVCOM['time'][0])*60*60*24

