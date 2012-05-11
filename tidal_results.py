#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plot_unstruct_grid as gp
from readFVCOM import readFVCOM


def extractTideElevation(data, elementIdx):
    """
    Extract the tidal elevation only (no water depth) from an element (or
    range of elements) and return as arrays for plotting or further analysis.

    Give input data as 2D array (location and time). Specify the indices to
    extract the tidal elevation.

    Specify the elements as an array (even if there's only one).

    """

    try:
        import numpy as np
    except ImportError:
        print 'NumPy not found'


    numSteps, numElem = np.shape(data)
    numStations = np.shape(elementIdx)[0]
    output = np.zeros((numStations, numSteps)) * np.nan

    # Extract the time series for each element requested
    output = data[:,elementIdx]

    return output


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

    # Number of subplots
    numPlots = 5

    dt = (FVCOM['time'][1]-FVCOM['time'][0])*60*60*24

    samplingIdx = [174, 259, 3] # start, end, skip
    positionIdx = np.arange(samplingIdx[0], samplingIdx[1], samplingIdx[2])
    skippedIdx = np.arange(samplingIdx[0], samplingIdx[1], samplingIdx[2] * numPlots)

    try:
        Z = FVCOM['zeta']
    except:
        print 'Did not find tidal elevation (zeta) in model output'

    tidalHeights = extractTideElevation(Z, positionIdx)

    t = FVCOM['time']-np.min(FVCOM['time']) # start time at zero

    plt.figure(1)
    plt.clf()
    for i in xrange(numPlots):
        plt.subplot(numPlots,1,i)
        colourIdx = int((i/float(numPlots))*255)
        plt.plot(t, Z[:, skippedIdx[i]], '-x', label='Station 0', color=cm.rainbow(colourIdx))
        plt.axis('auto')


    # Plot figure of the locations of the extracted points
    gp.plotUnstructuredGrid(triangles, nodes, x, y, FVCOM['zeta'][-1,:], 'Tidal elevation (m)')
    plt.plot(x[positionIdx], y[positionIdx], '.')
    for i in positionIdx:
        plt.text(x[i], y[i], str(i),
            horizontalalignment='center', verticalalignment='center', size=8)

    for i in skippedIdx:
        plt.text(x[i], y[i], str(i),
            horizontalalignment='center', verticalalignment='center', size=18)
        


