"""
Series of tools to calculate various parameters from an FVCOM model output
NetCDF file.

"""

import inspect

import numpy as np
import matplotlib.pyplot as plt

from warnings import warn


def residual_flow(FVCOM, idxRange=False, checkPlot=False, noisy=False):
    """
    Calculate the residual flow. By default, the calculation will take place
    over the entire duration of FVCOM['Times']. To limit the calculation to a
    specific range, give the index range as idxRange = [0, 100], for the first
    to 100th time step.  Alternatively, specify idxRange as 'daily' or
    'spring-neap' for daily and spring neap cycle residuals.

    Parameters
    ----------
    FVCOM : dict
        Contains the FVCOM model results.
    idxRange : list or str, optional
        If a list, the start and end index for the time series analysis.
        If a string, then must be one of 'daily' or 'spring-neap' to
        clip the time series to a day or a spring-neap cycle.
    checkPlot : int
        Plot a PVD at element checkPlot of the first vertical layer to
        check the code is working properly.
    noisy : bool
        Set to True to enable verbose output.

    Returns
    -------
    uRes : ndarray
        Raw summed velocity u-direction vector component. Useful for PVD
        plots.
    vRes : ndarray
        Raw summed velocity v-direction vector component. Useful for PVD
        plots.
    rDir : ndarray
        Residual direction array for each element centre in the
        unstructured grid.
    rMag : ndarray
        Residual magnitude array for each element centre in the
        unstructured grid.

    Notes
    -----
    Based on my MATLAB do_residual.m function.


    """

    toSecFactor = 24 * 60 * 60

    # Get the output interval (in days)
    dt = FVCOM['time'][2] - FVCOM['time'][1]

    # Some tidal assumptions. This will need to change in areas in which the
    # diurnal tide dominates over the semidiurnal.
    tideCycle = (12.0 + (25.0 / 60)) / 24.0
    # The number of values in the output file which covers a tidal cycle
    tideWindow = np.ceil(tideCycle / dt)

    # Get the number of output time steps which cover the selected period (in
    # idxRange). If it's spring-neap, use 14.4861 days; daily is one day,
    # obviously.

    startIdx = np.ceil(3 / dt)  # start at the third day to skip the warm up period

    if idxRange == 'spring-neap':
        # To the end of the spring-neap cycle
        endIdx = startIdx + tideWindow + np.ceil(14.4861 / dt)
    elif idxRange == 'daily':
        endIdx = startIdx + tideWindow + np.ceil(1 / dt)
    elif idxRange is False:
        startIdx = 0
        endIdx = -1
    else:
        startIdx = idxRange[0]
        endIdx = idxRange[1]

    try:
        # 3D input
        nTimeSteps, nLayers, nElements = np.shape(FVCOM['u'][startIdx:endIdx, :, :])
    except:
        # 2D input
        nTimeSteps, nElements = np.shape(FVCOM['u'][startIdx:endIdx, :])
        nLayers = 1

    tideDuration = ((dt * nTimeSteps) - tideCycle) * toSecFactor

    # Preallocate outputs.
    uRes = np.zeros([nTimeSteps, nLayers, nElements])
    vRes = np.zeros([nTimeSteps, nLayers, nElements])
    uSum = np.empty([nTimeSteps, nLayers, nElements])
    vSum = np.empty([nTimeSteps, nLayers, nElements])
    uStart = np.empty([nLayers, nElements])
    vStart = np.empty([nLayers, nElements])
    uEnd = np.empty([nLayers, nElements])
    vEnd = np.empty([nLayers, nElements])

    for hh in range(nLayers):
        if noisy:
            print('Layer {} of {}'.format(hh + 1, nLayers))

        try:
            # 3D
            uSum[:, hh, :] = np.cumsum(np.squeeze(FVCOM['u'][startIdx:endIdx, hh, :]), axis=0)
            vSum[:, hh, :] = np.cumsum(np.squeeze(FVCOM['v'][startIdx:endIdx, hh, :]), axis=0)
        except:
            # 2D
            uSum[:, hh, :] = np.cumsum(np.squeeze(FVCOM['u'][startIdx:endIdx, :]), axis=0)
            vSum[:, hh, :] = np.cumsum(np.squeeze(FVCOM['v'][startIdx:endIdx, :]), axis=0)

        for ii in range(nTimeSteps):
            # Create progressive vectors for all time steps in the current layer
            if noisy:
                if ii == 0 or np.mod(ii, 99) == 0:
                    print('Create PVD at time step {} of {}'.format(ii + 1, nTimeSteps))

            uRes[ii, hh, :] = uRes[ii, hh, :] + (uSum[ii, hh, :] * (dt * toSecFactor))
            vRes[ii, hh, :] = vRes[ii, hh, :] + (vSum[ii, hh, :] * (dt * toSecFactor))

        uStart[hh, :] = np.mean(np.squeeze(uRes[0:tideWindow, hh, :]), axis=0)
        vStart[hh, :] = np.mean(np.squeeze(vRes[0:tideWindow, hh, :]), axis=0)
        uEnd[hh, :] = np.mean(np.squeeze(uRes[-tideWindow:, hh, :]), axis=0)
        vEnd[hh, :] = np.mean(np.squeeze(vRes[-tideWindow:, hh, :]), axis=0)

    uDiff = uEnd - uStart
    vDiff = vEnd - vStart

    # Calculate direction and magnitude.
    rDir = np.arctan2(uDiff, vDiff) * (180 / np.pi)  # in degrees.
    rMag = np.sqrt(uDiff**2 + vDiff**2) / tideDuration  # in units/s.

    # Plot to check everything's OK
    if checkPlot:
        if noisy:
            print('Plotting element {}'.format(checkPlot - 1))

        elmt = checkPlot - 1
        lyr = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(uRes[:, lyr, elmt], vRes[:, lyr, elmt])
        ax.plot(uRes[0:tideWindow, lyr, elmt], vRes[0:tideWindow, lyr, elmt], 'gx')
        ax.plot(uRes[-tideWindow:, lyr, elmt], vRes[-tideWindow:, lyr, elmt], 'rx')
        ax.plot(uStart[lyr, elmt], vStart[lyr, elmt], 'go')
        ax.plot(uEnd[lyr, elmt], vEnd[lyr, elmt], 'ro')
        ax.plot([uStart[lyr, elmt], uEnd[lyr, elmt]], [vStart[lyr, elmt], vEnd[lyr, elmt]], 'k')
        ax.set_xlabel('Displacement west-east')
        ax.set_ylabel('Displacement north-south')
        ax.set_aspect('equal')
        ax.autoscale(tight=True)
        fig.show()

    return uRes, vRes, rDir, rMag


# For backwards compatibility.
def unstructuredGridVolume(*args, **kwargs):
    warn('{} is deprecated. Use unstructured_grid_volume instead.'.format(inspect.stack()[0][3]))
    return unstructured_grid_volume(*args, **kwargs)


def residualFlow(*args, **kwargs):
    warn('{} is deprecated. Use residual_flow instead.'.format(inspect.stack()[0][3]))
    return residual_flow(*args, **kwargs)


