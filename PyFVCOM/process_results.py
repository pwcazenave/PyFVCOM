"""
Series of tools to calculate various parameters from an FVCOM model output
NetCDF file.

"""

import inspect

import numpy as np
import matplotlib.pyplot as plt

from warnings import warn

def calculate_total_CO2(FVCOM, varPlot, startIdx, layerIdx, leakIdx, dt, noisy=False):
    """
    Calculates total CO2 input and plots accordingly. Nothing too fancy.

    Give a NetCDF object as the first input (i.e. the output of readFVCOM()).
    The variable of interest is defined as a string in varPlot and the
    summation begins at startIdx. The total is calculated at that layer
    layerIdx (you probably want this to be zero) and at the point leakIdx.

    Plot is of the input at leakIdx for the duration of FVCOM['time'].

    Optionally specify noisy as True to get more verbose output.

    FIXME(pica) This doesn't work as is.

    """

    Z = FVCOM[varPlot]

    TCO2 = np.zeros(FVCOM['time'].shape)

    for i in range(startIdx, Z.shape[0]):
        if i > 0:
            if len(np.shape(Z)) == 3:
                TCO2[i] = TCO2[i-1] + (Z[i, layerIdx, leakIdx].squeeze() * dt)
            else:
                TCO2[i] = TCO2[i-1] + (Z[i, leakIdx].squeeze() * dt)

    # Scale to daily input. Input rate begins two days into model run
    nDays = FVCOM['time'].max()-FVCOM['time'].min()-2
    TCO2Scaled = TCO2/nDays

    # Get the total CO2 in the system at the end of the simulation
    totalCO2inSystem = np.sum(Z[np.isfinite(Z)])  # skip NaNs

    # Some results
    if noisy:
        print("Leak:\t\t{}\nLayer:\t\t{}\nStart:\t\t{}".format(leakIdx, layerIdx, startIdx))
        print("Total input per day:\t\t{:.2f}".format(TCO2Scaled[-1]))
        print("Total in the system:\t\t{:.2f}".format(totalCO2inSystem))
        print("Total in the system per day:\t{:.2f}".format(totalCO2inSystem / nDays))

    # Make a pretty picture
    # plt.figure(100)
    # plt.clf()
    # #plt.plot(FVCOM['time'],TCO2,'r-x')
    # plt.plot(xrange(Z.shape[0]),np.squeeze(Z[:,layerIdx,leakIdx]),'r-x')
    # plt.xlabel('Time')
    # plt.ylabel(varPlot + ' input')
    # plt.show()

    return totalCO2inSystem


def calculate_CO2_leak_budget(FVCOM, leakIdx, startDay):
    """
    Replicate Riqui's CO2leak_budget.m code.

    Calculates the total CO2 input in the system over a 24 hour period. Code
    automatically calculates output file timestep in hours to obtain a 24 hour
    stretch.

    Specify the first argument as the output of readFVCOM(), the second as the
    leak point in the grid and the third argument as the leak start.

    FIXME(pica) Not yet working (and probably doesn't match Riqui's code...)

    """

    # Get output file sampling in hours
    dt = int(round((FVCOM['time'][1] - FVCOM['time'][0]) * 24, 1))
    # Calculate number of steps required to get a day's worth of results
    timeSteps = np.r_[0:(24/dt)+1]+startDay

    # Preallocate the output arrays
    CO2 = np.ones(len(timeSteps))*np.nan
    CO2Leak = np.ones(np.shape(CO2))*np.nan

    for i, tt in enumerate(timeSteps):
        dump = FVCOM['h'] + FVCOM['zeta'][tt, :]
        dz = np.abs(np.diff(FVCOM['siglev'], axis=0))
        data = FVCOM['DYE'][tt, :, :]*dz
        data = np.sum(data, axis=0)
        CO2[i] = np.sum(data*FVCOM['art1'] * dump)
        CO2Leak[i] = np.sum(data[leakIdx] * FVCOM['art1'][leakIdx])

    maxCO2 = np.max(CO2)

    return CO2, CO2Leak, maxCO2


def data_average(data, **args):
    """ Depth average a given FVCOM output data set along a specified axis """

    dataMask = np.ma.masked_array(data, np.isnan(data))
    dataMeaned = np.ma.filled(dataMask.mean(**args), fill_value=np.nan).squeeze()

    return dataMeaned


def unstructured_grid_volume(FVCOM):
    """
    Calculate the volume for every cell in the unstructured grid.

    Parameters
    ----------
    FVCOM : dict
        Dict which contains the following keys:
            - art1 - element area
            - h - water depth
            - zeta - surface elevation time series
            - siglev - sigma level layer thickness (range 0-1)

    Returns
    -------
    allVolumes : np.ndarray
        Array with the volumes of all the elements with time.

    """

    elemAreas = FVCOM['art1']
    elemDepths = FVCOM['h']
    elemTides = FVCOM['zeta']
    elemThickness = np.abs(np.diff(FVCOM['siglev'], axis=0))

    # Get volumes for each cell at each time step to include tidal changes
    tt, xx = FVCOM['zeta'].shape  # time, node
    ll = FVCOM['siglev'].shape[0] - 1  # layers = levels - 1
    allVolumes = ((elemDepths
        + np.tile(elemTides, [ll, 1, 1]).transpose(1, 0, 2))
        * np.tile(elemThickness, [tt, 1, 1])) * elemAreas

    return allVolumes


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
def calculateTotalCO2(*args, **kwargs):
    warn('{} is deprecated. Use calculate_total_CO2 instead.'.format(inspect.stack()[0][3]))
    return calculate_total_CO2(*args, **kwargs)


def CO2LeakBudget(*args, **kwargs):
    warn('{} is deprecated. Use calculate_CO2_leak_budget instead.'.format(inspect.stack()[0][3]))
    return calculate_CO2_leak_budget(*args, **kwargs)


def dataAverage(*args, **kwargs):
    warn('{} is deprecated. Use data_average instead.'.format(inspect.stack()[0][3]))
    return data_average(*args, **kwargs)


def unstructuredGridVolume(*args, **kwargs):
    warn('{} is deprecated. Use unstructured_grid_volume instead.'.format(inspect.stack()[0][3]))
    return unstructured_grid_volume(*args, **kwargs)


def residualFlow(*args, **kwargs):
    warn('{} is deprecated. Use residual_flow instead.'.format(inspect.stack()[0][3]))
    return residual_flow(*args, **kwargs)


