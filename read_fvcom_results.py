#!/usr/bin/env python

import numpy as np
import plot_unstruct_grid as gp
import matplotlib.pyplot as plt
from readFVCOM import readFVCOM
from sys import argv


def calculateTotalCO2(FVCOM, varPlot, startIdx, layerIdx, leakIdx, dt, noisy=False):
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

    for i in xrange(startIdx, Z.shape[0]):
        if i > 0:
            if len(np.shape(Z)) == 3:
                TCO2[i] = TCO2[i-1] + (Z[i,layerIdx,leakIdx].squeeze() * dt)
            else:
                TCO2[i] = TCO2[i-1] + (Z[i,leakIdx].squeeze() * dt)

            # Maybe a little too noisy...
            #if noisy:
            #    print "Total " + varPlot + ": " + str(TCO2[i]) + "\n\t" + varPlot + ": " + str(Z[i,0,leakIdx].squeeze())

    # Scale to daily input. Input rate begins two days into model run
    nDays = FVCOM['time'].max()-FVCOM['time'].min()-2
    TCO2Scaled = TCO2/nDays

    # Get the total CO2 in the system at the end of the simulation
    totalCO2inSystem = np.sum(Z[np.isfinite(Z)]) # skip NaNs

    # Some results
    if noisy:
        print "Leak:\t\t%i\nLayer:\t\t%i\nStart:\t\t%i" % (leakIdx, layerIdx, startIdx)
        print "Total input per day:\t\t%.2f" % TCO2Scaled[-1]
        print "Total in the system:\t\t%.2f" % totalCO2inSystem
        print "Total in the system per day:\t%.2f" % (totalCO2inSystem/nDays)

    # Make a pretty picture
    #plt.figure(100)
    #plt.clf()
    ##plt.plot(FVCOM['time'],TCO2,'r-x')
    #plt.plot(xrange(Z.shape[0]),np.squeeze(Z[:,layerIdx,leakIdx]),'r-x')
    #plt.xlabel('Time')
    #plt.ylabel(varPlot + ' input')
    #plt.show()

    return totalCO2inSystem


def animateModelOutput(FVCOM, varPlot, startIdx, skipIdx, layerIdx, addVectors=False, noisy=False):
    """
    Animated model output (for use in ipython).

    Give a NetCDF object as the first input (i.e. the output of readFVCOM()).
    Specify the variable of interest as a string (e.g. 'DYE'). This is case
    sensitive. Specify a starting index, a skip index of n to skip n time steps
    in the animation. The layerIdx is either the sigma layer to plot or, if
    negative, means the depth averaged value is calcualted. 

    Optionally add current vectors to the plot which will be colour coded by
    the magnitude.

    Noisy, if True, turns on printing of various bits of potentially
    relevant information to the console.

    """

    Z = FVCOM[varPlot]

    if layerIdx < 0:
        # Depth average the input data
        Z = dataAverage(Z, axis=1)

    plt.figure(200)
    plt.clf()

    # Initialise the plot
    plt.tripcolor(
        FVCOM['x'],
        FVCOM['y'],
        triangles,
        np.zeros(np.shape(FVCOM['x'])),
        shading='interp')
    plt.axes().set_aspect('equal', 'datalim')
    plt.colorbar()
    #plt.clim(6, 8)
    plt.draw()

    # len(FVCOM['time'])+1 so range goes upto the length so that when i-1 is
    # called we get the last time step included in the animation.
    for i in xrange(startIdx, len(FVCOM['time'])+1, skipIdx):
        # Start animation at the beginning of the array or at startIdx-1
        # (i.e. i-2), whichever is larger.
        if i == startIdx:
            getIdx = np.max([startIdx-1, 0])
        else:
            getIdx = i-1

        if len(np.shape(Z)) == 3: # dim1=time, dim2=sigma, dim3=dye
            plotZ = np.squeeze(Z[getIdx,layerIdx,:])
        else: # dim1=time, dim2=dye (depth averaged)
            # Can't do difference here because we've depth averaged
            plotZ = np.squeeze(Z[getIdx,:])

        # Update the plot
        plt.clf()
        plt.tripcolor(FVCOM['x'], FVCOM['y'], triangles, plotZ, shading='interp')
        plt.colorbar()
        #plt.clim(-1.5, 1.5)
        # Add the vectors
        plt.hold('on')
        if addVectors:
            UU = np.squeeze(FVCOM['u'][i,layerIdx,:])
            VV = np.squeeze(FVCOM['v'][i,layerIdx,:])
            CC = np.sqrt(np.power(UU,2)+np.power(VV,2))
            Q = plt.quiver(FVCOM['xc'], FVCOM['yc'], UU, VV, CC, scale=10)
            plt.quiverkey(Q, 0.5, 0.92, 1, r'$1 ms^{-1}$', labelpos='W')
        plt.axes().set_aspect('equal', 'datalim')
        plt.draw()
        plt.show()

        # Some useful output
        if noisy:
            print '%i of %i (date %.2f)' % (i, len(FVCOM['time']), FVCOM['time'][i-1])
            print 'Min: %g Max: %g Range: %g Standard deviation: %g' % (plotZ.min(), plotZ.max(), plotZ.max()-plotZ.min(), plotZ.std())
        else:
            print


def dataAverage(data, **args):
    """ Depth average a given FVCOM output data set along a specified axis """

    dataMask = np.ma.masked_array(data,np.isnan(data))
    dataMeaned = np.ma.filled(dataMask.mean(**args), fill_value=np.nan).squeeze()

    return dataMeaned


def CO2LeakBudget(FVCOM, leakIdx, startDay):
    """
    Replicate Riqui's CO2leak_budget.m code.

    FIXME(pica) Not yet working (and probably doesn't match Riqui's code...)

    """

    timeSteps = np.r_[0:25]+startDay
    CO2 = np.ones(len(timeSteps))*np.nan
    CO2Leak = np.ones(np.shape(CO2))*np.nan

    for i, tt in enumerate(timeSteps):
        dump = FVCOM['h']+FVCOM['zeta'][tt,:]
        dz = np.abs(np.diff(FVCOM['siglev'], axis=0))
        data = FVCOM['DYE'][tt,:,:]*dz
        data = np.sum(data, axis=0)
        CO2[i] = np.sum(data*FVCOM['art1']*dump)
        CO2Leak[i] = np.sum(data[leakIdx]*FVCOM['art1'][leakIdx])

    maxCO2 = np.max(CO2)

    return CO2, CO2Leak, maxCO2


def unstructuredGridVolume(FVCOM):
    """ Calculate the volume for every cell in the unstructured grid """
    elemAreas = FVCOM['art1']
    elemDepths = FVCOM['h']
    elemTides = FVCOM['zeta']
    elemThickness = np.abs(np.diff(FVCOM['siglev'], axis=0))

    # Get volumes for each cell at each time step to include tidal changes
    Z = FVCOM['DYE']
    (tt, ll, xx) = np.shape(Z) # time, layers, node
    allVolumes = np.zeros([tt, ll, xx])*np.nan
    for i in xrange(tt):
        allVolumes[i,:,:] = ((elemDepths + elemTides[i,:]) * elemThickness) * elemAreas

    return allVolumes





if __name__ == '__main__':

    # Be verbose?
    noisy = True

    getVars = ['x', 'y', 'xc', 'yc', 'zeta', 'art1', 'h', 'time', 'TCO2', 'PH', 'DYE', 'siglev']

    # If running as a script:
    #in1 = argv[1]
    #in2 = argv[2]
    #FVCOM = readFVCOM(in1, getVars)
    #[triangles, nodes, x, y, z] = gp.parseUnstructuredGridFVCOM(argv[2])

    # Running from ipython
    # Riqui's
    #base = '/data/medusa/rito/models/FVCOM/runCO2_leak/'
    #in1 = base + '/output/co2_S5.1.1.2_0002.nc'
    #in1 = base + '/output/co2_S5.1.2.1_0002.nc'
    #in1 = base + '/output/co2_V5.1.2.1_0001.nc'
    #in1 = base + '/output/co2_V7.1.1.1_0001.nc'
    #in1 = base + '/output/co2_V7.1.2.1_0001.nc'
    #in1 = base + '/output/co2_V7.1.2.1_0002.nc'
    #in2 = base + '/input/inputV5/co2_grd.dat'
    #in2 = base + '/input/inputV7/co2_grd.dat'

    #base = '/data/medusa/pica/models/FVCOM/runCO2_leak'
    # Low Fine
    #in1 = base + '/output/low_rate/co2_S7_run_0001.nc'
    # High Fine
    #in1 = base + '/output/high_rate/co2_S7_run_0001.nc'
    # Fine grid
    #in2 = base + '/input/configs/inputV7/co2_grd.dat'

    #base = '/data/medusa/pica/models/FVCOM/runCO2_leak'
    # Low Coarse
    #in1 = base + '/output/low_rate/co2_S5_run_0001.nc'
    # High Coarse
    #in1 = base + '/output/high_rate/co2_S5_run_0001.nc'
    # Coarse grid
    #in2 = base + '/input/configs/inputV5/co2_grd.dat'

    base = '/data/medusa/pica/models/FVCOM/runCO2_leak'
    # Coarse
    in1 = base + '/output/rate_ranges/11days/co2_S5_high_run_0001.nc'
    # Coarse grid
    in2 = base + '/input/configs/inputV5/co2_grd.dat'
    # Fine
    #in1 = base + '/output/rate_ranges/11days/co2_S7_0.000001_run_0001.nc'
    # Fine grid
    #in2 = base + '/input/configs/inputV7/co2_grd.dat'

    # Currently running
    #base = '/data/medusa/pica/models/FVCOM/runCO2_leak'
    #in1 = base + '/output/high_rate/co2_S1_0001.nc'
    #in2 = base + '/input/configs/inputV5/co2_grd.dat'
    #in2 = base + '/input/configs/inputV7/co2_grd.dat'

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

    if noisy:
        # Some basic info
        print 'Model result:\t%s' % in1
        print 'Input grid:\t%s' % in2
        # Some more complex info
        try:
            (tt, ll, xx) = np.shape(FVCOM['DYE'])
            print 'Time steps:\t%i (%.2f days) \nLayers:\t\t%i\nElements:\t%i' % (tt, (tt*dt)/86400.0, ll, xx)
        except KeyError:
            print 'Key \'DYE\' not found in FVCOM'

    if True:
        # This has been split off into CO2_budget.py to analyse multiple files
        # at once. 

        # Do total CO2 analysis
        totalCO2inSystem = calculateTotalCO2(FVCOM, 'DYE', startIdx, layerIdx, leakIdx, dt, noisy)

        # Calculate the total CO2 in the system using Riqui's algorithm
        allVolumes = unstructuredGridVolume(FVCOM)
        startDay = (5*24)
        CO2, CO2Leak, maxCO2 = CO2LeakBudget(FVCOM, leakIdx, startDay)

        if noisy:
            print 'Input at cell %i:\t\t%.4f' % (leakIdx, FVCOM['DYE'][startIdx+1,0,leakIdx])
            print 'Maximum CO2 in the system:\t%.2e' % maxCO2

        # Get the concentration for the model
        concZ = FVCOM['DYE']/allVolumes
        # Get the total concentration at n=72 (i.e. 24 hours after DYE release)
        dayConcZ = np.sum(concZ[np.r_[0:25]+startDay,:,:])
        # Scale the DYE by the volumes
        scaledZ = FVCOM['DYE']*allVolumes

        sumZ = np.sum(scaledZ, axis=1)
        totalZ = np.sum(sumZ, axis=1)
        if noisy:
            print 'Total DYE at day %i:\t\t%.2f' % (startDay, totalZ[startDay])
        #plt.figure()
        #plt.plot(FVCOM['time'], totalZ, '-x')

    # Animate some variable (ipython only)
    addVectors = False
    animateModelOutput(FVCOM, 'DYE', startIdx, skipIdx, layerIdx, addVectors, noisy)

    # Static figure
    #gp.plotUnstructuredGrid(triangles, nodes, FVCOM['x'], FVCOM['y'], np.squeeze(Z[47,:]), '')

