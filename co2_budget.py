#!/usr/bin/env python

# Stardard library
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
# Custom functions
import read_fvcom_results as rfvcom
import plot_unstruct_grid as gp
from readFVCOM import readFVCOM
from range_test_fit import calculateRegression

def coefficientOfDetermination(obs, model):
    """ Calculate the coefficient of determination for a modelled function """

    obsBar = np.mean(obs)
    modelBar = np.mean(model)

    SStot = np.sum((obs - obsBar)**2)
    SSreg = np.sum((model - obsBar)**2)
    R2 = SSreg / SStot

    return R2



if __name__ == '__main__':

    # Be verbose?
    noisy = False

    getVars = ['x', 'y', 'xc', 'yc', 'zeta', 'art1', 'h', 'time', 'TCO2', 'PH', 'DYE',         'siglev']

    # Sigma layer
    layerIdx = 0

    # Start index for each input file
    #startIdx = 770 # Riqui's
    startIdx = (5*24) # Mine

    # Skip value for animation (not zero)
    skipIdx = 5

    base = '/data/medusa/pica/models/FVCOM/runCO2_leak'

    # Get a list of files
    fileNames = glob(base + '/output/rate_ranges/11days/co2_S5_*_run_0001.nc')

    # Coarse grid
    in2 = base + '/input/configs/inputV5/co2_grd.dat'

    # Output for calculated CO2
    maxCO2 = np.zeros(np.shape(fileNames))*np.nan
    inputRate = np.zeros(np.shape(fileNames))*np.nan

    for aa, in1 in enumerate(fileNames):

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

        # Get the input amount (use the first time step at the leak site)
        inputRateCurr = FVCOM['DYE'][startIdx+1,0,leakIdx]

        dt = (FVCOM['time'][1]-FVCOM['time'][0])*60*60*24

        # Do total CO2 analysis
        totalCO2inSystem = rfvcom.calculateTotalCO2(FVCOM, 'DYE', startIdx, layerIdx, leakIdx, dt, noisy)

        # Calculate the total CO2 in the system using Riqui's algorithm
        allVolumes = rfvcom.unstructuredGridVolume(FVCOM)

        CO2, CO2Leak, maxCO2Curr = rfvcom.CO2LeakBudget(FVCOM, leakIdx, startIdx)

        # Skip out if the result is NaN
        if np.isnan(maxCO2Curr):
            continue
        else:
            maxCO2[aa] = maxCO2Curr
            inputRate[aa] = inputRateCurr

        if noisy:
            print 'Input at cell %i:\t\t%.4f' % (leakIdx, inputRate[aa])
            print 'Maximum CO2 in the system:\t%.2e' % maxCO2[aa]

        # Get the concentration for the model
        concZ = FVCOM['DYE']/allVolumes
        # Get the total concentration at n=72 (i.e. 24 hours after DYE release)
        dayConcZ = np.sum(concZ[np.r_[0:25]+startIdx,:,:])
        # Scale the DYE by the volumes
        scaledZ = FVCOM['DYE']*allVolumes

        sumZ = np.sum(scaledZ, axis=1)
        totalZ = np.sum(sumZ, axis=1)

        if noisy:                                                                              
            print 'Total DYE at day %i:\t\t%.2f' % (startDay, totalZ[startDay])
        #plt.figure()
        #plt.plot(FVCOM['time'], totalZ, '-x')


# Remove NaNs and reorder the results by the inputRate
resultsArray = np.transpose(np.array([inputRate, maxCO2]))
resultsArray = np.ma.masked_array(resultsArray,np.isnan(resultsArray))
order = resultsArray[:, 0].argsort()
sortedData = np.take(resultsArray, order, 0)

# Calculate a regression, omitting the largest synthetic inputs which
# are the least reliable
maxInput = 100
inputIdx = sortedData[:,0] <= maxInput
linX, linY, m, c, r = calculateRegression(sortedData[inputIdx,0],
                                          sortedData[inputIdx,1],
                                          'lin0')

if np.isnan(r):
    # We don't have a correlation coefficient, so calculate one
    r = np.sqrt(rfvcom.coefficientOfDetermination(sortedData[inputIdx,1], linY))

# What's the equation of the line?
print 'y = %sx + %s, r = %s' % (m[0], c, r)


    # Make a pretty picture
plt.figure()
plt.plot(sortedData[:,0], sortedData[:,1],'g-x', label='Data')
plt.plot(linX, linY, 'r-+', label='Linear regression')
plt.xlabel('Input rate')
plt.ylabel('Total CO2 in domain')
plt.legend(loc=2, frameon=False)



