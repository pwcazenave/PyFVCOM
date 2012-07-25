import numpy as np
import matplotlib.pyplot as plt

import plot_unstruct_grid as gp
from read_fvcom_results import readFVCOM, animateModelOutput

""" Read the shelf model outputs and make pretty pictures """

if __name__ == '__main__':

    base = '/data/medusa/pica/models/FVCOM/shelf/'
    in1 = base + 'output/tides/ukerc_v8_0001.nc'
    in2 = base + 'input/configs/ukerc_v8/ukerc_v8_grd.dat'
    #in1 = base + 'output/tides/severn_test_1_0001.nc'
    #in2 = base + 'input/configs/severn_test_1/severn_test_1_grd.dat'

    getVars = ['x', 'y', 'lat','lon', 'xc', 'yc', 'zeta', 'art1', 'h', 'time', 'siglev']

    noisy = True

    FVCOM = readFVCOM(in1, getVars, noisy)
    [triangles, nodes, x, y, z] = gp.parseUnstructuredGridFVCOM(in2)

    # Sigma layer
    layerIdx = 0

    startIdx = 0

    # Skip value for animation (not zero)
    skipIdx = 1

    dt = (FVCOM['time'][1]-FVCOM['time'][0])*60*60*24

    if noisy:
        # Some basic info
        print 'Model result:\t%s' % in1
        print 'Input grid:\t%s' % in2
        # Some more complex info
        try:
            try:
                (tt, ll, xx) = np.shape(FVCOM['zeta'])
            except:
                (tt, xx) = np.shape(FVCOM['zeta'])
                ll = 0

            print 'Time steps:\t%i (%.2f days) \nLayers:\t\t%i\nElements:\t%i' % (tt, (tt*dt)/86400.0, ll,   xx)
        except KeyError:
            print 'Key \'zeta\' not found in FVCOM'

    # Animate some variable (ipython only)
    addVectors = False
    animateModelOutput(FVCOM, 'zeta', startIdx, skipIdx, layerIdx, in2, addVectors, noisy)
